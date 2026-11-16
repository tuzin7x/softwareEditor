"""Composição em fluxo (PT5): cada quadro é composto, verificado e codificado assim que o worker
do Blender o anuncia (evento ``frame_ready``), enquanto a GPU ainda renderiza os seguintes.

- Cada EXR novo vira uma tarefa num conjunto de threads (NumPy, SciPy e OpenEXR soltam o GIL nas
  contas pesadas): leitura do recorte, composição só no recorte de influência
  (``Compositor.composite`` com ``origin``) e a verificação independente de ``verify.py``.
- Quadros reaproveitados (mesma pose do anterior) usam o quadro já composto, sem compor de novo;
  quadros só-foto são a própria placa.
- Uma thread escritora entrega os quadros ao FFmpeg na ordem do vídeo, só depois de a verificação
  do quadro passar, e apaga cada EXR depois do último quadro que o usa. Se um quadro falhar, o
  trabalho para ali e os EXR ainda não compostos ficam no disco para diagnóstico.
- A placa (ruído, nitidez e grão da foto) é estimada logo no início, durante o render.
- O ajuste do CG da calibração da luz (mapa de ganho, curva de tom, joelho: ``grade.py``) chega no
  evento ``calibration`` do worker, antes do primeiro quadro (``set_calibration``); o ``recompor``
  o lê do ``frames.json``. Cada quadro usa o ajuste vigente quando foi entregue.
- A sombra de contato (PT10, ``contact.py``) usa o modelo das silhuetas de baixo (evento ``contact``
  do worker ou ``frames.json["contact"]``, ``set_contact``) e as poses de cada quadro
  (``entry["contato"]``).
- Render em camadas (PT6): um quadro com ``entry["base"]`` (o marco estático: ``source``, ``border``,
  ``contato``) compõe a camada dinâmica sobre o marco já composto (``LayerAccumulator``); cada marco
  é composto uma vez e guardado até o fim. Com ``entry["transicao"]`` (troca de marco), o quadro é a
  mistura da composição nova com a antiga (``blend_frames``, peso ``peso`` da antiga). A
  verificação usa a união das máscaras de influência de todas as camadas do quadro. Um EXR é
  apagado quando nenhum quadro ainda por vir pode usá-lo (os marcos, no fim).
- No fim, a verificação do MP4 decodifica só o início, o meio e o fim, por *seek*.

O mesmo caminho serve ao ``recompor`` (todos os quadros entregues de uma vez, sem render).
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from ..video.encoder import EncodeSettings, VideoWriter, psnr, seek_frames
from .compositor import (CompositeSettings, Compositor, FrameStats, LayerAccumulator, StaticBase, blend_frames,
                         load_layers, load_render_passes)
from .contact import ContactModel
from .grade import CGGrade
from .verify import MARGIN_PX, TILE_PX, influence_mask, outside_mask_equal


class PreservationError(RuntimeError):
    """Um quadro mudou bytes da foto fora da máscara de influência dos móveis."""


def default_workers(streaming: bool) -> int:
    """Threads de composição: durante o render o Blender também usa a CPU (sincronização,
    gravação dos EXR, denoise quando não há GPU), então o fluxo usa metade dos núcleos."""
    n = os.cpu_count() or 2
    return max(1, min(4, n // 2 if streaming else n))


def _key(entry: dict) -> tuple:
    b = entry.get("border")
    return (int(entry["frame"]), entry.get("source"), tuple(int(v) for v in b) if b else None,
            bool(entry.get("reused")))


def entry_files(entry: dict) -> list[str]:
    """EXR que a composição do quadro lê: o da camada, o do marco e os da transição."""
    out = [entry["source"]]
    b = entry.get("base")
    if b:
        out.append(b["source"])
    t = entry.get("transicao")
    if t:
        out.append(t["source"])
        if t.get("base"):
            out.append(t["base"]["source"])
    return list(dict.fromkeys(out))


def _ckey(entry: dict):
    """Chave da composição do quadro: o EXR (quadro único/marco) ou camada + marco + transição."""
    b, t = entry.get("base"), entry.get("transicao")
    if not b and not t:
        return entry["source"]
    return (entry["source"], b["source"] if b else None,
            (t["source"], t["base"]["source"] if t.get("base") else None, float(t["peso"])) if t else None)


def _is_milestone(entry: dict) -> bool:
    """Quadro parado do render em camadas: o próprio marco (composto uma vez, base dos dinâmicos)."""
    return entry.get("camada") in ("marco", "final") and not entry.get("base") and not entry.get("transicao")


def _canon(obj) -> str:
    """Forma canônica (JSON) para comparar o ajuste do CG do evento com o do ``frames.json``
    (um NaN vira o mesmo texto dos dois lados)."""
    return json.dumps(obj or None, sort_keys=True)


def _area(box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1]) if box else 0


@dataclass
class _Item:
    pos: int                    # posição no vídeo (1 = primeiro quadro)
    entry: dict
    kind: str                   # "foto" | "novo" | "cache"
    src: str | None = None
    fut: Future | None = None


class StreamComposer:
    """Compõe e codifica em ordem os quadros entregues por ``push`` (na ordem do vídeo).

    ``last_use`` (EXR → última posição que o usa) vale quando todos os quadros já são conhecidos
    (``recompor``); sem ele (fluxo do render), um EXR deixa de ser usado quando chega um quadro com
    outro EXR, porque o worker só reaproveita o quadro renderizado mais recente. ``sequential``
    liga o modo do render incremental (camadas acumuladas em ordem, uma thread)."""

    def __init__(self, work: str | Path, plate8: np.ndarray, out_path: str | Path, fps: int, seed: int, em,
                 *, codec: str = "libx264", delete_exr: bool = False, workers: int | None = None,
                 sequential: bool = False, n_frames: int | None = None, last_use: dict | None = None,
                 streaming: bool = False, calibration: dict | None = None, contact: dict | None = None):
        self.t0 = time.perf_counter()
        self.work = Path(work)
        self.frames_dir = self.work / "render" / "frames"
        # placa imutável também na memória: o compositor lê este mesmo buffer, a verificação o usa
        # como referência e os quadros só-foto são ele. Uma escrita nele (bug) mudaria os três
        # juntos sem nenhuma conferência perceber; só de leitura, ela vira erro na hora.
        self.plate8 = np.array(plate8, copy=True)        # o Compositor ainda exige uint8
        self.plate8.flags.writeable = False
        self.H, self.W = plate8.shape[:2]
        self.fps = int(fps)
        self.em = em
        self.codec = codec
        self.delete_exr = bool(delete_exr)
        self.streaming = bool(streaming)
        self.sequential = bool(sequential)
        self.workers = 1 if self.sequential else int(workers or default_workers(self.streaming))
        self.n_hint = int(n_frames) if n_frames else None
        self.last_use = dict(last_use) if last_use is not None else None
        diag = float(np.hypot(self.W, self.H))
        self.shadow_feather = max(16, int(0.04 * diag))
        # raio da máscara de influência da verificação: o da regra (10% da diagonal) + folga
        self.verify_radius = max(8.0, CompositeSettings().influence_radius_frac * diag) + MARGIN_PX
        self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="compor")
        # estimativa da placa (ruído, nitidez, grão) já durante o render
        self._comp_fut = self._pool.submit(self._make_compositor, seed)
        # ajuste do CG (PT8): sem calibração, nenhum (o CG sai como renderizado)
        self._calib: dict | None = None
        self._grade_fut: Future = Future()
        self._grade_fut.set_result(None)
        # sombra de contato (PT10): sem modelo, nenhuma
        self._contact_meta: dict | None = None
        self._contact_fut: Future = Future()
        self._contact_fut.set_result(None)
        self._contact_px = 0
        self.settings = EncodeSettings(codec=codec)
        self._enc = VideoWriter(out_path, self.W, self.H, self.fps, self.settings,
                                log_path=self.work / "encode.log")
        self.out_path = Path(out_path)
        self._acc = (LayerAccumulator(self.H, self.W, feather_px=max(6, self.W // 120),
                                      shadow_feather_px=self.shadow_feather) if self.sequential else None)
        self._cond = threading.Condition()
        self._items: deque[_Item] = deque()
        self._pushed: list[dict] = []
        self._cache: dict = {}                          # chave da composição -> Future
        self._cache_files: dict = {}                    # chave -> EXR que ela lê
        self._bases: dict[str, Future] = {}             # marco (EXR) -> Future da base composta
        self._last_ref: dict[str, int] = {}             # EXR -> última posição que o usa
        self._file_futs: dict[str, list[Future]] = {}   # EXR -> composições que o leem
        self._closed_src: set[str] = set()
        self._inflight = 0
        self._max_inflight = 3 * self.workers + 2       # quadros compostos à espera da escrita
        self._closed = False
        self._error: BaseException | None = None
        self._written = 0
        self._t_finish: float | None = None
        # telemetria
        self.stats: list[FrameStats] = []
        self.per_frame: list[dict] = []
        self.previews: list[str] = []
        self._masters: dict[int, np.ndarray] = {}
        # posições potência de 2 recentes: o "meio" quando o total previsto não se confirma
        self._checkpoints: dict[int, np.ndarray] = {}
        self._last: tuple[int, dict, np.ndarray] | None = None
        self._crops: list[int] = []
        self._tot = {"leitura": 0.0, "composicao": 0.0, "verificacao": 0.0, "codificacao": 0.0,
                     "codificacao_cauda": 0.0, "previas": 0.0, "espera_placa": 0.0}
        self._t_plate = 0.0
        self._before_finish = 0
        self._exr = {"apagados": 0, "bytes": 0}
        self._writer = threading.Thread(target=self._write_loop, name="compor-escrita", daemon=True)
        self._writer.start()
        if calibration is not None:
            self.set_calibration(calibration)
        if contact is not None:
            self.set_contact(contact)
        self.t_setup = time.perf_counter() - self.t0

    # ------------------------------------------------------------------ entrada
    def set_calibration(self, calib: dict | None) -> None:
        """Ajuste do CG da calibração da luz (``frames.json["calibration"]`` ou o evento
        ``calibration`` do worker). Vale para os quadros entregues depois desta chamada; o mapa de
        ganho no tamanho do quadro é montado numa thread."""
        with self._cond:
            self._calib = dict(calib) if calib else None
            self._grade_fut = self._pool.submit(CGGrade.from_calibration, self._calib, (self.H, self.W))

    @property
    def calibration(self) -> dict | None:
        return self._calib

    def set_contact(self, meta: dict | None) -> None:
        """Modelo da sombra de contato (``frames.json["contact"]`` ou o evento ``contact`` do
        worker): vale para os quadros entregues depois desta chamada."""
        with self._cond:
            self._contact_meta = dict(meta) if meta else None
            self._contact_fut = self._pool.submit(ContactModel.from_meta, self._contact_meta, (self.H, self.W))

    def push(self, entry: dict) -> None:
        """Entrega o próximo quadro do vídeo (entrada do ``frames.json``). Bloqueia quando há
        quadros demais compostos à espera do codificador (memória limitada)."""
        with self._cond:
            self._raise_if_failed()
            pos = len(self._pushed) + 1
            self._pushed.append(entry)
            src = entry.get("source")
            if src is None:
                item = _Item(pos, entry, "foto")
                if self.sequential:
                    self._pool.submit(self._acc.reset)        # mesma ordem das tarefas (uma thread)
            else:
                files = entry_files(entry)
                if self.last_use is None:
                    # o worker só reaproveita o render do quadro anterior: um EXR que este quadro
                    # não lê não volta mais (os marcos ficam até o fim)
                    self._closed_src.update(s for s in self._last_ref if s not in files and s not in self._bases)
                key = _ckey(entry)
                fut = self._cache.get(key)
                if fut is not None:
                    item = _Item(pos, entry, "cache", src, fut)
                else:
                    while self._inflight >= self._max_inflight and self._error is None:
                        self._cond.wait()
                    self._raise_if_failed()
                    if _is_milestone(entry) and not self.sequential:
                        fut = self._base_future(entry)
                    else:
                        base_fut = self._base_future(entry["base"]) if entry.get("base") else None
                        tr = entry.get("transicao")
                        old_fut = self._base_future(tr["base"]) if tr and tr.get("base") else None
                        # transição para um marco: a composição nova é o próprio marco (já composto)
                        self_fut = (self._base_future(entry) if tr and not entry.get("base")
                                    and entry.get("camada") in ("marco", "final") else None)
                        fut = self._pool.submit(self._compose_source, entry, self._grade_fut, self._contact_fut,
                                                base_fut, old_fut, self_fut)
                    self._cache[key] = fut
                    self._cache_files[key] = files
                    self._inflight += 1
                    item = _Item(pos, entry, "novo", src, fut)
                for fl in files:
                    self._last_ref[fl] = pos
                    futs = self._file_futs.setdefault(fl, [])
                    if not futs or futs[-1] is not fut:
                        futs.append(fut)
            self._items.append(item)
            self._cond.notify_all()

    # ------------------------------------------------------------------ tarefas (threads)
    def _make_compositor(self, seed: int) -> Compositor:
        t = time.perf_counter()
        comp = Compositor(self.plate8, seed=seed)
        self._t_plate = time.perf_counter() - t
        return comp

    def _base_future(self, desc: dict) -> Future:
        """Composição do marco ``desc`` (``source``, ``border``, ``contato``) sobre a foto, uma vez
        por marco (chamar com o lock). É também o quadro parado que mostra o marco."""
        src = desc["source"]
        fut = self._bases.get(src)
        if fut is None:
            fut = self._pool.submit(self._compose_base, dict(desc), self._grade_fut, self._contact_fut)
            self._bases[src] = fut
        return fut

    def _layer(self, comp: Compositor, frame: int, desc: dict, grade, model, base=None):
        """Lê e compõe uma camada (sobre a foto ou sobre o marco ``base``): (out8, st, alfa, origem,
        máscara de influência, s de leitura, s de composição)."""
        t1 = time.perf_counter()
        contact = model.frame(desc.get("contato")) if model is not None else None
        rgba, sh, origin = load_layers(self.frames_dir / desc["source"], desc.get("border"), (self.H, self.W),
                                       self.shadow_feather)
        t2 = time.perf_counter()
        out8, st = LayerAccumulator.compose_dynamic(comp, frame, base, rgba, sh, origin, grade, contact)
        t3 = time.perf_counter()
        mask = influence_mask(rgba[..., 3], origin, (self.H, self.W), self.verify_radius)
        return out8, st, rgba[..., 3], origin, mask, t2 - t1, t3 - t2

    def _compose_base(self, desc: dict, grade_fut: Future, contact_fut: Future) -> dict:
        t0 = time.perf_counter()
        comp = self._comp_fut.result()
        grade = grade_fut.result()
        model = contact_fut.result()
        t1 = time.perf_counter()
        frame = int(desc.get("frame", 0))
        out8, st, alpha, origin, mask, t_read, t_comp = self._layer(comp, frame, desc, grade, model)
        out8.flags.writeable = False
        base = StaticBase(out8, st, np.array(alpha, np.float32, copy=True), origin, desc["source"])
        t3 = time.perf_counter()
        ok, n_bad, d_max = outside_mask_equal(out8, self.plate8, mask)
        t4 = time.perf_counter()
        return {"out8": out8, "st": st, "ok": bool(ok and st.preserved_exact), "fora_mascara_px": n_bad,
                "fora_mascara_max": d_max, "leitura_s": t_read, "composicao_s": t_comp,
                "verificacao_s": t4 - t3, "espera_placa_s": t1 - t0, "base": base, "mask": mask}

    def _compose_layered(self, entry: dict, comp: Compositor, grade, model, base_fut: Future | None,
                         old_fut: Future | None, self_fut: Future | None = None) -> dict:
        """Quadro do render em camadas: camada dinâmica sobre o marco (e a transição, se houver).
        ``self_fut``: o quadro é um marco já composto (só a transição é calculada aqui)."""
        t_read = t_comp = 0.0
        frame = int(entry["frame"])
        masks = []
        bases = []
        for bf in (base_fut, old_fut):
            if bf is None:
                bases.append(None)
                continue
            br = bf.result()
            if not br["ok"]:
                raise PreservationError(f"O marco {br['base'].source} mudou bytes da foto fora da máscara "
                                        f"({br['fora_mascara_px']} pixel(s)).")
            bases.append(br["base"])
            masks.append(br["mask"])
        if self_fut is not None:
            sr = self_fut.result()
            out8, st, mask = sr["out8"], replace(sr["st"], frame=frame), sr["mask"]
            if not sr["ok"]:
                raise PreservationError(f"O marco {entry['source']} mudou bytes da foto fora da máscara "
                                        f"({sr['fora_mascara_px']} pixel(s)).")
        else:
            out8, st, _, _, mask, tr, tc = self._layer(comp, frame, entry, grade, model, bases[0])
            t_read, t_comp = t_read + tr, t_comp + tc
        masks.append(mask)
        exact = st.preserved_exact
        tr_desc = entry.get("transicao")
        if tr_desc:
            old8, st_old, _, _, mask_old, tr, tc = self._layer(comp, frame, tr_desc, grade, model, bases[1])
            masks.append(mask_old)
            t_read, t_comp = t_read + tr, t_comp + tc
            tb = time.perf_counter()
            out8 = blend_frames(out8, old8, float(tr_desc["peso"]))
            t_comp += time.perf_counter() - tb
            exact = exact and st_old.preserved_exact
            st.object_pixels = max(st.object_pixels, st_old.object_pixels)
            st.contact_pixels += st_old.contact_pixels
        t3 = time.perf_counter()
        union = masks[0]
        for m in masks[1:]:
            union = union | m
        ok, n_bad, d_max = outside_mask_equal(out8, self.plate8, union)
        neq = out8 != self.plate8
        st.changed_pixels = int(np.count_nonzero(neq[..., 0] | neq[..., 1] | neq[..., 2]))
        st.preserved_exact = bool(exact)
        t4 = time.perf_counter()
        return {"out8": out8, "st": st, "ok": bool(ok and exact), "fora_mascara_px": n_bad,
                "fora_mascara_max": d_max, "leitura_s": t_read, "composicao_s": t_comp,
                "verificacao_s": t4 - t3}

    def _compose_source(self, entry: dict, grade_fut: Future, contact_fut: Future | None = None,
                        base_fut: Future | None = None, old_fut: Future | None = None,
                        self_fut: Future | None = None) -> dict:
        t0 = time.perf_counter()
        comp = self._comp_fut.result()
        grade = grade_fut.result()
        model = contact_fut.result() if contact_fut is not None else None
        t1 = time.perf_counter()
        if entry.get("base") or entry.get("transicao"):
            res = self._compose_layered(entry, comp, grade, model, base_fut, old_fut, self_fut)
            res["espera_placa_s"] = t1 - t0
            return res
        contact = model.frame(entry.get("contato")) if model is not None else None
        path = self.frames_dir / entry["source"]
        border = entry.get("border")
        if self.sequential:
            rgba, sh = load_render_passes(path, border, (self.H, self.W))
            t2 = time.perf_counter()
            self._acc.update(rgba, sh, border, entry.get("mode", "full"))
            out8, st = comp.composite(entry["frame"], self._acc.rgba, self._acc.shadow, None, grade=grade,
                                      contact=contact)
            alpha, origin = self._acc.rgba[..., 3], (0, 0)
        else:
            rgba, sh, origin = load_layers(path, border, (self.H, self.W), self.shadow_feather)
            t2 = time.perf_counter()
            out8, st = comp.composite(entry["frame"], rgba, sh, None, origin=origin, grade=grade, contact=contact)
            alpha = rgba[..., 3]
        t3 = time.perf_counter()
        # verificação independente: fora da máscara de influência, os bytes da foto
        mask = influence_mask(alpha, origin, (self.H, self.W), self.verify_radius)
        ok, n_bad, d_max = outside_mask_equal(out8, self.plate8, mask)
        t4 = time.perf_counter()
        return {"out8": out8, "st": st, "ok": bool(ok and st.preserved_exact), "fora_mascara_px": n_bad,
                "fora_mascara_max": d_max, "leitura_s": t2 - t1, "composicao_s": t3 - t2,
                "verificacao_s": t4 - t3, "espera_placa_s": t1 - t0}

    # ------------------------------------------------------------------ escrita em ordem
    def _write_loop(self) -> None:
        try:
            while True:
                with self._cond:
                    while not self._items and not self._closed and self._error is None:
                        self._cond.wait()
                    if self._error is not None or not self._items:
                        return
                    it = self._items[0]
                self._write_item(it)
                with self._cond:
                    self._items.popleft()
                    self._written = it.pos
                    if it.kind == "novo":
                        self._inflight -= 1
                    done = self._sources_done()
                    self._cond.notify_all()
                self._release(done)
        except BaseException as e:           # noqa: BLE001 — repassada em push/finish
            self._fail(e)

    def _write_item(self, it: _Item) -> None:
        fr = it.entry
        res = None
        if it.kind == "foto":
            out8, st = self.plate8, FrameStats(int(fr["frame"]))
        else:
            res = it.fut.result()            # exceção da tarefa sobe aqui
            if not res["ok"]:
                st = res["st"]
                raise PreservationError(
                    f"Violação da preservação de pixels no quadro {fr['frame']}: {res['fora_mascara_px']} "
                    f"pixel(s) fora da máscara de influência mudaram (diferença máx. {res['fora_mascara_max']}; "
                    f"fora de móvel+sombra: {st.max_diff_outside})")
            out8 = res["out8"]
            st = res["st"] if it.kind == "novo" else replace(res["st"], frame=int(fr["frame"]))
        te = time.perf_counter()
        self._enc.write(out8)
        t_enc = time.perf_counter() - te
        tail = self._t_finish is not None
        if not tail:
            self._before_finish += 1
        self.stats.append(st)
        tp = time.perf_counter()
        n = self.n_hint
        if it.pos in self._sample(n):
            self._masters[it.pos] = out8
        if it.pos & (it.pos - 1) == 0:
            self._checkpoints[it.pos] = out8
        for q in [q for q in self._checkpoints if 4 * q < it.pos]:    # longe demais do meio
            del self._checkpoints[q]
        self._last = (it.pos, fr, out8)
        step = max(1, n // 8) if n else 24
        if (it.pos - 1) % step == 0:
            self._save_preview(fr, out8)
        if (not self.streaming or tail) and it.pos % 10 == 1 and n:
            self.em.stage("composicao", it.pos / n, f"Quadro {it.pos}/{n}")
        t_prev = time.perf_counter() - tp
        if fr.get("source") and fr.get("border"):
            b = fr["border"]
            self._crops.append(max(0, b[2] - b[0]) * max(0, b[3] - b[1]))
        new = it.kind == "novo"
        rec = {"quadro": int(fr["frame"]),
               "leitura_s": round(res["leitura_s"], 4) if new else 0.0,
               "composicao_s": round(res["composicao_s"], 4) if new else 0.0,
               "verificacao_s": round(res["verificacao_s"], 4) if new else 0.0,
               "codificacao_s": round(t_enc, 4), "reutilizado": bool(fr.get("reused")),
               "cache": it.kind == "cache", "so_foto": fr.get("source") is None,
               "recorte_composicao_px": _area(st.box) if new else 0}
        self.per_frame.append(rec)
        if new:
            self._contact_px += int(getattr(st, "contact_pixels", 0) or 0)
            self._tot["leitura"] += res["leitura_s"]
            self._tot["composicao"] += res["composicao_s"]
            self._tot["verificacao"] += res["verificacao_s"]
            self._tot["espera_placa"] += res["espera_placa_s"]
        self._tot["codificacao"] += t_enc
        if tail:
            self._tot["codificacao_cauda"] += t_enc
        self._tot["previas"] += t_prev

    @staticmethod
    def _sample(n: int | None) -> set[int]:
        return {1, max(1, n // 2), n} if n else {1}

    def _save_preview(self, fr: dict, out8: np.ndarray) -> None:
        d = self.work / "quadros"
        d.mkdir(exist_ok=True)
        pth = d / f"quadro_{int(fr['frame']):05d}.jpg"
        Image.fromarray(out8).save(pth, quality=92)
        self.previews.append(str(pth))
        self.em.preview(pth)

    # ------------------------------------------------------------------ EXR fora de uso
    def _sources_done(self) -> list[tuple[str, list[Future]]]:
        """EXR cujo último quadro já foi escrito e que nenhum quadro por vir lê (chamar com o lock).
        As composições guardadas que liam um deles saem do cache."""
        done = []
        for src in list(self._last_ref):
            last = self._last_ref[src]
            if self.last_use is not None:
                closed = self._written >= self.last_use.get(src, last)
            else:
                closed = (src in self._closed_src or self._closed) and self._written >= last
            if closed:
                done.append((src, self._file_futs.pop(src, [])))
                del self._last_ref[src]
                self._closed_src.discard(src)
                for key in [k for k, fl in self._cache_files.items() if src in fl]:
                    self._cache.pop(key, None)
                    self._cache_files.pop(key, None)
        return done

    def _release(self, done: list[tuple[str, list[Future]]]) -> None:
        if not self.delete_exr:
            return
        for src, futs in done:
            if any(not f.done() or f.exception() is not None or not f.result()["ok"] for f in futs):
                continue
            p = self.frames_dir / src
            try:
                size = p.stat().st_size
                p.unlink()
                self._exr["bytes"] += size          # só conta o que saiu mesmo do disco
                self._exr["apagados"] += 1
            except OSError as e:          # arquivo preso (antivírus): só deixa de liberar o disco
                self.em.log(f"Não foi possível apagar {p.name}: {e}", "warning")

    # ------------------------------------------------------------------ falhas
    def _fail(self, e: BaseException) -> None:
        with self._cond:
            if self._error is None:
                self._error = e
            self._cond.notify_all()

    def _raise_if_failed(self) -> None:
        e = self._error
        if e is None:
            return
        if isinstance(e, (PreservationError, FileNotFoundError)):
            raise e
        raise RuntimeError(f"A composição dos quadros falhou: {e}") from e

    def abort(self) -> None:
        """Encerra tudo sem gerar o MP4 (erro no render ou na composição)."""
        with self._cond:
            if self._error is None:
                self._error = RuntimeError("composição cancelada")
            self._closed = True
            self._cond.notify_all()
        self._enc.abort()                    # destrava a escrita se o FFmpeg estiver cheio
        self._writer.join(timeout=60)
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ fim
    def finish(self, render_meta: dict | None = None) -> dict:
        """Fim da entrada: compõe o que falta, fecha o MP4 e verifica 3 quadros por *seek*.

        Com ``render_meta`` (o ``frames.json``), confere que os quadros anunciados durante o render
        são exatamente os do arquivo e entrega os que faltarem (worker sem eventos por quadro)."""
        self._t_finish = time.perf_counter()
        tail0 = self._t_finish if self.streaming else self.t0
        self._raise_if_failed()
        if render_meta is not None:
            frames = render_meta.get("frames") or []
            with self._cond:
                pushed = list(self._pushed)
            if len(pushed) > len(frames) or any(_key(a) != _key(b) for a, b in zip(pushed, frames)):
                raise RuntimeError("Os quadros anunciados pelo worker do Blender não batem com o frames.json.")
            cal_meta = render_meta.get("calibration")
            if cal_meta and _canon(cal_meta.get("ajuste_cg")) != _canon((self._calib or {}).get("ajuste_cg")):
                if any(fr.get("source") for fr in pushed):
                    raise RuntimeError("A calibração do frames.json não é a que o worker anunciou antes dos quadros.")
                self.set_calibration(cal_meta)        # worker sem o evento: vale o arquivo
            con_meta = render_meta.get("contact")
            if con_meta and _canon(con_meta) != _canon(self._contact_meta):
                if any(fr.get("source") for fr in pushed):
                    raise RuntimeError("A sombra de contato do frames.json não é a que o worker anunciou antes dos quadros.")
                self.set_contact(con_meta)
            for fr in frames[len(pushed):]:
                self.push(fr)
        n = len(self._pushed)
        if n != self.n_hint:                 # p. ex. render só de alguns quadros: amostras pelo real
            self.n_hint = n
        if self.streaming:
            self.em.stage("composicao", self._written / max(1, n), "Compondo os últimos quadros")
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._writer.join()
        self._raise_if_failed()
        with self._cond:
            done = self._sources_done()
        self._release(done)
        if self._last is not None and str(self.work / "quadros" / f"quadro_{int(self._last[1]['frame']):05d}.jpg") \
                not in self.previews:
            self._save_preview(self._last[1], self._last[2])
        if self._last is not None:
            self._masters.setdefault(self._last[0], self._last[2])
        self.em.stage("video", 0.5, "Finalizando o MP4")
        self._pool.shutdown(wait=True)
        t_drained = time.perf_counter()
        tc = time.perf_counter()
        self._enc.close()
        t_close = time.perf_counter() - tc
        t_done = time.perf_counter()

        # ---- verificação do encode: só início, meio e fim, cada um por seek
        tv = time.perf_counter()
        mid = max(1, n // 2)
        if mid not in self._masters and n > 2:   # total diferente do previsto: o mais perto do meio
            cands = {**self._checkpoints, **self._masters}
            near = min((p for p in cands if 1 < p < n), key=lambda p: abs(p - mid), default=None)
            if near is not None:
                self._masters[near] = cands[near]
                mid = near
        sample = sorted(p for p in {1, mid, n} if p in self._masters)
        check = []
        for idx, dec in seek_frames(self.out_path, self.W, self.H, sample, self.fps):
            ref = self._masters[idx]
            untouched = (ref == self.plate8).all(-1)
            check.append({"frame": int(self._pushed[idx - 1]["frame"]), "posicao": idx,
                          "psnr_total_db": round(psnr(dec, ref), 2),
                          "psnr_fundo_db": round(psnr(dec, self.plate8, untouched), 2)})
        verify_s = time.perf_counter() - tv
        self.em.stage("video", 1.0, "Vídeo pronto")
        return self._result(tail0, t_drained, t_close, t_done, verify_s, check)

    def _grade_summary(self) -> dict:
        g = self._grade_fut.result() if self._grade_fut.done() and self._grade_fut.exception() is None else None
        if g is None:
            return {"aplicado": False}
        return {"aplicado": True, **g.describe()}

    def _contact_summary(self) -> dict:
        f = self._contact_fut
        m = f.result() if f.done() and f.exception() is None else None
        if m is None:
            return {"aplicada": False}
        return {"aplicada": True, "pecas": m.n_parts, "forca": CompositeSettings().contact_strength,
                "pixels_total": self._contact_px}

    def _result(self, tail0, t_drained, t_close, t_done, verify_s, check) -> dict:
        W, H = self.W, self.H
        per_frame = self.per_frame
        composed = [p for p in per_frame if not p["so_foto"] and not p["cache"]]
        boxes = [p["recorte_composicao_px"] for p in composed]
        enc_tail = self._tot["codificacao_cauda"] + t_close
        comp_tail = max(0.0, (t_drained - tail0) - self._tot["codificacao_cauda"])

        changed = max((s.changed_pixels for s in self.stats), default=0)
        exact_all = all(s.preserved_exact for s in self.stats)
        n_cache = sum(1 for p in per_frame if p["cache"])
        n_foto = sum(1 for p in per_frame if p["so_foto"])
        preset = "p7" if "nvenc" in self.codec else self.settings.preset
        return {
            "seconds": t_done - tail0,
            "verify_seconds": verify_s,
            # composição = cauda depois do render menos o envio ao FFmpeg; codificação = envio +
            # finalização (no fluxo, o que rodou durante o render não entra: ficou escondido)
            "tempos": {"composicao": comp_tail, "codificacao": enc_tail, "leitura": self._tot["leitura"],
                       "composicao_pura": self._tot["composicao"], "verificacao_quadros": self._tot["verificacao"],
                       "previas": self._tot["previas"], "preparo": self.t_setup, "placa": self._t_plate,
                       "finalizacao_mp4": t_close},
            "previas": self.previews,
            "composicao": {
                "s_por_quadro_medio": round(sum(p["leitura_s"] + p["composicao_s"] for p in per_frame)
                                            / max(1, len(per_frame)), 4),
                "s_por_quadro_composto_medio": round(sum(p["leitura_s"] + p["composicao_s"] for p in composed)
                                                     / max(1, len(composed)), 4),
                "recorte_medio_px": round(sum(self._crops) / len(self._crops)) if self._crops else 0,
                "cobertura_recorte_media": (round(sum(self._crops) / len(self._crops) / (W * H), 4)
                                            if self._crops else 0.0),
                "recorte_composicao_medio_px": round(sum(boxes) / len(boxes)) if boxes else 0,
                "cobertura_composicao_media": round(sum(boxes) / len(boxes) / (W * H), 4) if boxes else 0.0,
                "paralela": self.streaming,
                "trabalhadores": self.workers,
                "sequencial": self.sequential,
                "quadros_compostos": len(composed),
                "quadros_cache": n_cache,
                "quadros_so_foto": n_foto,
                "compostos_durante_render": self._before_finish if self.streaming else 0,
                "cauda_s": round(t_done - tail0, 3),
                "placa_s": round(self._t_plate, 3),
                "leitura_s": round(self._tot["leitura"], 2),
                "composicao_s": round(self._tot["composicao"], 2),
                "verificacao_s": round(self._tot["verificacao"], 2),
                "previas_s": round(self._tot["previas"], 2),
                "exr_apagados": self._exr["apagados"],
                "exr_bytes_apagados": self._exr["bytes"],
                "ajuste_cg": self._grade_summary(),
                "sombra_contato": self._contact_summary(),
                "por_quadro": per_frame,
            },
            "preservacao": {
                "quadros": len(self.stats),
                "todos_bit_exatos_fora_dos_objetos": exact_all,
                "preservado_exato_todos": exact_all,
                # todo quadro escrito passou pela comparação (os do cache são o mesmo quadro já
                # verificado; os só-foto são a própria placa)
                "quadros_verificados": len(self.stats),
                "verificacao": {"metodo": "np.array_equal(quadro[~mascara], placa[~mascara])",
                                "mascara": f"alfa do CG dilatado pelo raio de influência (blocos de {TILE_PX} px)",
                                "raio_px": round(self.verify_radius, 1), "compostos": len(composed),
                                "cache": n_cache, "so_foto": n_foto},
                "psnr_fundo_db": min((c["psnr_fundo_db"] for c in check), default=None),
                "max_pixels_alterados_em_um_quadro": changed,
                "fracao_max_alterada": round(changed / (W * H), 4),
                "quadros_identicos_a_foto": sum(1 for s in self.stats if s.changed_pixels == 0),
            },
            "encode": {"codec": self.codec, "codificador": self.codec, "preset": preset, "crf": self.settings.crf,
                       "s": round(self._tot["codificacao"] + t_close, 2), "s_cauda": round(enc_tail, 2),
                       "verificacao": check, "verificacao_metodo": "seek", "arquivo": str(self.out_path),
                       "bytes": self.out_path.stat().st_size},
        }
