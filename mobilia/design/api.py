"""API do redesenho de ambientes: ``redesign``, ``edit_area`` e ``text_to_design``.

Fluxo de ``redesign`` (todas as funções com foto, menos "Editar área"):

1. foto -> tamanho de trabalho do SDXL (proporção mantida, lado maior 1024 na GPU / 768 na CPU, múltiplos de 64);
2. análise: profundidade (MoGe-2) -> mapa de controle (disparidade normalizada, perto = claro) e, se a função trava
   a arquitetura, segmentação (OneFormer) -> máscara de janelas/portas (+ teto no "Mobiliar");
3. N variações (sementes diferentes, paleta do estilo girando) pelo mesmo pipeline em cache;
4. trava da arquitetura: as aberturas originais voltam por cima (``archlock.lock_openings``);
5. JPEGs em ``out_dir`` + ``antes.jpg`` (a foto no mesmo tamanho, para o comparador e o vídeo) + ``design.json``.

Tudo é injetável (``engine``, ``depth``, ``groups``) para testes sem modelo e para reaproveitar análises.
Eventos de progresso: ``progress({"event": "progress", "stage", "fraction", "message"})``,
``{"event": "preview", "path", "index"}`` e ``{"event": "log", "level", "message"}``.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import archlock as AL
from . import presets as P
from .engine import Generator, get_engine, pick_device

Progress = Callable[[dict], None]

STAGES = (("preparo", "Preparando a foto", 0.03), ("analise", "Analisando paredes, janelas e profundidade", 0.17),
          ("geracao", "Gerando as novas versões do ambiente", 0.75), ("final", "Finalizando", 0.05))


@dataclass
class DesignResult:
    outputs: list[str]
    before: str | None
    report: str | None
    funcao: str
    estilo: str | None
    ambiente: str
    seeds: list[int]
    size: tuple[int, int]
    timings: dict = field(default_factory=dict)
    images: list = field(default_factory=list, repr=False)

    def to_event(self) -> dict:
        return {"event": "done", "output": self.outputs[0] if self.outputs else None, "outputs": self.outputs,
                "before": self.before, "report": self.report, "funcao": self.funcao, "estilo": self.estilo,
                "ambiente": self.ambiente, "seeds": self.seeds, "size": list(self.size)}


class _Prog:
    def __init__(self, cb: Progress | None, stages=STAGES):
        self.cb = cb
        self.w = {k: w for k, _, w in stages}
        self.order = [k for k, _, _ in stages]
        self.labels = {k: lab for k, lab, _ in stages}

    def __call__(self, stage: str, local: float = 0.0, message: str | None = None) -> None:
        if self.cb is None:
            return
        done = sum(self.w[s] for s in self.order[: self.order.index(stage)])
        frac = (done + self.w[stage] * float(np.clip(local, 0, 1))) / sum(self.w.values())
        self.cb({"event": "progress", "stage": stage, "fraction": round(frac, 4),
                 "message": message or self.labels[stage]})

    def emit(self, ev: dict) -> None:
        if self.cb is not None:
            self.cb(ev)


# ====================================================================================== entrada
def _load_rgb(photo) -> np.ndarray:
    if isinstance(photo, np.ndarray):
        return np.ascontiguousarray(photo[..., :3]).astype(np.uint8)
    from ..photo import load_photo

    return load_photo(photo).rgb8


def _load_mask(mask) -> np.ndarray:
    if isinstance(mask, np.ndarray):
        return mask
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(Image.open(mask))
    if im.mode in ("RGBA", "LA"):
        return np.asarray(im.convert("RGBA"))
    return np.asarray(im.convert("L"))


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LANCZOS4
    return cv2.resize(img, (w, h), interpolation=interp)


def _seeds(seed: int | None, n: int) -> list[int]:
    base = random.randint(1, 2 ** 31 - 1) if seed is None else int(seed)
    return [(base + 7919 * i) % (2 ** 31 - 1) for i in range(max(1, int(n)))]


def _save(img: np.ndarray, path: Path) -> str:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, quality=95, subsampling=0)
    return str(path)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    return f"<{type(o).__name__}>"


def _out_dir(out_dir) -> Path:
    if out_dir:
        return Path(out_dir)
    from ..config import JOBS_DIR

    return JOBS_DIR / f"design_{time.strftime('%Y%m%d_%H%M%S')}"


# ====================================================================================== análise
def depth_control(depth: np.ndarray, size: tuple[int, int], lo_pct: float = 2.0, hi_pct: float = 98.0
                  ) -> np.ndarray:
    """Profundidade métrica (z, metros; <= 0 ou não finito = inválido) -> mapa de controle (H, W) em [0, 1] no
    formato do ControlNet depth do SDXL: disparidade normalizada, perto = claro."""
    z = np.asarray(depth, np.float32)
    valid = np.isfinite(z) & (z > 1e-4)
    if valid.sum() < 16:
        return np.zeros((size[1], size[0]), np.float32)
    disp = np.zeros_like(z)
    disp[valid] = 1.0 / z[valid]
    lo, hi = np.percentile(disp[valid], [lo_pct, hi_pct])
    c = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
    if (~valid).any():          # buracos (céu, reflexo): o mais longe
        c[~valid] = 0.0
    c = cv2.resize(c, size, interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(c, (0, 0), 0.6).astype(np.float32)


def estimate_depth(rgb: np.ndarray, device: str, long_side: int = 768) -> np.ndarray:
    """Profundidade (z) pelo MoGe-2 numa versão reduzida da foto (o controle é suave; não precisa de 4K)."""
    from ..analysis.geometry import estimate_geometry
    from ..config import GEOMETRY_MODEL_ID, GEOMETRY_MODEL_ID_SMALL

    s = min(1.0, long_side / max(rgb.shape[:2]))
    small = _resize(rgb, (int(round(rgb.shape[1] * s)), int(round(rgb.shape[0] * s))))
    gpu = device.startswith("cuda")
    geo = estimate_geometry(small, GEOMETRY_MODEL_ID if gpu else GEOMETRY_MODEL_ID_SMALL, device=device,
                            use_fp16=gpu)
    d = geo.depth.copy()
    d[~geo.mask] = 0
    return d


def estimate_groups(rgb: np.ndarray, device: str, long_side: int = 768) -> np.ndarray:
    """Grupos semânticos (OneFormer ADE20K, ``analysis/segmentation.GROUPS``) numa versão reduzida da foto."""
    from ..analysis.segmentation import segment
    from ..config import SEGMENTATION_MODEL_ID

    s = min(1.0, long_side / max(rgb.shape[:2]))
    small = _resize(rgb, (int(round(rgb.shape[1] * s)), int(round(rgb.shape[0] * s))))
    gpu = device.startswith("cuda")
    return segment(small, SEGMENTATION_MODEL_ID, device=device, use_fp16=gpu).groups


def _groups_of(groups):
    """Grupos de um ``SegmentationResult`` (recalculados dos rótulos ADE20K com o mapeamento atual, que pode ter
    mudado desde que a análise foi salva) ou o próprio array de grupos."""
    labels, names = getattr(groups, "labels", None), getattr(groups, "id2label", None)
    if labels is not None and names:
        from ..analysis.segmentation import labels_to_groups

        return labels_to_groups(labels, names)
    return getattr(groups, "groups", groups)


# ====================================================================================== redesign
def redesign(photo, funcao: str = "mudar_estilo", estilo: str | None = "moderno", ambiente: str = "sala",
             n: int = 4, seed: int | None = None, prompt: str = "", out_dir=None, quality: str = "final",
             evento: str | None = None, engine: Generator | None = None, depth: np.ndarray | None = None,
             groups=None, progress: Progress | None = None, device: str = "auto",
             overrides: dict | None = None, long_side: int | None = None, candidates: int | None = None,
             rank_fn=None) -> DesignResult:
    """Gera ``n`` versões redesenhadas do ambiente da foto.

    ``funcao``: mudar_estilo | reforma | redesign_simples | mobiliar | fachada | evento (para "Editar área" use
    :func:`edit_area`; para "Texto para ambiente", :func:`text_to_design`). ``estilo``: chave de
    ``presets.STYLES`` (``None`` só no evento). ``evento``: chave de ``presets.EVENTS`` (função "evento").
    ``depth`` (z em metros, qualquer resolução) e ``groups`` (grupos semânticos ou ``SegmentationResult``) evitam
    rodar o MoGe/OneFormer. ``overrides``: campos de ``FunctionPreset`` (strength, control_scale, control_end,
    guidance, steps) para ajuste fino. ``candidates``: quantas versões gerar antes de escolher as ``n`` melhores
    (``None`` = automático: n + 2 no "Mobiliar" com GPU, n no resto); ``rank_fn(rgb) -> grupos`` troca o
    OneFormer da escolha (testes)."""
    t_all = time.perf_counter()
    fp = P.with_overrides(P.function(funcao), **(overrides or {}))
    if fp.key in ("editar_area", "texto"):
        raise ValueError(f"Use {'edit_area' if fp.key == 'editar_area' else 'text_to_design'} para '{fp.label}'.")
    room_key = "fachada" if fp.exterior else P.norm_room(ambiente)
    st_key = P.norm_style(estilo) if (estilo and not (fp.event and estilo in P.EVENTS)) else None
    if fp.event:
        evento = P.norm_event(evento or (estilo if estilo in P.EVENTS else "aniversario"))
    elif st_key is None:
        raise ValueError("Escolha um estilo.")
    prog = _Prog(progress)
    out = _out_dir(out_dir)
    timings: dict = {}

    prog("preparo")
    rgb = _load_rgb(photo)
    eng = engine or get_engine(device)
    dev = getattr(eng, "device", pick_device(device))
    qp = P.quality_profile(quality, dev)
    W, H = P.work_size(rgb.shape[1], rgb.shape[0], long_side or qp.long_side)
    base = _resize(rgb, (W, H))
    steps = int((overrides or {}).get("steps") or (fp.steps if dev.startswith("cuda") else min(fp.steps, qp.steps)))
    if qp.key == "rapido":
        steps = min(steps, qp.steps)

    # ---- análise
    t0 = time.perf_counter()
    prog("analise", 0.0, "Medindo a profundidade do ambiente")
    if depth is None and fp.control_scale > 0:
        depth = estimate_depth(rgb, dev)
    control = depth_control(depth, (W, H)) if depth is not None else None
    lock_names, keep_names = tuple(fp.lock), tuple(fp.keep)
    lock = np.zeros((H, W), bool)
    keep = np.zeros((H, W), bool)
    lock_info: dict = {}
    if lock_names or keep_names:
        prog("analise", 0.5, "Encontrando janelas e portas")
        if groups is None:
            groups = estimate_groups(rgb, dev)
        gr = _groups_of(groups)
        if lock_names:
            lock = AL.lock_mask(gr, lock_names, (W, H))
        if keep_names:
            keep = AL.lock_mask(gr, keep_names, (W, H))
    timings["analise_s"] = round(time.perf_counter() - t0, 2)
    gen_mask = AL.generation_mask(lock | keep)

    # ---- geração (com ``candidates`` > n: gera mais e fica com as n de melhor nota, ver ``quality.py``)
    n = max(1, int(n))
    if candidates is None:
        candidates = n + 2 if (fp.key == "mobiliar" and dev.startswith("cuda")) else n
    m = max(n, int(candidates))
    seeds = _seeds(seed, m)
    prompts = [P.build_prompt(fp.key, st_key, room_key, i, prompt, evento) for i in range(m)]
    if hasattr(eng, "embeddings"):
        prog("geracao", 0.0, "Preparando o texto do estilo")
        eng.embeddings(prompts)
    tag = f"{fp.key}_{st_key or evento}"
    cand_dir = out / "candidatos" if m > n else out
    paths, images = [], []
    t_gen = time.perf_counter()
    for i, (sd, pr) in enumerate(zip(seeds, prompts)):
        def cb(k, tot, i=i):
            prog("geracao", (i + k / max(tot, 1)) / m * (0.92 if m > n else 1.0),
                 f"Gerando a versão {i + 1} de {m}")

        cb(0, 1)
        img = eng.generate(base, gen_mask, control, pr, sd, fp.strength, steps, fp.guidance, fp.control_scale,
                           fp.control_end, cb)
        img = _resize(img, (W, H))
        if lock.any():
            img, lock_info = AL.lock_openings(img, base, lock)
        path = _save(img, cand_dir / f"{tag}_{i + 1}.jpg")
        paths.append(path)
        images.append(img)
        prog.emit({"event": "preview", "path": path, "index": i})
    timings["geracao_s"] = round(time.perf_counter() - t_gen, 2)
    timings["por_imagem_s"] = round(timings["geracao_s"] / m, 2)
    notas = None
    order = list(range(m))
    if m > n:
        prog("geracao", 0.95, "Escolhendo as melhores versões")
        t0 = time.perf_counter()
        from .quality import rank

        notas = rank(images, lock | keep, dev, furnish=fp.furnish, group_fn=rank_fn)
        order = sorted(range(m), key=lambda k: -notas[k]["nota"])[:n]
        timings["escolha_s"] = round(time.perf_counter() - t0, 2)
    outputs = []
    for j, k in enumerate(order):
        outputs.append(_save(images[k], out / f"{tag}_{j + 1}.jpg") if m > n else paths[k])
    sel_seeds = [seeds[k] for k in order]
    images = [images[k] for k in order]

    prog("final")
    before = _save(base, out / "antes.jpg")
    timings["total_s"] = round(time.perf_counter() - t_all, 2)
    rep = {"funcao": fp.key, "estilo": st_key, "evento": evento, "ambiente": room_key, "tamanho": [W, H],
           "qualidade": qp.key, "dispositivo": dev, "sementes": sel_seeds, "passos": steps,
           "preset": asdict(fp), "prompts": [asdict(prompts[k]) for k in order],
           "trava": {"grupos": list(lock_names), "mantidos": list(keep_names), **lock_info}, "saidas": outputs, "antes": before, "tempos": timings,
           "candidatos": ({"sementes": seeds, "arquivos": paths, "notas": notas, "escolhidos": order}
                          if notas is not None else None),
           "motor": {"carga_s": round(getattr(eng, "load_s", 0.0), 2), "chamadas": getattr(eng, "calls", [])[-m:]}}
    report = out / "design.json"
    report.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    return DesignResult(outputs, before, str(report), fp.key, st_key or evento, room_key, sel_seeds, (W, H),
                        timings, images)


# ====================================================================================== editar área
def edit_area(photo, mask, prompt: str = "", estilo: str | None = None, ambiente: str = "sala", n: int = 4,
              seed: int | None = None, out_dir=None, quality: str = "final", engine: Generator | None = None,
              depth: np.ndarray | None = None, progress: Progress | None = None, device: str = "auto",
              overrides: dict | None = None, feather_px: float | None = None) -> DesignResult:
    """Refaz só a área pintada pelo usuário (``mask``: PNG ou array, branco = editar), guiada por ``prompt`` e/ou
    ``estilo``. A região é recortada com contexto, gerada na resolução do SDXL e colada de volta na foto ORIGINAL
    (resolução cheia) com uma rampa suave por dentro da máscara: todo pixel fora da máscara fica idêntico."""
    t_all = time.perf_counter()
    fp = P.with_overrides(P.function("editar_area"), **(overrides or {}))
    room_key = P.norm_room(ambiente)
    st_key = P.norm_style(estilo) if estilo else None
    if not (prompt or "").strip() and st_key is None:
        raise ValueError("Descreva o que colocar na área ou escolha um estilo.")
    prog = _Prog(progress)
    out = _out_dir(out_dir)
    timings: dict = {}
    prog("preparo")
    rgb = _load_rgb(photo)
    Hf, Wf = rgb.shape[:2]
    um = AL.user_mask(_load_mask(mask), (Wf, Hf))
    if not um.any():
        raise ValueError("A máscara está vazia: pinte a área que deve mudar.")
    eng = engine or get_engine(device)
    dev = getattr(eng, "device", pick_device(device))
    qp = P.quality_profile(quality, dev)
    steps = int((overrides or {}).get("steps") or (fp.steps if dev.startswith("cuda") else min(fp.steps, qp.steps)))

    x0, y0, x1, y1 = AL.mask_bbox(um, 0.25)
    crop, cm = rgb[y0:y1, x0:x1], um[y0:y1, x0:x1]
    W, H = P.work_size(x1 - x0, y1 - y0, qp.long_side)
    base = _resize(crop, (W, H))
    mw = cv2.resize(cm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    r = max(2, int(0.012 * np.hypot(W, H)))
    gen_mask = cv2.dilate(mw.astype(np.uint8), AL._disk(r)).astype(bool)

    t0 = time.perf_counter()
    prog("analise", 0.0, "Medindo a profundidade do ambiente")
    control = None
    if fp.control_scale > 0:
        if depth is None:
            depth = estimate_depth(rgb, dev)
        dfull = cv2.resize(np.asarray(depth, np.float32), (Wf, Hf), interpolation=cv2.INTER_NEAREST)
        control = depth_control(dfull[y0:y1, x0:x1], (W, H))
    timings["analise_s"] = round(time.perf_counter() - t0, 2)

    seeds = _seeds(seed, n)
    prompts = [P.build_prompt("editar_area", st_key, room_key, i, prompt) for i in range(len(seeds))]
    if hasattr(eng, "embeddings"):
        eng.embeddings(prompts)
    fr = feather_px if feather_px is not None else max(3.0, 0.006 * np.hypot(Wf, Hf))
    alpha = AL.feather_alpha(um, fr, inside=1.0)
    outputs, images = [], []
    t_gen = time.perf_counter()
    for i, (sd, pr) in enumerate(zip(seeds, prompts)):
        def cb(k, tot, i=i):
            prog("geracao", (i + k / max(tot, 1)) / len(seeds), f"Gerando a versão {i + 1} de {len(seeds)}")

        cb(0, 1)
        g = eng.generate(base, gen_mask, control, pr, sd, fp.strength, steps, fp.guidance, fp.control_scale,
                         fp.control_end, cb)
        g = _resize(_resize(g, (W, H)), (x1 - x0, y1 - y0))
        top = rgb.copy()
        top[y0:y1, x0:x1] = g
        img = AL.composite(rgb, top, alpha)
        path = _save(img, out / f"editar_area_{i + 1}.jpg")
        outputs.append(path)
        images.append(img)
        prog.emit({"event": "preview", "path": path, "index": i})
    timings["geracao_s"] = round(time.perf_counter() - t_gen, 2)
    timings["por_imagem_s"] = round(timings["geracao_s"] / len(seeds), 2)
    prog("final")
    before = _save(rgb, out / "antes.jpg")
    timings["total_s"] = round(time.perf_counter() - t_all, 2)
    rep = {"funcao": "editar_area", "estilo": st_key, "ambiente": room_key, "recorte": [x0, y0, x1, y1],
           "tamanho_geracao": [W, H], "tamanho": [Wf, Hf], "sementes": seeds, "passos": steps,
           "preset": asdict(fp), "prompts": [asdict(p) for p in prompts], "saidas": outputs, "tempos": timings}
    report = out / "design.json"
    report.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    return DesignResult(outputs, before, str(report), "editar_area", st_key, room_key, seeds, (Wf, Hf), timings,
                        images)


# ====================================================================================== texto para ambiente
ASPECTS = {"paisagem": (4, 3), "retrato": (3, 4), "quadrado": (1, 1), "16:9": (16, 9), "9:16": (9, 16)}


def text_to_design(prompt: str = "", estilo: str | None = "moderno", ambiente: str = "sala", n: int = 4,
                   seed: int | None = None, out_dir=None, quality: str = "final", aspect: str = "paisagem",
                   engine: Generator | None = None, progress: Progress | None = None, device: str = "auto",
                   overrides: dict | None = None) -> DesignResult:
    """Cria o ambiente do zero a partir da descrição + tipo de ambiente + estilo (sem foto)."""
    t_all = time.perf_counter()
    fp = P.with_overrides(P.function("texto"), **(overrides or {}))
    room_key = P.norm_room(ambiente)
    st_key = P.norm_style(estilo) if estilo else None
    prog = _Prog(progress, (("preparo", "Preparando", 0.03), ("geracao", "Criando o ambiente", 0.92),
                            ("final", "Finalizando", 0.05)))
    out = _out_dir(out_dir)
    prog("preparo")
    eng = engine or get_engine(device)
    dev = getattr(eng, "device", pick_device(device))
    qp = P.quality_profile(quality, dev)
    aw, ah = ASPECTS.get(aspect, ASPECTS["paisagem"])
    W, H = P.work_size(aw * 1000, ah * 1000, qp.long_side)
    steps = int((overrides or {}).get("steps") or (fp.steps if dev.startswith("cuda") else min(fp.steps, qp.steps)))
    blank = np.full((H, W, 3), 127, np.uint8)
    full = np.ones((H, W), bool)
    seeds = _seeds(seed, n)
    prompts = [P.build_prompt("texto", st_key, room_key, i, prompt) for i in range(len(seeds))]
    if hasattr(eng, "embeddings"):
        eng.embeddings(prompts)
    outputs, images = [], []
    t_gen = time.perf_counter()
    for i, (sd, pr) in enumerate(zip(seeds, prompts)):
        def cb(k, tot, i=i):
            prog("geracao", (i + k / max(tot, 1)) / len(seeds), f"Criando a versão {i + 1} de {len(seeds)}")

        cb(0, 1)
        img = _resize(eng.generate(blank, full, None, pr, sd, 1.0, steps, fp.guidance, 0.0, 1.0, cb), (W, H))
        path = _save(img, out / f"texto_{st_key or 'livre'}_{i + 1}.jpg")
        outputs.append(path)
        images.append(img)
        prog.emit({"event": "preview", "path": path, "index": i})
    timings = {"geracao_s": round(time.perf_counter() - t_gen, 2)}
    timings["por_imagem_s"] = round(timings["geracao_s"] / len(seeds), 2)
    timings["total_s"] = round(time.perf_counter() - t_all, 2)
    prog("final")
    rep = {"funcao": "texto", "estilo": st_key, "ambiente": room_key, "tamanho": [W, H], "sementes": seeds,
           "passos": steps, "preset": asdict(fp), "prompts": [asdict(p) for p in prompts], "saidas": outputs,
           "tempos": timings}
    report = out / "design.json"
    report.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    return DesignResult(outputs, None, str(report), "texto", st_key, room_key, seeds, (W, H), timings, images)


__all__ = ["redesign", "edit_area", "text_to_design", "DesignResult", "depth_control", "estimate_depth",
           "estimate_groups", "STAGES", "ASPECTS"]
