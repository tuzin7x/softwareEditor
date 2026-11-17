"""Modo B do pipeline (PLANO_V3 §3, PT-V3-8): foto mobiliada -> sala vazia por trás -> móveis reais entram -> foto.

Chamado por :func:`mobilia.pipeline.run_job` depois da análise (MoGe-2 + OneFormer semântico e panóptico + sala),
quando a detecção de modo (``removal.instances.detect_mode``) diz "B" ou o pedido força o modo B. Etapas:

1. **Peças** (``removal/instances.py``): instâncias de móveis da foto na resolução do vídeo (OneFormer da análise,
   reamostrado, + SAM 2.1 se os pesos estão no disco), classificadas em soltos e planejados.
2. **Sala vazia + portão de qualidade** (``removal/plate.py``, :func:`build_plate_gated`): placa crua por plano,
   com a nota automática (OneFormer rodado de novo na placa para achar móvel que "voltou"). Reprovada: tenta
   mantendo os planejados, depois LaMa direto na imagem e máscara mais larga. Ainda reprovada: o trabalho termina
   **sem MP4** e com um aviso claro (a interface oferece gerar mesmo assim, usar o modo A ou outra foto), a não ser
   que o pedido traga ``accept_weak_plate`` (o usuário escolheu gerar mesmo assim; o aviso vai para o relatório).
3. **Camadas e coreografia** (``layers2d/layers.py`` e ``choreo.py``): RGBA pré-multiplicado de cada peça, ordem de
   pintura e entradas sem fade nem parte oculta à mostra.
4. **Vídeo** (``layers2d/video.py``): composição + verificação §3.5 em 100% dos quadros (quadro 0 = placa, último =
   foto byte a byte, fora da máscara ∪ caixa móvel = foto, anti-morph) + MP4; qualquer falha aborta sem MP4.

O ``report.json`` registra o modo e o motivo, a área gerada (%), as peças (e as que ficaram no fundo), a semente,
a nota da placa com todas as tentativas do portão, as verificações e os tempos.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import __version__
from .jobs import JobRequest, JobResult, ProgressEmitter
from .telemetry import ETAPAS_B, Stopwatch, stages_sum, torch_peak_mb, torch_reset_peak, value_total

# "0" desliga o OneFormer rodado de novo na placa (nota mais fraca; só para depuração e testes rápidos)
ENV_REDETECTAR = "MOBILIA_MODO_B_REDETECTAR"
# "1" grava também ``auditoria.mkv`` (sem perdas) e confere cada quadro dele byte a byte com os mestres
ENV_AUDITORIA = "MOBILIA_MODO_B_MKV"
GEOMETRY_STRIDE = 4          # passo da geometria entregue à regra solto x planejado (como nas fixtures)
NARRATIVE_TITLE = "O QUE O 7xAi FEZ"


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "sim", "yes")


@dataclass
class ModeBContext:
    req: JobRequest
    em: ProgressEmitter
    sw: Stopwatch
    vram: Any
    hw: Any
    prof: Any
    q: dict
    work: Path
    stamp: str
    original: Any                # Photo original (dimensões para o relatório)
    photo: Any                   # Photo na resolução do vídeo (lados pares): o último quadro é ela, byte a byte
    rgb: np.ndarray              # foto na resolução da análise
    geo: Any                     # GeometryResult (resolução da análise)
    seg: Any                     # SegmentationResult (resolução da análise)
    pan: Any                     # PanopticResult (resolução da análise)
    room: Any
    decision: Any                # ModeDecision
    vram_torch: float | None = None
    first: bool = False
    state_path: Path | None = None
    warm_state: dict | None = None
    t_ref: float = 0.0
    t_click: float = 0.0


# ------------------------------------------------------------------ modelos (pontos de troca para os testes)
def load_inpainter(device: str):
    """big-lama (CUDA quando há, senão CPU; baixa e confere o SHA-256 no 1º uso)."""
    from .removal import plate as rp

    return rp.load_lama(device)


def load_redetector(device: str, use_fp16: bool):
    """OneFormer carregado uma vez para a nota da placa (``redetect(placa) -> (seg, pan)``). None = sem."""
    if not _flag(ENV_REDETECTAR, "1"):
        return None, None
    from .removal import instances as ri

    proc, model = ri.load_oneformer(device, use_fp16=use_fp16)
    return (lambda img: ri.oneformer_infer(proc, model, img, device)), model


def sam_choice(device: str):
    """SAM 2.1 para refinar as bordas: "auto" usa os pesos se estão no disco (sem baixar)."""
    return "auto"


# ------------------------------------------------------------------ reamostragem da análise para o vídeo
def _nearest(a: np.ndarray, W: int, H: int) -> np.ndarray:
    if a.shape[:2] == (H, W):
        return a
    return np.asarray(Image.fromarray(a).resize((W, H), Image.NEAREST))


def to_video_resolution(seg, pan, geo, W: int, H: int):
    """Segmentação, panóptico e geometria da análise na resolução do vídeo (vizinho mais próximo)."""
    import cv2

    from .analysis.segmentation import SegmentationResult
    from .removal.instances import PanopticResult

    seg_v = SegmentationResult(_nearest(seg.labels, W, H), _nearest(seg.groups, W, H), seg.id2label, seg.model_id)
    ids = _nearest(pan.ids, W, H)
    rows = np.array(pan.segments, np.float64).reshape(-1, 4).copy()
    for r in rows:
        r[3] = float((ids == int(r[0])).sum())
    pan_v = PanopticResult(ids, rows)
    s = GEOMETRY_STRIDE
    gw, gh = -(-W // s), -(-H // s)
    pts = cv2.resize(np.asarray(geo.points, np.float32), (gw, gh), interpolation=cv2.INTER_NEAREST)
    msk = cv2.resize(np.asarray(geo.mask, np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST).astype(bool)
    geometry = {"points": pts, "mask": msk, "stride": s}
    return seg_v, pan_v, geometry


# ------------------------------------------------------------------ textos
# rótulos ADE20K das peças removíveis em português (texto da tela de resultado); o relatório guarda o original
LABELS_PT = {
    "bed": "cama", "table": "mesa", "chair": "cadeira", "sofa": "sofá", "armchair": "poltrona", "seat": "assento",
    "desk": "escrivaninha", "lamp": "luminária", "cushion": "almofada", "pillow": "travesseiro", "rug": "tapete",
    "coffee table": "mesa de centro", "swivel chair": "cadeira giratória", "bench": "banco", "ottoman": "pufe",
    "stool": "banqueta", "pool table": "mesa de sinuca", "tv": "TV", "crt screen": "TV", "computer": "computador",
    "monitor": "monitor", "box": "caixa", "plant": "planta", "flower": "flores", "vase": "vaso", "pot": "vaso",
    "basket": "cesto", "painting": "quadro", "poster": "pôster", "clock": "relógio", "bag": "bolsa",
    "trash can": "lixeira", "clothes": "roupas", "blanket": "manta", "book": "livros", "towel": "toalha",
    "bicycle": "bicicleta", "plaything": "brinquedo", "bottle": "garrafa", "sculpture": "escultura",
    "tray": "bandeja", "glass": "copo", "plate": "prato", "food": "comida", "cradle": "berço", "ball": "bola",
    "barrel": "barril", "arcade machine": "fliperama", "base": "base", "counter": "balcão",
    "countertop": "bancada", "kitchen island": "ilha", "bar": "bar", "sink": "pia", "toilet": "vaso sanitário",
    "tub": "banheira", "shower": "box", "stove": "fogão", "oven": "forno", "refrigerator": "geladeira",
    "dishwasher": "lava-louças", "microwave": "micro-ondas", "hood": "coifa", "washer": "lavadora",
    "cabinet": "armário", "shelf": "prateleira", "wardrobe": "guarda-roupa", "chest of drawers": "cômoda",
    "bookcase": "estante", "buffet": "aparador", "case": "vitrine", "mirror": "espelho",
}


def label_pt(label: str) -> str:
    from .removal.instances import base_name

    return LABELS_PT.get(base_name(label), LABELS_PT.get(label, label))


def _labels_pt(labels) -> str:
    from collections import Counter

    c = Counter(label_pt(lb) for lb in labels)
    parts = [f"{n}× {k}" if n > 1 else k for k, n in c.most_common()]
    return ", ".join(parts[:8]) + (f" e mais {len(parts) - 8}" if len(parts) > 8 else "")


def gate_message(quality, accepted: bool) -> str:
    motivos = "; ".join(quality.reasons) if quality is not None else ""
    if accepted:
        return ("Vídeo gerado a seu pedido com a sala vazia abaixo da qualidade mínima (" + motivos + "). "
                "Confira o início do vídeo antes de publicar.")
    return ("A sala vazia do início do vídeo não passou na verificação automática de qualidade (" + motivos + "). "
            "Para não entregar um vídeo ruim, nenhum MP4 foi gerado. Você pode gerar mesmo assim, usar o modo "
            "sala vazia → mobiliada (modo A) ou enviar outra foto.")


# ------------------------------------------------------------------ execução
def run_mode_b(ctx: ModeBContext) -> JobResult:
    from .layers2d import choreo as C
    from .layers2d import layers as L
    from .layers2d import video as VB
    from .removal import instances as ri
    from .removal import plate as rp

    req, em, sw, prof = ctx.req, ctx.em, ctx.sw, ctx.prof
    dev = prof.torch_device
    work = ctx.work
    foto = ctx.photo.rgb8
    H, W = foto.shape[:2]
    out_path = Path(req.output_path)
    out_path.unlink(missing_ok=True)            # nunca sobra um MP4 antigo com o nome deste trabalho

    # -------------------------------------------------------------- peças
    em.stage("pecas", 0.05, "Separando os móveis da foto, peça por peça")
    with sw.measure("pecas"):
        seg_v, pan_v, geometry = to_video_resolution(ctx.seg, ctx.pan, ctx.geo, W, H)
        with ctx.vram.track("modo_b"):
            inst = ri.detect_instances(foto, seg_v, pan_v, sam=sam_choice(dev), device=dev, geometry=geometry,
                                       room=ctx.room)
        inst.mode = ctx.decision
        if not inst.instances:
            raise ValueError("Nenhum móvel foi encontrado na foto para retirar (modo B). Use o modo sala vazia → "
                             "mobiliada (modo A) ou envie uma foto do ambiente mobiliado.")
        for w in inst.warnings:
            em.log(w, "warning")
    em.stage("pecas", 1.0, f"{len(inst.instances)} peças encontradas: {_labels_pt(i.label for i in inst.instances)}")

    # -------------------------------------------------------------- sala vazia + portão
    em.stage("sala_vazia", 0.02, "Montando a sala vazia (por trás, não aparece no vídeo)")
    attempt_no = [0]
    names = {"crua_por_plano": "sala crua, por plano", "planejados_mantidos": "mantendo os planejados",
             "lama_imagem": "preenchimento direto na imagem", "mascara_ampla": "máscara mais larga"}

    def on_attempt(name, r):
        attempt_no[0] += 1
        qq = r.quality
        nota = "" if qq is None else f": nota {qq.score:.0f}/100 ({'aprovada' if qq.ok else 'reprovada'})"
        em.stage("sala_vazia", min(0.95, 0.25 * attempt_no[0]), f"Sala vazia, tentativa {attempt_no[0]} "
                 f"({names.get(name, name)}){nota}")
        em.log(f"Portão de qualidade: tentativa \"{name}\"{nota}"
               + ("" if qq is None or qq.ok else " — " + "; ".join(qq.reasons)))

    with sw.measure("sala_vazia"):
        with ctx.vram.track("modo_b"):
            torch_reset_peak(dev)
            t0 = time.perf_counter()
            inpaint = load_inpainter(dev)
            t_load = time.perf_counter() - t0
            redetect, red_model = load_redetector(dev, bool(getattr(prof, "use_fp16", False)))
            try:
                res = rp.build_plate_gated(foto, inst, seg_v, ctx.room, ctx.geo.depth, inpaint, redetect,
                                           seed=req.seed, on_attempt=on_attempt)
            finally:
                del redetect, red_model, inpaint
                gc.collect()
                if dev.startswith("cuda"):
                    import torch

                    torch.cuda.empty_cache()
            vram_b = torch_peak_mb(dev)
        Image.fromarray(res.plate).save(work / "sala_vazia.png")
        Image.fromarray(res.plate).save(work / "sala_vazia.jpg", quality=90)
    em.preview(work / "sala_vazia.jpg")
    quality = res.quality
    ok = quality is None or quality.ok
    accepted = (not ok) and bool(req.accept_weak_plate)
    warning = None if ok else gate_message(quality, accepted)
    gate = {"aprovada": ok, "nota": None if quality is None else round(quality.score, 1),
            "estrategia": res.info.get("estrategia_portao", "crua_por_plano"),
            "tentativas": len(res.info.get("tentativas", [])),
            "motivos": [] if quality is None else list(quality.reasons),
            "reconferida_oneformer": _flag(ENV_REDETECTAR, "1"),
            "aceita_pelo_usuario": accepted, "aviso": warning}
    for w in res.warnings:
        em.log(w, "warning")
    pieces = inst.removal(res.kept_builtins)
    base = _base_report(ctx, inst, res, pieces, gate, t_load, vram_b)
    if not ok and not accepted:
        em.log(warning, "warning")
        return _finish(ctx, base, "", warning, gate)

    # -------------------------------------------------------------- camadas + coreografia
    em.stage("camadas", 0.1, "Recortando os móveis reais da foto")
    with sw.measure("camadas"):
        st = L.build_layers(foto, res, pieces, ctx.geo.depth)
        em.stage("camadas", 0.6, "Planejando a entrada de cada peça")
        ch = C.plan(st, fps=int(ctx.q.get("fps") or C.FPS))
        cv = C.verify(st, ch)
    em.stage("camadas", 1.0, f"{len(ch.entrances)} entradas em {ch.duration:.1f} s")

    # -------------------------------------------------------------- vídeo
    em.stage("video", 0.0, "Compondo, verificando cada quadro e gerando o MP4")
    audit = work / "auditoria.mkv" if _flag(ENV_AUDITORIA, "0") else None

    def progress(n, total):
        em.stage("video", n / max(1, total), f"Quadro {n} de {total} (verificado)")

    with sw.measure("video"):
        with ctx.vram.track("modo_b"):
            vrep = VB.render_mode_b(st, ch, foto, res.mask, out_path, audit_mkv=audit, backend=None,
                                    gpu=dev == "cuda", choreo_stats=cv, log_dir=work, progress=progress)
    em.stage("video", 1.0, "MP4 pronto")
    base.update(st.report())
    base.update(ch.report(st))
    base["coreografia_verificacao"] = cv
    base.update(vrep)
    base["encode"] = {"arquivo": str(out_path), "codificador": vrep["video_modo_b"].get("codificador"),
                      "auditoria_mkv": str(audit) if audit else None}
    base["narrativa"] += [f"{len(ch.entrances)} entradas animadas em {ch.duration:.1f} s; o vídeo termina "
                          "idêntico à sua foto (verificado byte a byte em todos os quadros)."]
    return _finish(ctx, base, str(out_path), warning, gate)


def _base_report(ctx: ModeBContext, inst, res, pieces, gate: dict, t_load: float, vram_b) -> dict:
    req, q, hw, prof, room = ctx.req, ctx.q, ctx.hw, ctx.prof, ctx.room
    gpu = hw.best_gpu
    H, W = ctx.photo.rgb8.shape[:2]
    left = [i.label for i in inst.instances if res.kept_builtins and i.kind == "planejado"]
    area = float(res.mask.mean())
    d = ctx.decision
    narr = [f"Foto mobiliada detectada ({d.reason})." if not d.forced else f"Modo B escolhido manualmente.",
            f"{len(pieces)} peças da própria foto retiradas por trás e devolvidas uma a uma: "
            f"{_labels_pt(p.label for p in pieces)}."]
    if left:
        narr.append(f"Planejados mantidos no fundo para preservar a qualidade: {_labels_pt(left)}.")
    narr.append(f"A sala vazia do início foi reconstruída em {100 * area:.1f}% da imagem (só onde havia móveis, "
                "sombras e reflexos); o resto é a sua foto, pixel a pixel.")
    if gate["nota"] is not None:
        narr.append(f"Nota automática da sala vazia: {gate['nota']:.0f}/100 "
                    f"({'aprovada' if gate['aprovada'] else 'reprovada'}).")
    return {
        "versao": __version__,
        "data": ctx.stamp,
        "primeira_execucao": bool(ctx.first),
        "modo": {"modo": "B", "motivo": d.reason, "incerto": d.uncertain, "forcado": d.forced,
                 "medidas": d.features},
        "foto": {"arquivo": str(req.photo_path), "original": [ctx.original.width, ctx.original.height],
                 "placa_video": [W, H], "megapixels": round(ctx.original.width * ctx.original.height / 1e6, 3),
                 "megapixels_placa": round(W * H / 1e6, 3),
                 "reduzida": [ctx.original.width, ctx.original.height] != [W, H]},
        "ambiente": req.room_type,
        "qualidade": q,
        "hardware": json.loads(hw.to_json()),
        "perfil": prof.__dict__,
        "gpu": {"nome": gpu.name if gpu else None, "driver": gpu.driver if gpu else None,
                "vram_total_mb": gpu.vram_total_mb if gpu else None,
                "vram_pico_mb": {"analise": ctx.vram.peak("analise"), "modo_b": ctx.vram.peak("modo_b"),
                                 "analise_torch": ctx.vram_torch, "modo_b_torch": vram_b},
                "nvml": ctx.vram.active},
        "sala": {"camera_altura_m": room.camera_height, "pe_direito_m": room.ceiling_height,
                 "paredes": len(room.walls), "aberturas": [o.kind for o in room.openings], "qualidade": room.quality},
        # regra absoluta (ARQUITETURA §1): onde há pixel gerado e quanto
        "pixels_gerados": {"area_pct": round(100 * area, 2), "area_px": int(res.mask.sum()),
                           "onde": "só nos quadros com a sala vazia ou parcialmente vazia, dentro da máscara de "
                                   "remoção; o último quadro é a foto"},
        "pecas": inst.report(),
        "pecas_no_video": [f"{p.label}#{p.id}" for p in pieces],
        "pecas_no_fundo": left,
        "semente": req.seed,
        **res.report(),
        "portao_qualidade": {**gate, "tentativas_detalhe": list(res.info.get("tentativas", []))},
        "modelos_carga_s": {"lama": round(t_load, 2)},
        "narrativa": narr,
        "titulo_narrativa": NARRATIVE_TITLE,
        "aviso": gate["aviso"],
        "plataforma": platform.platform(),
    }


def _finish(ctx: ModeBContext, report: dict, output: str, warning: str | None, gate: dict) -> JobResult:
    sw, em = ctx.sw, ctx.em
    with sw.measure("relatorio"):
        if ctx.first and ctx.state_path is not None and output:
            try:
                from . import bootstrap

                bootstrap.record_run(ctx.state_path)
            except OSError:
                pass
    now = time.time()
    tempos = sw.rounded()
    tempos["total"] = round(now - ctx.t_ref, 2)
    tempos["total_percebido"] = round(now - ctx.t_click, 2)
    tempos["soma_etapas"] = round(stages_sum(tempos, ETAPAS_B), 2)
    report["tempos_s"] = tempos
    report["video_entregue"] = bool(output)
    rp = ctx.work / "report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    flat = {k: value_total(v) for k, v in tempos.items()}
    em.log("Tempos: " + ", ".join(f"{k} {flat[k]:.1f} s" for k in ETAPAS_B if flat.get(k, 0) >= 0.05)
           + f" | total {tempos['total']:.1f} s")
    return JobResult(output, str(rp), str(ctx.work), flat, mode="B", warning=warning,
                     gate={k: gate[k] for k in ("aprovada", "nota", "estrategia", "aceita_pelo_usuario")})


__all__ = ["ModeBContext", "run_mode_b", "to_video_resolution", "gate_message"]
