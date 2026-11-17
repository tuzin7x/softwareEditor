"""Modo B do pipeline (PLANO_V3 §3, PT-V3-8): foto mobiliada -> sala vazia por trás -> móveis reais entram -> foto.

Chamado por :func:`mobilia.pipeline.run_job` depois da análise (MoGe-2 + OneFormer semântico e panóptico + sala),
quando a detecção de modo (``removal.instances.detect_mode``) diz "B" ou o pedido força o modo B. Etapas:

1. **Peças** (``removal/instances.py``): instâncias de móveis da foto na resolução do vídeo (OneFormer da análise,
   reamostrado, + SAM 2.1 se os pesos estão no disco), classificadas em soltos e planejados.
2. **Sala vazia + portão de qualidade** (``removal/plate.py``, :func:`build_plate_gated`): a sala vazia é
   **refeita pela geometria** (``removal/geofill.py``): piso, paredes e teto da caixa da sala em vistas
   retificadas, só com o material real visível (síntese por exemplos alinhada à grade, luz suave medida, rodapé e
   divisas retas); nada é imaginado. Nota automática (OneFormer rodado de novo na placa para achar móvel que
   "voltou"). Reprovada: tenta mantendo os planejados e com a máscara mais larga. Ainda reprovada: o trabalho
   termina **sem MP4** e com um aviso claro (a interface oferece gerar mesmo assim, usar o modo A ou outra foto), a
   não ser que o pedido traga ``accept_weak_plate``. A LaMa só é carregada se a foto não tiver modelo da sala; o
   refino SDXL fica desligado por padrão (``MOBILIA_SDXL=auto`` ou ``1`` liga).
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
# refino SDXL da sala vazia: desligado por padrão (a sala vazia é refeita pela geometria, sem imaginar nada);
# ``MOBILIA_SDXL=auto`` volta a deixar o perfil de hardware decidir, "1" força
SDXL_DEFAULT = "0"
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
    """big-lama (CUDA quando há, senão CPU; baixa e confere o SHA-256 no 1º uso), carregado só se alguma tentativa
    do portão precisar dele: a sala vazia padrão é refeita pela geometria (``removal/geofill.py``), sem LaMa."""
    from .removal import plate as rp

    return rp.LazyLama(device)


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


def sdxl_choice(hw) -> tuple[bool, bool, str]:
    """(usar, offload, motivo) do refino SDXL da sala vazia (PT-V3-4). ``MOBILIA_SDXL``: "0" desliga; "1" força
    (até na CPU, só para testes; precisa dos pesos); padrão "auto": o perfil da V3 decide (GPU NVIDIA com VRAM
    suficiente, RAM para o offload e pesos instalados). Na GPU de 8 GB o SDXL roda com ``enable_model_cpu_offload``:
    o OneFormer da nota de cada semente divide a VRAM com ele."""
    from .gen.sdxl import ENV_SDXL

    mode = os.environ.get(ENV_SDXL, SDXL_DEFAULT).strip().lower()
    if mode in ("0", "false", "nao", "não", "off", "desligado"):
        return False, False, f"refino SDXL desligado ({ENV_SDXL}=0)"
    from .analysis.weights import sdxl_installed
    from .hardware import choose_profile_v3, sdxl_capacity

    force = mode in ("1", "true", "sim", "forcar", "forçar")
    try:
        cabe, _, why_cap = sdxl_capacity(hw)
        if not cabe and not force:
            return False, False, why_cap
        # sem GPU que caiba não confere os pesos (conferir o SHA-256 de 5 GB custa segundos no 1º uso)
        perfil = choose_profile_v3(hw, "B", sdxl_instalado=None if (cabe or force) else False)
    except Exception as e:  # relatório de hardware incompleto: segue sem SDXL
        return False, False, f"perfil de hardware indisponível ({type(e).__name__})"
    if perfil.sdxl:
        gpu = getattr(hw, "best_gpu", None)
        big = gpu is not None and (gpu.vram_total_mb or 0) >= 12000
        return True, bool(perfil.sdxl_offload or not big), perfil.motivo
    if force and sdxl_installed():
        return True, False, f"refino SDXL forçado ({ENV_SDXL}=1) apesar de: {perfil.motivo}"
    return False, False, perfil.motivo


def cuda_free_mb(device: str) -> int | None:
    """VRAM livre agora (``torch.cuda.mem_get_info``), com o que este processo já carregou. None fora da CUDA."""
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        return int(torch.cuda.mem_get_info()[0] // 2**20)
    except Exception:  # noqa: BLE001
        return None


class VramSwap:
    """Tira da GPU os modelos do portão já carregados (LaMa, OneFormer da nota) enquanto o SDXL gera e os devolve
    depois. Com ``enable_model_cpu_offload`` o UNet inteiro (4,9 GiB) sobe para a GPU de uma vez; na GPU de 8 GB, com
    a LaMa e o OneFormer residentes, não sobra o bastante (revisão de código, rodada 2). O SDXL devolve os pesos dele
    à CPU no fim de cada chamada (ganchos do offload), então o OneFormer da nota volta a caber logo depois.
    Custo: copiar ~1 GB por semente entre RAM e VRAM (estimado em menos de 1 s; não medido)."""

    def __init__(self, device: str, modules):
        self.device = str(device)
        self.mods = [m for m in modules if m is not None and hasattr(m, "to")]
        self.active = self.device.startswith("cuda") and bool(self.mods)

    def reclaim_mb(self) -> int:
        """VRAM (MB) que os modelos ocupam e que a troca libera durante a geração."""
        if not self.active:
            return 0
        tot = 0
        for m in self.mods:
            try:
                for t in list(m.parameters()) + list(m.buffers()):
                    if t.is_cuda:
                        tot += t.numel() * t.element_size()
            except Exception:  # noqa: BLE001 - módulo sem parâmetros enumeráveis: não conta
                pass
        return int(tot // 2**20)

    def _move(self, dev: str) -> None:
        for m in self.mods:
            m.to(dev)
        if dev == "cpu":
            import torch

            torch.cuda.empty_cache()

    def wrap(self, generate):
        """``generate`` com os modelos do portão fora da GPU durante a chamada (e de volta mesmo com erro)."""
        if not self.active:
            return generate

        def gen(*a, **k):
            self._move("cpu")
            try:
                return generate(*a, **k)
            finally:
                self._move(self.device)

        return gen


def sdxl_recheck(device: str, offload: bool, reclaim_mb: int = 0) -> tuple[bool, bool, str, int | None]:
    """Reconfere a VRAM livre **depois** de a LaMa e o OneFormer da nota estarem na GPU (o perfil leu a VRAM livre
    antes deles). Devolve (usar, offload, motivo, livre_mb): com menos que ``SDXL_VRAM_FULL_MB`` livres liga o
    offload; com menos que ``SDXL_VRAM_MIN_MB``, desliga o SDXL (fica a LaMa) em vez de arriscar falta de memória
    ou o *sysmem fallback* do driver no meio do trabalho. ``reclaim_mb``: VRAM que :class:`VramSwap` libera durante
    a geração (somada à livre). Fora da CUDA nada muda."""
    from .hardware import SDXL_VRAM_FULL_MB, SDXL_VRAM_MIN_MB

    free = cuda_free_mb(device)
    if free is None:
        return True, offload, "", None
    free += max(0, int(reclaim_mb))
    if free < SDXL_VRAM_MIN_MB:
        return False, offload, (f"VRAM livre depois da LaMa e do OneFormer: {free} MB (mínimo {SDXL_VRAM_MIN_MB}); "
                                "refino SDXL desligado nesta foto"), free
    if free < SDXL_VRAM_FULL_MB and not offload:
        return True, True, f"VRAM livre depois da LaMa e do OneFormer: {free} MB; SDXL com offload", free
    return True, offload, f"VRAM livre depois da LaMa e do OneFormer: {free} MB", free


def load_refiner(device: str, offload: bool):
    """Refinador SDXL (``generate`` de :func:`mobilia.gen.sdxl.refine_plate`) e a configuração do dispositivo."""
    from .gen import sdxl

    cfg = sdxl.RefineConfig.for_device(device)
    return sdxl.load_sdxl(device, offload=offload, cfg=cfg), cfg


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
    names = {"crua_geometria": "sala crua, refeita pela geometria",
             "crua_por_plano": "sala crua, por plano", "planejados_mantidos": "mantendo os planejados",
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
            # refino SDXL (perfil "B com SDXL"): estágio depois da LaMa em cada tentativa do portão; sem pesos, sem
            # VRAM ou com erro de carga, segue só com a LaMa (o motivo vai para o relatório)
            use_sdxl, offload, why = sdxl_choice(ctx.hw)
            sdxl_rep = {"ligado": use_sdxl, "motivo": why, "offload": offload,
                        "vram_pico_mb_por_etapa": {"lama_e_oneformer_carregados": torch_peak_mb(dev)}}
            refiner = refine = None
            swap = VramSwap(dev, [getattr(inpaint, "model", None), red_model])
            if use_sdxl:
                use_sdxl, offload, why2, free = sdxl_recheck(dev, offload, swap.reclaim_mb())
                sdxl_rep["vram_trocada_mb"] = swap.reclaim_mb()
                sdxl_rep.update(ligado=use_sdxl, offload=offload, vram_livre_antes_sdxl_mb=free)
                if why2:
                    sdxl_rep["reconferencia_vram"] = why2
                    if not use_sdxl:
                        sdxl_rep["motivo"] = why2
                        em.log(why2, "warning")
            if use_sdxl:
                em.stage("sala_vazia", 0.05, "Carregando o refino da sala vazia (SDXL)")
                t1 = time.perf_counter()
                try:
                    refiner, rcfg = load_refiner(dev, offload)
                    sdxl_rep["carga_s"] = round(time.perf_counter() - t1, 2)
                    sdxl_rep["vram_pico_mb_por_etapa"]["sdxl_carregado"] = torch_peak_mb(dev)
                except Exception as e:  # WeightsError, falta de memória, diffusers ausente...
                    sdxl_rep.update(ligado=False, motivo=f"refino SDXL indisponível: {str(e)[:300]}")
                    em.log(f"Refino SDXL indisponível ({type(e).__name__}); a sala vazia fica só com a LaMa.",
                           "warning")
                if refiner is not None:
                    from .gen import sdxl as gsdxl

                    generate = swap.wrap(refiner)

                    def refine(r, name):
                        em.stage("sala_vazia", min(0.9, 0.25 * attempt_no[0] + 0.15),
                                 f"Refinando a sala vazia com SDXL ({names.get(name, name)})")
                        return gsdxl.refine_plate(foto, r, seg_v, ctx.room, ctx.geo.depth, generate, redetect,
                                                  seed=req.seed, cfg=rcfg, room_type=req.room_type)
            try:
                res = rp.build_plate_gated(foto, inst, seg_v, ctx.room, ctx.geo.depth, inpaint, redetect,
                                           seed=req.seed, on_attempt=on_attempt, refine=refine)
            finally:
                if refiner is not None:
                    sdxl_rep["chamadas"] = list(refiner.calls)
                    refiner.release()
                refine = refiner = redetect = red_model = inpaint = None    # libera os modelos (VRAM)
                swap = generate = None
                gc.collect()
                if dev.startswith("cuda"):
                    import torch

                    torch.cuda.empty_cache()
            vram_b = torch_peak_mb(dev)
            sdxl_rep["vram_pico_mb_por_etapa"]["portao_completo"] = vram_b
        Image.fromarray(res.plate).save(work / "sala_vazia.png")
        Image.fromarray(res.plate).save(work / "sala_vazia.jpg", quality=90)
    em.preview(work / "sala_vazia.jpg")
    quality = res.quality
    ok = quality is None or quality.ok
    accepted = (not ok) and bool(req.accept_weak_plate)
    warning = None if ok else gate_message(quality, accepted)
    gate = {"aprovada": ok, "nota": None if quality is None else round(quality.score, 1),
            "estrategia": res.info.get("estrategia_portao", "crua_geometria"),
            "tentativas": len(res.info.get("tentativas", [])),
            "motivos": [] if quality is None else list(quality.reasons),
            "reconferida_oneformer": _flag(ENV_REDETECTAR, "1"),
            "aceita_pelo_usuario": accepted, "aviso": warning}
    for w in res.warnings:
        em.log(w, "warning")
    pieces = inst.removal(res.kept_builtins)
    ri_ = res.info.get("refino_sdxl") or {}
    sdxl_rep.update(usado=bool(ri_.get("usado")), semente=ri_.get("semente"),
                    resultado=ri_.get("motivo"), sementes=ri_.get("sementes"),
                    nota_lama=ri_.get("nota_lama"), prompt=ri_.get("prompt"))
    base = _base_report(ctx, inst, res, pieces, gate, t_load, vram_b)
    base["refino_sdxl"] = sdxl_rep
    base["semente_sdxl"] = sdxl_rep["semente"]
    if "carga_s" in sdxl_rep:
        base["modelos_carga_s"]["sdxl"] = sdxl_rep["carga_s"]
    if sdxl_rep["usado"]:
        base["narrativa"].append(f"A sala vazia foi refinada com SDXL (semente {sdxl_rep['semente']}), só dentro da "
                                 "área reconstruída; janelas, portas e o resto da foto ficaram intactos.")
    if not ok and not accepted:
        em.log(warning, "warning")
        return _finish(ctx, base, "", warning, gate)

    # -------------------------------------------------------------- camadas + coreografia
    em.stage("camadas", 0.1, "Recortando os móveis reais da foto")
    with sw.measure("camadas"):
        st = L.build_layers(foto, res, pieces, ctx.geo.depth)
        for w in st.warnings:             # ex.: pedaço de móvel voltando para a sala vazia (o portão não vê isso)
            em.log(w, "warning")
        em.stage("camadas", 0.6, "Planejando a entrada de cada peça")
        ch = C.plan(st, fps=int(ctx.q.get("fps") or C.FPS))
        cv = C.verify(st, ch)
        for w in ch.warnings:             # ex.: corte à vista pela verificação independente (o vídeo será recusado)
            em.log(w, "warning")
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
        why = (res.info.get("cobertura") or {}).get("motivo") or "a sala crua reprovou na verificação"
        narr.append(f"No modo B os planejados também saem (sala crua), mas aqui ficaram na foto porque {why}: "
                    f"{_labels_pt(left)}.")
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
