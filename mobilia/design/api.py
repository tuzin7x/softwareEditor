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
from . import structure as S
from ..analysis.segmentation import GROUP_IDS as GROUP_IDS_
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
    aviso: str | None = None            # mensagem para o usuário (ex.: ambiente vazio -> mobiliado)
    requested: str | None = None        # a função pedida (``funcao`` é a aplicada)

    def to_event(self) -> dict:
        ev = {"event": "done", "output": self.outputs[0] if self.outputs else None, "outputs": self.outputs,
              "before": self.before, "report": self.report, "funcao": self.funcao, "estilo": self.estilo,
              "ambiente": self.ambiente, "seeds": self.seeds, "size": list(self.size)}
        if self.aviso:
            ev["aviso"] = self.aviso
        if self.requested and self.requested != self.funcao:
            ev["funcao_pedida"] = self.requested
        return ev


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
def depth_control(depth: np.ndarray, size: tuple[int, int], lo_pct: float = 2.0, hi_pct: float = 98.0,
                  ref: np.ndarray | None = None) -> np.ndarray:
    """Profundidade métrica (z, metros; <= 0 ou não finito = inválido) -> mapa de controle (H, W) em [0, 1] no
    formato do ControlNet depth do SDXL: disparidade normalizada, perto = claro. ``ref``: outra profundidade cujos
    percentis definem a escala (a estrutural usa a da foto: a mesma parede tem o mesmo tom nos dois mapas)."""
    z = np.asarray(depth, np.float32)
    valid = np.isfinite(z) & (z > 1e-4)
    if valid.sum() < 16:
        return np.zeros((size[1], size[0]), np.float32)
    disp = np.zeros_like(z)
    disp[valid] = 1.0 / z[valid]
    src = disp[valid]
    if ref is not None:
        zr = np.asarray(ref, np.float32)
        vr = np.isfinite(zr) & (zr > 1e-4)
        if vr.sum() >= 16:
            src = 1.0 / zr[vr]
    lo, hi = np.percentile(src, [lo_pct, hi_pct])
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
    """Segmentação (OneFormer ADE20K) numa versão reduzida da foto: ``SegmentationResult`` com os rótulos (os
    eletrodomésticos e a cozinha integrada são lidos deles) e os grupos de ``analysis/segmentation.GROUPS``."""
    from ..analysis.segmentation import segment
    from ..config import SEGMENTATION_MODEL_ID

    s = min(1.0, long_side / max(rgb.shape[:2]))
    small = _resize(rgb, (int(round(rgb.shape[1] * s)), int(round(rgb.shape[0] * s))))
    gpu = device.startswith("cuda")
    return segment(small, SEGMENTATION_MODEL_ID, device=device, use_fp16=gpu)


def _groups_of(groups):
    """Grupos de um ``SegmentationResult`` (recalculados dos rótulos ADE20K com o mapeamento atual, que pode ter
    mudado desde que a análise foi salva) ou o próprio array de grupos."""
    labels, names = getattr(groups, "labels", None), getattr(groups, "id2label", None)
    if labels is not None and names:
        from ..analysis.segmentation import labels_to_groups

        return labels_to_groups(labels, names)
    return getattr(groups, "groups", groups)


# ====================================================================================== redesign
EMPTY_NOTICE = ("Ambiente vazio detectado: em vez de só trocar o estilo de um cômodo vazio, o 7xAi mobiliou o "
                "ambiente no estilo escolhido.")
KITCHEN_LABELS = ("cabinet", "counter", "countertop", "stove", "oven", "refrigerator", "dishwasher", "microwave",
                  "hood", "kitchen island", "sink")


def furniture_coverage(groups: np.ndarray) -> float:
    """Fração da área que não é janela/porta coberta por móveis e bancadas (grupos ``furniture`` e ``fixture``):
    abaixo de ``presets.EMPTY_FURNITURE_FRAC`` o ambiente é tratado como vazio."""
    from ..analysis.segmentation import GROUP_IDS

    g = np.asarray(groups)
    free = ~np.isin(g, [GROUP_IDS["window"], GROUP_IDS["door"]])
    furn = np.isin(g, [GROUP_IDS["furniture"], GROUP_IDS["fixture"]]) & free
    return float(furn.sum() / max(free.sum(), 1))


def _seg_labels(groups):
    labels, names = getattr(groups, "labels", None), getattr(groups, "id2label", None)
    return (labels, names) if (labels is not None and names) else (None, None)


@dataclass
class _Analysis:
    lock: np.ndarray
    keep: np.ndarray
    openings: np.ndarray          # todas as aberturas da foto (janela, porta, cortina) — referência do reparo
    free: np.ndarray              # janelas/cortinas fora da trava: não podem sumir (conferência pós-geração)
    windows: bool
    open_kitchen: bool
    info: dict


def _analyse_lock(fp: P.FunctionPreset, gr: np.ndarray | None, seg, base: np.ndarray, depth, room_key: str,
                  size: tuple[int, int]) -> _Analysis:
    W, H = size
    lock = np.zeros((H, W), bool)
    keep = np.zeros((H, W), bool)
    openings = np.zeros((H, W), bool)
    info: dict = {}
    if gr is None:
        return _Analysis(lock, keep, openings, openings.copy(), True, False, info)
    lock_names, keep_names = tuple(fp.lock), tuple(fp.keep)
    if lock_names:
        lock = AL.lock_mask(gr, lock_names, (W, H))
    if keep_names:
        keep = AL.lock_mask(gr, keep_names, (W, H))
    openings = AL.lock_mask(gr, ("window", "door", "window_covering"), (W, H))
    win = AL.lock_mask(gr, ("window",), (W, H))
    if "window" in lock_names and "window_covering" not in lock_names:
        bright = AL.bright_coverings(AL.lock_mask(gr, ("window_covering",), (W, H)), base)
        if bright.any():
            lock |= bright
            win |= bright
            info["cortinas_claras"] = round(float(bright.mean()), 4)
    if "door" in lock_names:
        door = AL.lock_mask(gr, ("door",), (W, H))
        if depth is not None:
            z = cv2.resize(np.asarray(depth, np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            views = AL.doorway_views(door, z) & ~lock
            if views.any():
                lock |= views
                openings |= views
                info["vaos_de_porta"] = round(float(views.mean()), 4)
        lock, framed = AL.cap_doors(lock, door, all_frames=(room_key == "banheiro"))
        if framed:
            info["portas_so_moldura"] = framed
    labels, names = _seg_labels(seg)
    if fp.fixed and labels is not None:
        fixed = AL.label_mask(labels, names, tuple(fp.fixed), (W, H)) & ~lock
        if fixed.any():
            lock |= fixed
            info["eletrodomesticos"] = round(float(fixed.mean()), 4)
    open_kitchen = False
    if room_key in ("sala", "sala_jantar") and labels is not None:
        open_kitchen = float(AL.label_mask(labels, names, KITCHEN_LABELS).mean()) > 0.10
    free = AL.lock_mask(gr, ("window", "window_covering"), (W, H)) & ~lock
    return _Analysis(lock, keep, openings, free, bool(win.mean() > 0.003), open_kitchen, info)


GLASS_MIN_FRAC = 0.02     # componentes de porta/box/espelho a partir disto são segmentados de novo, ampliados
GLASS_LABELS = ("screen door", "mirror", "shower", "glass")


def refine_openings(rgb: np.ndarray, groups: np.ndarray, seg_fn, labels=None, id2label=None,
                    max_crops: int = 3) -> tuple[np.ndarray, int]:
    """Janelas escondidas atrás de vidro: a segmentação da foto inteira marca o box do chuveiro (ou a porta de
    vidro, o espelho) como UM componente de porta, e a janelinha dentro dele some — e no banheiro só a moldura do
    box é travada, então o gerador apagava a janela. Cada componente grande desses é recortado (com margem) da foto
    em resolução cheia e segmentado de novo (``seg_fn(rgb) -> grupos``): a janela que aparece dentro dele vira
    ``window`` nos grupos. Devolve os grupos corrigidos e quantos pixels mudaram."""
    g = np.asarray(groups).copy()
    h, w = g.shape
    cand = g == GROUP_IDS_["door"]
    if labels is not None and id2label:
        cand |= AL.label_mask(labels, id2label, GLASS_LABELS, (w, h))
    if not cand.any():
        return g, 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    order = sorted(range(1, n), key=lambda k: -stats[k, cv2.CC_STAT_AREA])[:max_crops]
    H, W = rgb.shape[:2]
    sx, sy = W / w, H / h
    changed = 0
    for k in order:
        if stats[k, cv2.CC_STAT_AREA] < GLASS_MIN_FRAC * h * w:
            break
        x, y, bw, bh = stats[k, :4]
        mx, my = int(0.1 * bw) + 2, int(0.1 * bh) + 2
        x0, y0, x1, y1 = max(0, x - mx), max(0, y - my), min(w, x + bw + mx), min(h, y + bh + my)
        crop = rgb[int(y0 * sy):int(np.ceil(y1 * sy)), int(x0 * sx):int(np.ceil(x1 * sx))]
        if min(crop.shape[:2]) < 32:
            continue
        cg = np.asarray(seg_fn(crop))
        cg = cv2.resize(cg, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        win = (cg == GROUP_IDS_["window"]) & (lab[y0:y1, x0:x1] == k)
        win = AL.clean_mask(win, min_frac=0.01)
        if win.any():
            sub = g[y0:y1, x0:x1]
            sub[win] = GROUP_IDS_["window"]
            changed += int(win.sum())
    return g, changed


def _repair_openings(eng, img: np.ndarray, new: np.ndarray, lock: np.ndarray, control, room_key: str, seed: int,
                     steps: int, has_windows: bool) -> tuple[np.ndarray, dict]:
    """Refaz como parede lisa as janelas/portas/cortinas que o gerador inventou (``new``): pré-preenche a área
    com o entorno (Telea) e gera só ali, com a profundidade da foto inteira guiando o plano da parede."""
    H, W = new.shape
    diag = float(np.hypot(H, W))
    m = cv2.dilate(new.astype(np.uint8), AL._disk(int(0.02 * diag))).astype(bool) & ~lock
    if not m.any():
        return img, {}
    pre = cv2.inpaint(img, m.astype(np.uint8) * 255, max(3, int(0.01 * diag)), cv2.INPAINT_TELEA)
    r = P.ROOMS[room_key]
    tail = P.PHOTO_TAIL.format(light=P.LIGHT_WINDOWS if has_windows else P.LIGHT_NONE)
    pr = P.DesignPrompt(f"plain painted wall, same color and texture as the surrounding wall, seamless, inside a "
                        f"{r['en']}, {tail}",
                        "window, door, doorway, opening, glass, curtains, blinds, view, sky, trees, frame, "
                        + P.NEGATIVE_BASE)
    g = _resize(eng.generate(pre, m, control, pr, int(seed) + 101, 0.8, steps, 1.8, 0.8, 1.0, None), (W, H))
    alpha = AL.feather_alpha(m, max(2.0, 0.006 * diag), inside=1.0)
    n_comp = cv2.connectedComponents(new.astype(np.uint8))[0] - 1
    return AL.composite(img, g, alpha), {"fracao": round(float(m.mean()), 4), "componentes": int(n_comp)}


def _refine(eng, img: np.ndarray, size2: tuple[int, int], depth, prompt, seed: int,
            guidance: float) -> np.ndarray:
    """Passada de detalhe: amplia 1,5× (Lanczos) e refaz com força baixa (0,3) e a profundidade a meio peso — o
    SDXL inventa a textura fina que a ampliação não tem (tecido, madeira, folhas)."""
    up = _resize(img, size2)
    ctrl = depth_control(depth, size2) if depth is not None else None
    full = np.ones((size2[1], size2[0]), bool)
    return _resize(eng.generate(up, full, ctrl, prompt, int(seed), 0.3, 8, guidance, 0.5, 1.0, None), size2)


def redesign(photo, funcao: str = "mudar_estilo", estilo: str | None = "moderno", ambiente: str = "sala",
             n: int = 4, seed: int | None = None, prompt: str = "", out_dir=None, quality: str = "final",
             evento: str | None = None, engine: Generator | None = None, depth: np.ndarray | None = None,
             groups=None, progress: Progress | None = None, device: str = "auto",
             overrides: dict | None = None, long_side: int | None = None, candidates: int | None = None,
             rank_fn=None, auto_empty: bool = True, repair: bool | None = None, refine: bool | None = None,
             tone: bool = True, refine_glass: bool | None = None) -> DesignResult:
    """Gera ``n`` versões redesenhadas do ambiente da foto.

    ``funcao``: mudar_estilo | reforma | redesign_simples | mobiliar | fachada | evento (para "Editar área" use
    :func:`edit_area`; para "Texto para ambiente", :func:`text_to_design`). ``estilo``: chave de
    ``presets.STYLES`` (``None`` só no evento). ``evento``: chave de ``presets.EVENTS`` (função "evento").
    ``depth`` (z em metros, qualquer resolução) e ``groups`` (grupos semânticos ou ``SegmentationResult``) evitam
    rodar o MoGe/OneFormer. ``overrides``: campos de ``FunctionPreset`` (strength, control_scale, control_end,
    guidance, steps) para ajuste fino. ``candidates``: quantas versões gerar antes de escolher as ``n`` melhores
    (``None`` = automático: na GPU n + 2 no "Mobiliar" e n + 1 em "Mudar estilo"/"Reforma"; n na CPU);
    ``rank_fn(rgb) -> grupos`` troca o OneFormer da escolha e do reparo (testes).

    ``auto_empty``: numa foto de ambiente vazio, "Mudar estilo", "Redesign simples" e "Reforma" viram "Mobiliar"
    (com aviso no ``done``). ``repair`` (``None`` = o do preset): janela/porta/cortina inventada numa parede sem
    abertura é refeita como parede. ``refine`` (``None`` = só na GPU, qualidade final): passada de detalhe a 1,5×.
    ``tone``: ancora exposição e balanço de branco na foto (quanto, por função: ``FunctionPreset.tone_ab``).
    ``refine_glass`` (``None`` = só quando a segmentação é feita aqui): janelas atrás de vidro (box, porta de vidro)
    são procuradas de novo num recorte ampliado (``refine_openings``)."""
    t_all = time.perf_counter()
    fp = P.with_overrides(P.function(funcao), **(overrides or {}))
    if fp.key in ("editar_area", "texto"):
        raise ValueError(f"Use {'edit_area' if fp.key == 'editar_area' else 'text_to_design'} para '{fp.label}'.")
    requested = fp.key
    room_key = "fachada" if fp.exterior else P.norm_room(ambiente)
    st_key = P.norm_style(estilo) if (estilo and not (fp.event and estilo in P.EVENTS)) else None
    if fp.event:
        evento = P.norm_event(evento or (estilo if estilo in P.EVENTS else "aniversario"))
    elif st_key is None:
        raise ValueError("Escolha um estilo.")
    prog = _Prog(progress)
    out = _out_dir(out_dir)
    timings: dict = {}
    aviso = None

    prog("preparo")
    rgb = _load_rgb(photo)
    eng = engine or get_engine(device)
    dev = getattr(eng, "device", pick_device(device))
    gpu = dev.startswith("cuda")
    qp = P.quality_profile(quality, dev)
    W, H = P.work_size(rgb.shape[1], rgb.shape[0], long_side or qp.long_side)
    base = _resize(rgb, (W, H))

    # ---- análise
    t0 = time.perf_counter()
    prog("analise", 0.0, "Medindo a profundidade do ambiente")
    if depth is None and fp.control_scale > 0:
        depth = estimate_depth(rgb, dev)
    control = depth_control(depth, (W, H)) if depth is not None else None
    gr = None
    coverage = None
    seg_here = False
    if fp.lock or fp.keep or fp.fixed or (auto_empty and fp.key in P.EMPTY_TO_STAGING):
        prog("analise", 0.5, "Encontrando janelas, portas e móveis")
        if groups is None:
            groups = estimate_groups(rgb, dev)
            seg_here = True
        gr = _groups_of(groups)
        coverage = round(furniture_coverage(gr), 4)
        if auto_empty and fp.key in P.EMPTY_TO_STAGING and coverage < P.EMPTY_FURNITURE_FRAC:
            fp = P.with_overrides(P.function("mobiliar"), **(overrides or {}))
            aviso = EMPTY_NOTICE
            prog("analise", 0.6, "Ambiente vazio detectado: mobiliando")
            prog.emit({"event": "log", "level": "info", "message": aviso})
    seg = None
    seg_fn = rank_fn

    def get_seg():
        nonlocal seg, seg_fn
        if seg_fn is None:
            from .quality import Segmenter

            seg = Segmenter(dev)
            seg_fn = seg.groups
        return seg_fn

    glass = 0
    if gr is not None and fp.lock and (refine_glass if refine_glass is not None else seg_here):
        prog("analise", 0.8, "Procurando janelas atrás de vidros")
        labels, names = _seg_labels(groups)
        gr, glass = refine_openings(rgb, gr, get_seg(), labels, names)
    an = _analyse_lock(fp, gr, groups, base, depth, room_key, (W, H))
    if glass:
        an.info["janelas_atras_de_vidro_px"] = glass
    lock, keep = an.lock, an.keep
    # Reforma: profundidade da sala SEM os móveis (a disposição muda; paredes, janelas e câmera ficam)
    struct_depth = None
    gen_base = base
    if fp.structural and gr is not None and depth is not None:
        hole = S.movable_mask(gr, (W, H), fixtures=not an.open_kitchen) & ~lock
        if hole.mean() > 0.01:
            dz = cv2.resize(np.asarray(depth, np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            struct_depth = S.structural_depth(dz, hole, gr)
            control = depth_control(struct_depth, (W, H), ref=dz)
            dw = float(np.hypot(W, H))
            pre = cv2.inpaint(base, hole.astype(np.uint8) * 255, max(3, int(0.01 * dw)), cv2.INPAINT_TELEA)
            gen_base = AL.composite(base, cv2.GaussianBlur(pre, (0, 0), 0.01 * dw), hole.astype(np.float32))
            an.info["estrutural_moveis"] = round(float(hole.mean()), 4)
    timings["analise_s"] = round(time.perf_counter() - t0, 2)
    gen_mask = AL.generation_mask(lock | keep)
    steps = int((overrides or {}).get("steps") or (fp.steps if gpu else min(fp.steps, qp.steps)))
    if qp.key == "rapido":
        steps = min(steps, qp.steps)
    do_repair = bool(fp.repair if repair is None else repair) and gr is not None and fp.needs_photo
    do_refine = bool((gpu and qp.key == "final") if refine is None else refine)
    if do_refine:
        Wf, Hf = P.work_size(rgb.shape[1], rgb.shape[0], min(1536, int(round(max(W, H) * 1.5))))
    else:
        Wf, Hf = W, H
    base_f = _resize(rgb, (Wf, Hf)) if do_refine else base
    rs = (lambda m: m) if not do_refine else (lambda m: cv2.resize(
        m.astype(np.uint8), (Wf, Hf), interpolation=cv2.INTER_NEAREST).astype(bool))
    lock_f, keep_f, open_f, free_f = rs(lock), rs(keep), rs(an.openings), rs(an.free)
    depth_g = struct_depth if struct_depth is not None else depth
    control_f = (depth_control(depth_g, (Wf, Hf), ref=depth) if (do_refine and depth_g is not None)
                 else control)

    # ---- geração (com ``candidates`` > n: gera mais e fica com as n de melhor nota, ver ``quality.py``)
    n = max(1, int(n))
    if candidates is None:
        candidates = (n + 2 if fp.key == "mobiliar" else n + 1 if fp.key in ("mudar_estilo", "reforma") else n) \
            if gpu else n
    m = max(n, int(candidates))
    seeds = _seeds(seed, m)
    prompts = [P.build_prompt(fp.key, st_key, room_key, i, prompt, evento, windows=an.windows,
                              open_kitchen=an.open_kitchen) for i in range(m)]
    if hasattr(eng, "embeddings"):
        prog("geracao", 0.0, "Preparando o texto do estilo")
        eng.embeddings(prompts)
    tag = f"{fp.key}_{st_key or evento}"
    do_ceiling = fp.keep_ceiling and gr is not None
    checks = m > n or do_repair or do_ceiling
    span = 0.85 if checks else 1.0
    cand_dir = out / "candidatos" if m > n else out
    paths, images = [], []
    lock_info: dict = {}
    t_gen = time.perf_counter()
    for i, (sd, pr) in enumerate(zip(seeds, prompts)):
        def cb(k, tot, i=i):
            prog("geracao", (i + k / max(tot, 1)) / m * span, f"Gerando a versão {i + 1} de {m}")

        cb(0, 1)
        img = eng.generate(gen_base, gen_mask, control, pr, sd, fp.strength, steps, fp.guidance,
                           fp.control_scale, fp.control_end, cb)
        img = _resize(img, (W, H))
        if do_refine:
            prog("geracao", (i + 1) / m * span, f"Refinando os detalhes da versão {i + 1}")
            # na Reforma o refino não usa profundidade: a da foto traria os móveis antigos de volta e a da sala vazia
            # achataria os novos
            img = _refine(eng, img, (Wf, Hf), depth if struct_depth is None else None, pr, sd, fp.guidance)
        if tone:
            img = AL.anchor_tone(img, base_f, amount_ab=fp.tone_ab,
                                 chroma_cap=1.15 if fp.tone_ab >= 0.3 else 1.5)
        if lock_f.any():
            img, lock_info = AL.lock_openings(img, base_f, lock_f)
        path = _save(img, cand_dir / f"{tag}_{i + 1}.jpg")
        paths.append(path)
        images.append(img)
        prog.emit({"event": "preview", "path": path, "index": i})
    timings["geracao_s"] = round(time.perf_counter() - t_gen, 2)
    timings["por_imagem_s"] = round(timings["geracao_s"] / m, 2)

    # ---- conferência: nota dos candidatos e reparo de aberturas inventadas (um OneFormer para tudo)
    notas = None
    order = list(range(m))
    repairs: dict = {}
    gen_groups: list = [None] * m

    def groups_of(k):
        if gen_groups[k] is None:
            gen_groups[k] = get_seg()(images[k])
        return gen_groups[k]

    try:
        if m > n:
            prog("geracao", 0.87, "Escolhendo as melhores versões")
            t0 = time.perf_counter()
            from .quality import score

            notas = [score(groups_of(k), lock_f | keep_f, furnish=fp.furnish, source_openings=free_f)
                     for k in range(m)]
            order = sorted(range(m), key=lambda k: -notas[k]["nota"])[:n]
            timings["escolha_s"] = round(time.perf_counter() - t0, 2)
        if do_repair:
            t0 = time.perf_counter()
            for j, k in enumerate(order):
                prog("geracao", 0.88 + 0.06 * j / len(order), "Conferindo janelas e portas")
                new = AL.new_openings(groups_of(k), open_f | lock_f, (Wf, Hf), source_groups=gr)
                if new.any():
                    prog("geracao", 0.88 + 0.06 * (j + 0.5) / len(order), "Corrigindo uma janela inventada")
                    fixed, info = _repair_openings(eng, images[k], new, lock_f, control_f, room_key, seeds[k],
                                                   steps, an.windows)
                    if info:
                        # a versão antes do reparo fica em candidatos/ (conferência), com a área refeita
                        info["antes_do_reparo"] = _save(images[k], out / "candidatos" / f"{tag}_{k + 1}_sem_reparo.jpg")
                        _save((new.astype(np.uint8) * 255), out / "candidatos" / f"{tag}_{k + 1}_aberturas_novas.png")
                        if lock_f.any():
                            fixed, _ = AL.lock_openings(fixed, base_f, lock_f)
                        images[k] = fixed
                        repairs[str(k)] = info
                # janela/cortina da foto que sumiu (virou parede, tijolo, armário): a original volta
                lost = AL.missing_openings(groups_of(k), free_f, (Wf, Hf))
                if lost.any():
                    images[k], _ = AL.lock_openings(images[k], base_f, lost)
                    repairs.setdefault(str(k), {})["aberturas_perdidas"] = round(float(lost.mean()), 4)
            timings["reparo_s"] = round(time.perf_counter() - t0, 2)
        if do_ceiling:
            t0 = time.perf_counter()
            for j, k in enumerate(order):
                prog("geracao", 0.94 + 0.05 * j / len(order), "Conferindo o teto")
                cm = AL.ceiling_keep_mask(gr, groups_of(k), (Wf, Hf), lock=lock_f)
                if cm.any():
                    images[k], cinfo = AL.paste_ceiling(images[k], base_f, cm)
                    repairs.setdefault(str(k), {})["teto"] = cinfo["fracao"]
            timings["teto_s"] = round(time.perf_counter() - t0, 2)
    finally:
        if seg is not None:
            seg.close()
    outputs = []
    for j, k in enumerate(order):
        if m > n or str(k) in repairs:
            outputs.append(_save(images[k], out / f"{tag}_{j + 1}.jpg"))
        else:
            outputs.append(paths[k])
    sel_seeds = [seeds[k] for k in order]
    images = [images[k] for k in order]

    prog("final")
    before = _save(base_f, out / "antes.jpg")
    timings["total_s"] = round(time.perf_counter() - t_all, 2)
    rep = {"funcao": fp.key, "funcao_pedida": requested, "aviso": aviso, "mobiliado_na_foto": coverage,
           "estilo": st_key, "evento": evento, "ambiente": room_key, "tamanho": [Wf, Hf],
           "tamanho_geracao": [W, H], "qualidade": qp.key, "dispositivo": dev, "sementes": sel_seeds,
           "passos": steps, "refino": do_refine, "tom": tone, "preset": asdict(fp),
           "prompts": [asdict(prompts[k]) for k in order],
           "trava": {"grupos": list(fp.lock), "mantidos": list(fp.keep), "fixos": list(fp.fixed),
                     "cozinha_integrada": an.open_kitchen, "janelas": an.windows, "teto_original": do_ceiling,
                     "profundidade_estrutural": struct_depth is not None, **an.info, **lock_info},
           "reparos": repairs, "saidas": outputs, "antes": before, "tempos": timings,
           "candidatos": ({"sementes": seeds, "arquivos": paths, "notas": notas, "escolhidos": order}
                          if notas is not None else None),
           "motor": {"carga_s": round(getattr(eng, "load_s", 0.0), 2), "chamadas": getattr(eng, "calls", [])[-m:]}}
    report = out / "design.json"
    report.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    return DesignResult(outputs, before, str(report), fp.key, st_key or evento, room_key, sel_seeds, (Wf, Hf),
                        timings, images, aviso=aviso, requested=requested)


# ====================================================================================== editar área
def _match_edge(g: np.ndarray, ref: np.ndarray, mask: np.ndarray, ring: np.ndarray, max_shift: float = 5.0
                ) -> np.ndarray:
    """Desloca a luz/cor (Lab, limitado) da faixa interna da área gerada para a da faixa da foto logo fora dela:
    sem isso a borda da área nova às vezes fica com um véu cinza (ou mais escura) que a rampa não esconde."""
    m = np.asarray(mask, bool)
    r = max(2, int(0.02 * np.hypot(*m.shape)))
    inner = m & ~cv2.erode(m.astype(np.uint8), AL._disk(r)).astype(bool)
    if inner.sum() < 32 or ring.sum() < 32:
        return g
    lg = cv2.cvtColor(g.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    lr = cv2.cvtColor(ref.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    shift = np.clip((lr[ring].mean(0) - lg[inner].mean(0)) * 0.5, -max_shift, max_shift)
    # só perto da borda: o miolo (o objeto novo) mantém a cor pedida
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    w = np.clip(1.0 - d / (3.0 * r), 0, 1)[..., None]
    out = cv2.cvtColor((lg + shift[None, None] * w).astype(np.float32), cv2.COLOR_Lab2RGB)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


EDIT_CONTEXT = 0.6        # margem de contexto do recorte (× o lado da área): o gerador precisa "ver" a sala
EDIT_MIN_CROP = 0.5       # o recorte cobre pelo menos esta fração do lado maior da foto


def edit_area(photo, mask, prompt: str = "", estilo: str | None = None, ambiente: str = "sala", n: int = 4,
              seed: int | None = None, out_dir=None, quality: str = "final", engine: Generator | None = None,
              depth: np.ndarray | None = None, progress: Progress | None = None, device: str = "auto",
              overrides: dict | None = None, feather_px: float | None = None, grow_frac: float = 0.01
              ) -> DesignResult:
    """Refaz só a área pintada pelo usuário (``mask``: PNG ou array, branco = editar), guiada por ``prompt`` (em
    português ou inglês: ``glossary.to_english``) e/ou ``estilo``.

    A região é recortada com bastante contexto (``EDIT_CONTEXT``: o gerador vê o piso, as paredes e os móveis em
    volta e põe o objeto novo na mesma perspectiva e luz), gerada na resolução do SDXL e colada de volta na foto
    ORIGINAL (resolução cheia). A geração usa uma máscara SUAVE nos latentes (peso 1 no miolo, rampa na borda):
    a cada passo a borda mistura a foto com o que está sendo gerado, então o tapete/piso continua pela área nova
    em vez de terminar num "remendo oval". A área é pré-preenchida com o entorno (o objeto antigo não semeia) e a
    força fica um pouco abaixo de 1 (a luz e a cor do entorno entram no ponto de partida). A colagem final usa uma
    rampa por dentro da máscara crescida ``grow_frac`` da diagonal: todo pixel fora dela fica idêntico à foto."""
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

    # a pincelada do usuário raramente cobre a sombra e a borda do objeto antigo: a área cresce um pouco (1% da
    # diagonal) para o "fantasma" dele não sobrar em volta; fora da área crescida tudo fica idêntico à foto
    grow = int(round(grow_frac * np.hypot(Wf, Hf)))
    ue = cv2.dilate(um.astype(np.uint8), AL._disk(grow)).astype(bool) if grow > 0 else um
    x0, y0, x1, y1 = AL.mask_bbox(ue, EDIT_CONTEXT, min_side=int(EDIT_MIN_CROP * max(Wf, Hf)))
    crop, cm = rgb[y0:y1, x0:x1], ue[y0:y1, x0:x1]
    W, H = P.work_size(x1 - x0, y1 - y0, qp.long_side)
    crop_ref = _resize(crop, (W, H))
    mw = cv2.resize(cm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    dw = float(np.hypot(W, H))
    # peso da geração nos latentes: 1 dentro da área, rampa de ~3% da diagonal metade por dentro, metade por fora
    weight = AL.feather_alpha(mw, max(4.0, 0.03 * dw), inside=0.5)
    weight[mw & (weight < 1.0) & (cv2.erode(mw.astype(np.uint8), AL._disk(2)) > 0)] = 1.0
    # o objeto antigo não semeia a geração: a área é pré-preenchida com o entorno (só a luz e a cor ficam)
    hole = weight > 0.5
    base = cv2.inpaint(crop_ref, hole.astype(np.uint8) * 255, max(3, int(0.01 * dw)), cv2.INPAINT_TELEA)
    base = AL.composite(crop_ref, cv2.GaussianBlur(base, (0, 0), 0.01 * dw), hole.astype(np.float32))
    ring = cv2.dilate(mw.astype(np.uint8), AL._disk(max(2, int(0.025 * dw)))).astype(bool) & ~mw

    t0 = time.perf_counter()
    prog("analise", 0.0, "Medindo a profundidade do ambiente")
    control = None
    if fp.control_scale > 0:
        if depth is None:
            depth = estimate_depth(rgb, dev)
        dfull = cv2.resize(np.asarray(depth, np.float32), (Wf, Hf), interpolation=cv2.INTER_NEAREST)
        dcrop = cv2.resize(dfull[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_NEAREST)
        # a profundidade do objeto antigo redesenharia a silhueta dele: dentro da área vale a do plano em volta
        dcrop = S.structural_depth(dcrop, hole)
        control = depth_control(dcrop, (W, H))
    timings["analise_s"] = round(time.perf_counter() - t0, 2)

    seeds = _seeds(seed, n)
    prompts = [P.build_prompt("editar_area", st_key, room_key, i, prompt) for i in range(len(seeds))]
    if hasattr(eng, "embeddings"):
        eng.embeddings(prompts)
    fr = feather_px if feather_px is not None else max(3.0, 0.006 * np.hypot(Wf, Hf))
    alpha = AL.feather_alpha(ue, fr, inside=1.0)
    outputs, images = [], []
    t_gen = time.perf_counter()
    for i, (sd, pr) in enumerate(zip(seeds, prompts)):
        def cb(k, tot, i=i):
            prog("geracao", (i + k / max(tot, 1)) / len(seeds), f"Gerando a versão {i + 1} de {len(seeds)}")

        cb(0, 1)
        g = eng.generate(base, weight, control, pr, sd, fp.strength, steps, fp.guidance, fp.control_scale,
                         fp.control_end, cb)
        g = _resize(g, (W, H))
        g = _match_edge(g, crop_ref, mw, ring)        # sem halo: a luz da borda nova casa com a do entorno
        g = _resize(g, (x1 - x0, y1 - y0))
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
           "margem_px": grow,
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
           "estimate_groups", "furniture_coverage", "STAGES", "ASPECTS", "EMPTY_NOTICE"]
