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
from . import memplan as MP
from . import presets as P
from . import structure as S
from ..analysis.segmentation import GROUP_IDS as GROUP_IDS_
from .engine import Generator, free_cuda_cache, get_engine, pick_device

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


def _announce_downloads(prog: "_Prog", dev: str) -> list[str]:
    """Avisa (evento ``log``) quais modelos do redesenho faltam no disco e serão baixados agora — o instalador
    deveria ter baixado todos (``weights.prefetch_models``); a primeira "Mudar estilo" levou 942 s baixando o
    OneFormer sem dizer nada."""
    try:
        from ..analysis.weights import ESRGAN_WEIGHT, missing_design_models
        from ..config import GEOMETRY_MODEL_ID, GEOMETRY_MODEL_ID_SMALL

        skip = {ESRGAN_WEIGHT.label, GEOMETRY_MODEL_ID_SMALL if dev.startswith("cuda") else GEOMETRY_MODEL_ID}
        miss = [m for m in missing_design_models() if m not in skip]
    except Exception:  # noqa: BLE001 - só um aviso
        return []
    if miss:
        prog.emit({"event": "log", "level": "warning",
                   "message": "Baixando modelos de IA que faltam (só desta vez; o instalador deveria tê-los "
                              "baixado): " + ", ".join(miss)})
    return miss


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


REPAIR_STRENGTH = 0.9      # 0,8 deixava o pré-preenchimento (liso, acinzentado) aparecer: um retângulo chapado


def _repair_openings(eng, img: np.ndarray, new: np.ndarray, lock: np.ndarray, control, room_key: str, seed: int,
                     steps: int, has_windows: bool, keep_out: np.ndarray | None = None,
                     gen_size: tuple[int, int] | None = None) -> tuple[np.ndarray, dict]:
    """Refaz como parede lisa as janelas/portas/cortinas que o gerador inventou (``new``): pré-preenche a área
    com o entorno (Telea) e gera só ali, com a profundidade da foto inteira guiando o plano da parede.
    ``keep_out`` (o teto da foto): fica fora do reparo — "parede lisa" pintada no teto deixava uma faixa escura.

    Sem retângulo chapado de borda dura (visto num canto de parede): a máscara vai SUAVE para os latentes, a força é
    0,9 (o gerador refaz a textura e a luz da parede), e a área refeita entra por mistura de Poisson
    (``archlock.poisson_blend``: a borda dela é exatamente a da foto, cor e luz casadas por construção) com uma
    rampa larga por dentro. ``gen_size``: tamanho da geração (o refino pode ter deixado ``img`` maior; o reparo
    gera no tamanho da geração e é ampliado — numa GPU de 8 GB, gerar a 1280 com CFG transbordaria a VRAM)."""
    H, W = new.shape
    diag = float(np.hypot(H, W))
    m = cv2.dilate(new.astype(np.uint8), AL._disk(int(0.02 * diag))).astype(bool) & ~lock
    if keep_out is not None:
        m &= ~np.asarray(keep_out, bool)
    if not m.any():
        return img, {}
    gw, gh = gen_size or (W, H)
    if (gw, gh) != (W, H):
        img_s = _resize(img, (gw, gh))
        m_s = cv2.resize(m.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST).astype(bool)
        ctl_s = None if control is None else cv2.resize(np.asarray(control, np.float32), (gw, gh),
                                                        interpolation=cv2.INTER_LINEAR)
    else:
        img_s, m_s, ctl_s = img, m, control
    dg = float(np.hypot(gw, gh))
    pre = cv2.inpaint(img_s, m_s.astype(np.uint8) * 255, max(3, int(0.01 * dg)), cv2.INPAINT_TELEA)
    weight = AL.feather_alpha(m_s, max(4.0, 0.03 * dg), inside=0.5)
    weight[m_s & (cv2.erode(m_s.astype(np.uint8), AL._disk(2)) > 0)] = 1.0
    r = P.ROOMS[room_key]
    tail = P.PHOTO_TAIL.format(light=P.LIGHT_WINDOWS if has_windows else P.LIGHT_NONE)
    pr = P.DesignPrompt(f"plain painted wall, same color and texture as the surrounding wall, seamless, soft natural "
                        f"shading, inside a {r['en']}, {tail}",
                        "window, door, doorway, opening, glass, curtains, blinds, view, sky, trees, frame, flat grey "
                        "patch, rectangle, panel, seam, visible border, " + P.NEGATIVE_BASE)
    g = eng.generate(pre, weight, ctl_s, pr, int(seed) + 101, REPAIR_STRENGTH, steps, 1.8, 0.8, 1.0, None)
    g = _resize(_resize(g, (gw, gh)), (W, H))
    g = AL.poisson_blend(img, g, m)
    alpha = AL.feather_alpha(m, max(3.0, 0.015 * diag), inside=1.0)
    n_comp = cv2.connectedComponents(new.astype(np.uint8))[0] - 1
    out = AL.composite(img, g, alpha)
    return out, {"fracao": round(float(m.mean()), 4), "componentes": int(n_comp),
                 "degrau_borda": round(AL.seam_step(out, m), 2)}


def _refine(eng, img: np.ndarray, size2: tuple[int, int], depth, prompt, seed: int,
            guidance: float) -> np.ndarray:
    """Passada de detalhe: amplia (Lanczos) para o tamanho do plano de memória (``memplan``: 1280 px numa GPU de
    8 GB, 1536 nas folgadas) e refaz com força baixa (0,3) e a profundidade a meio peso — o SDXL inventa a textura
    fina que a ampliação não tem (tecido, madeira, folhas). ``guidance`` 1 = sem CFG (lote 1, metade da memória)."""
    up = _resize(img, size2)
    ctrl = depth_control(depth, size2) if depth is not None else None
    full = np.ones((size2[1], size2[0]), bool)
    return _resize(eng.generate(up, full, ctrl, prompt, int(seed), 0.3, 8, guidance, 0.5, 1.0, None), size2)


def _green_mask(rgb: np.ndarray) -> np.ndarray:
    """Verde de vegetação e aglomerados de brilho quente com cara de árvore acesa (a fachada ensolarada da foto
    de Stafford) já presentes na foto, crescidos um pouco: ficam fora da busca pela árvore de Natal."""
    from .quality import lit_tree

    hsv = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2HSV)
    m = (hsv[..., 0] >= 33) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 50) & (hsv[..., 2] >= 20)
    m |= lit_tree(rgb, return_mask=True)[1]          # só os aglomerados "de árvore acesa" que já estavam lá
    r = max(1, int(0.01 * np.hypot(*m.shape)))
    return cv2.dilate(m.astype(np.uint8), AL._disk(r)).astype(bool)


# abaixo disto a versão de Natal "não tem árvore" e ganha uma, colocada à mão. Baixo de propósito: com uma
# arvorezinha já lá (nota ~0,4 no teste em CPU) uma segunda árvore grande seria pior que nenhuma
TREE_MIN_SCORE = 0.3
TREE_FREE_MIN = 0.75       # fração do lugar da árvore que precisa ser parede/piso livre


def tree_spot(groups: np.ndarray, size: tuple[int, int], lock: np.ndarray | None = None) -> np.ndarray | None:
    """Lugar para a árvore de Natal: um canto de parede livre (parede/piso, sem janela, porta, móvel ou área
    travada), apoiado no piso perto da parede do fundo, com ~45% da altura da foto. Devolve a silhueta (cone +
    base com os presentes) ou ``None`` se não há canto livre. Os cantos (laterais) têm preferência: árvore no meio
    da sala tampa a sala."""
    W, H = size
    g = np.asarray(groups)
    if g.shape != (H, W):
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)
    floor = g == GROUP_IDS_["floor"]
    free = np.isin(g, [GROUP_IDS_["floor"], GROUP_IDS_["wall"]])
    if lock is not None:
        lk = np.asarray(lock, bool)
        if lk.shape != (H, W):
            lk = cv2.resize(lk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        free &= ~lk
    if floor.mean() < 0.03:
        return None
    ceil = g == GROUP_IDS_["ceiling"]
    th = int(0.45 * H)
    tw = int(0.42 * th)
    best = None
    for cx in sorted(range(int(0.12 * W), int(0.88 * W), max(4, W // 40)),
                     key=lambda x: min(x, W - x)):                      # dos cantos para o meio
        x0, x1 = max(0, cx - tw // 2), min(W, cx + tw // 2)
        cols = floor[:, x0:x1]
        rows = np.nonzero(cols.any(1))[0]
        if not len(rows):
            continue
        # base: um pouco abaixo de onde o piso começa nesta faixa (a árvore fica junto da parede do fundo)
        ytop_floor = int(np.percentile(np.argmax(cols, axis=0)[cols.any(0)], 50))
        base = min(H - 1, ytop_floor + int(0.06 * H))
        top = base - th
        if top < 0:
            continue
        if ceil[top:base, x0:x1].mean() > 0.05:                          # bateria no teto: baixo demais
            continue
        f = float(free[top:base, x0:x1].mean())
        side = 1.0 - min(cx, W - cx) / (0.5 * W)
        sc = f + 0.15 * side
        if f >= TREE_FREE_MIN and (best is None or sc > best[0]):
            best = (sc, cx, base, top)
    if best is None:
        return None
    _, cx, base, top = best
    m = np.zeros((H, W), np.uint8)
    cone = np.array([[cx, top], [cx - tw // 2, base - int(0.1 * th)], [cx + tw // 2, base - int(0.1 * th)]],
                    np.int32)
    cv2.fillPoly(m, [cone], 1)
    cv2.rectangle(m, (cx - int(0.6 * tw), base - int(0.16 * th)), (cx + int(0.6 * tw), base), 1, -1)
    return m.astype(bool)


TREE_PROMPT = ("a tall green christmas tree standing on the floor, decorated with glowing warm fairy lights and red "
               "and gold ornaments, a star on top, wrapped presents under the tree, realistic scale, photorealistic, "
               "sharp focus, high detail, natural indoor lighting")
TREE_NEGATIVE = ("empty corner, bare wall, houseplant, potted plant, window, door, furniture, sofa, blur, smudge, "
                 "seam, visible border, floating tree, tiny tree, " + P.NEGATIVE_BASE)


def _insert_tree(eng, img: np.ndarray, groups: np.ndarray, lock: np.ndarray | None, seed: int, steps: int,
                 gen_size: tuple[int, int] | None = None) -> tuple[np.ndarray, dict]:
    """Põe a árvore de Natal num canto livre (``tree_spot``) gerando só ali, com máscara suave — como a "Editar
    área". Na GPU (0.9.0) nenhuma versão de Natal tinha árvore e uma tinha uma arvorezinha em cima do sofá."""
    H, W = img.shape[:2]
    spot = tree_spot(groups, (W, H), lock)
    if spot is None:
        return img, {"inserida": False, "motivo": "sem canto livre"}
    gw, gh = gen_size or (W, H)
    img_s = _resize(img, (gw, gh))
    sp_s = cv2.resize(spot.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST).astype(bool)
    dg = float(np.hypot(gw, gh))
    weight = AL.feather_alpha(sp_s, max(4.0, 0.02 * dg), inside=0.5)
    weight[sp_s & (cv2.erode(sp_s.astype(np.uint8), AL._disk(2)) > 0)] = 1.0
    g = eng.generate(img_s, weight, None, P.DesignPrompt(TREE_PROMPT, TREE_NEGATIVE), int(seed) + 303, 1.0, steps,
                     2.4, 0.0, 0.0, None)
    g = _resize(_resize(g, (gw, gh)), (W, H))
    dw = float(np.hypot(W, H))
    region = cv2.dilate(spot.astype(np.uint8), AL._disk(max(2, int(0.012 * dw)))).astype(bool)
    if lock is not None:
        lk = np.asarray(lock, bool)
        if lk.shape == region.shape:
            region &= ~lk
    out = AL.composite(img, g, AL.feather_alpha(region, max(3.0, 0.01 * dw), inside=1.0))
    return out, {"inserida": True, "fracao": round(float(spot.mean()), 4)}


def _refine_spilled(eng) -> bool:
    """A última chamada (refino) contra a penúltima (geração): passo por pixel muito mais lento = a VRAM
    transbordou para a memória do sistema (``memplan.refine_spilled``)."""
    calls = getattr(eng, "calls", None) or []
    if len(calls) < 2 or "s_por_passo" not in calls[-1] or "s_por_passo" not in calls[-2]:
        return False
    g, r = calls[-2], calls[-1]
    return MP.refine_spilled(g["s_por_passo"], g["tamanho"][0] * g["tamanho"][1], r["s_por_passo"],
                             r["tamanho"][0] * r["tamanho"][1])


def redesign(photo, funcao: str = "mudar_estilo", estilo: str | None = "moderno", ambiente: str = "sala",
             n: int = 4, seed: int | None = None, prompt: str = "", out_dir=None, quality: str = "final",
             evento: str | None = None, engine: Generator | None = None, depth: np.ndarray | None = None,
             groups=None, progress: Progress | None = None, device: str = "auto",
             overrides: dict | None = None, long_side: int | None = None, candidates: int | None = None,
             rank_fn=None, auto_empty: bool = True, repair: bool | None = None, refine: bool | None = None,
             tone: bool = True, refine_glass: bool | None = None, tree_fix: bool = True,
             retry_cut: bool = True) -> DesignResult:
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
    são procuradas de novo num recorte ampliado (``refine_openings``). ``tree_fix``: no Natal, a versão escolhida
    sem árvore ganha uma num canto livre (``_insert_tree``). ``retry_cut``: no "Mobiliar" com candidatos extras, se uma
    escolhida tem móvel cortado na borda, mais 2 sementes entram na disputa."""
    t_all = time.perf_counter()
    fp = P.with_overrides(P.function(funcao), **(overrides or {}))
    if fp.key in ("editar_area", "texto", "melhorar"):
        alt = {"editar_area": "edit_area", "texto": "text_to_design", "melhorar": "enhance.enhance"}[fp.key]
        raise ValueError(f"Use {alt} para '{fp.label}'.")
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
    if engine is None:
        _announce_downloads(prog, dev)
    qp = P.quality_profile(quality, dev)
    mplan = getattr(eng, "plan", None) or MP.current_plan(dev)
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
        # só os móveis soltos saem: armários, bancadas, estantes, louças e eletrodomésticos são a estrutura de uma
        # cozinha/banheiro (e da cozinha integrada de uma sala) — sem eles a "sala vazia" virava outro cômodo,
        # com outra câmera, e a cozinha integrada sumia
        hole = S.movable_mask(gr, (W, H), fixtures=False) & ~lock
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
    do_refine = bool((gpu and qp.key == "final" and mplan.refine) if refine is None else refine)
    if do_refine:
        # lado da passada de detalhe pelo plano de memória (nunca acima de 1,5× a geração; numa GPU de 8 GB, 1280)
        ref_side = min(int(round(max(W, H) * 1.5)), mplan.refine_long_side or 1536)
        Wf, Hf = P.work_size(rgb.shape[1], rgb.shape[0], max(ref_side, max(W, H)))
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
        # Natal: +2 candidatos e a escolha confere se há árvore (no teste real nenhuma das versões tinha)
        candidates = (n + 2 if (fp.key == "mobiliar" or (fp.event and evento == "natal"))
                      else n + 1 if fp.key in ("mudar_estilo", "reforma") else n) if gpu else n
    m = max(n, int(candidates))
    seeds = _seeds(seed, m)
    prompts = [P.build_prompt(fp.key, st_key, room_key, i, prompt, evento, windows=an.windows,
                              open_kitchen=an.open_kitchen) for i in range(m)]
    if hasattr(eng, "embeddings"):
        prog("geracao", 0.0, "Preparando o texto do estilo")
        eng.embeddings(prompts)
    tag = f"{fp.key}_{st_key or evento}"
    # o OneFormer da análise (janelas atrás de vidro) não fica na GPU enquanto o SDXL gera
    if seg is not None and mplan.seg_off_gpu and hasattr(seg, "park"):
        seg.park()
    free_cuda_cache(dev)
    ref_guidance = fp.guidance if mplan.refine_cfg else 1.0
    spilled = False
    do_ceiling = fp.keep_ceiling and gr is not None
    ceil_f = AL.lock_mask(gr, ("ceiling",), (Wf, Hf)) if do_ceiling else None
    checks = m > n or do_repair or do_ceiling
    span = 0.85 if checks else 1.0
    cand_dir = out / "candidatos" if m > n else out
    paths, images = [], []
    lock_info: dict = {}
    t_gen = time.perf_counter()
    def gen_one(i: int, sd: int, pr, total: int) -> tuple[str, np.ndarray]:
        nonlocal spilled, lock_info

        def cb(k, tot):
            prog("geracao", min(1.0, (i + k / max(tot, 1)) / total) * span, f"Gerando a versão {i + 1} de {total}")

        cb(0, 1)
        img = eng.generate(gen_base, gen_mask, control, pr, sd, fp.strength, steps, fp.guidance,
                           fp.control_scale, fp.control_end, cb)
        img = _resize(img, (W, H))
        if do_refine and not spilled:
            prog("geracao", min(1.0, (i + 1) / total) * span, f"Refinando os detalhes da versão {i + 1}")
            # na Reforma o refino não usa profundidade: a da foto traria os móveis antigos de volta e a da sala vazia
            # achataria os novos
            img = _refine(eng, img, (Wf, Hf), depth if struct_depth is None else None, pr, sd, ref_guidance)
            spilled = _refine_spilled(eng)
            if spilled:
                prog.emit({"event": "log", "level": "warning",
                           "message": "A placa de vídeo ficou sem memória na passada de detalhe: as próximas versões "
                                      "saem sem ela (mais rápido)."})
        elif do_refine:
            img = _resize(img, (Wf, Hf))
        if tone:
            img = AL.anchor_tone(img, base_f, amount_ab=fp.tone_ab,
                                 chroma_cap=1.15 if fp.tone_ab >= 0.3 else 1.5)
        if lock_f.any():
            img, lock_info = AL.lock_openings(img, base_f, lock_f)
        path = _save(img, cand_dir / f"{tag}_{i + 1}.jpg")
        prog.emit({"event": "preview", "path": path, "index": i})
        return path, img

    for i, (sd, pr) in enumerate(zip(seeds, prompts)):
        path, img = gen_one(i, sd, pr, m)
        paths.append(path)
        images.append(img)
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
            from .quality import TREE_WEIGHT, score, tree_score

            src_furn = (AL.lock_mask(gr, ("furniture",), (Wf, Hf)) if gr is not None else None)
            notas = [score(groups_of(k), lock_f | keep_f, furnish=fp.furnish, source_openings=free_f,
                           source_furniture=src_furn) for k in range(m)]
            if fp.event and evento == "natal":
                # verde que já estava na foto (plantas, a vista da janela) não conta como árvore
                photo_green = _green_mask(base_f)
                for k in range(m):
                    t = tree_score(images[k], exclude=photo_green | lock_f)
                    notas[k].update({f"arvore_{a}": b for a, b in t.items()})
                    notas[k]["nota"] = round(notas[k]["nota"] + TREE_WEIGHT * t["arvore"], 4)
            order = sorted(range(m), key=lambda k: -notas[k]["nota"])[:n]
            from .quality import CUT_RETRY

            if fp.key == "mobiliar" and retry_cut and any(notas[k]["cortados"] > CUT_RETRY for k in order):
                # móvel cortado na borda numa das escolhidas (poltrona/cadeiras cortadas na GPU): mais 2 sementes
                prog("geracao", 0.87, "Gerando versões extras (móvel cortado na borda)")
                if seg is not None and mplan.seg_off_gpu and hasattr(seg, "park"):
                    seg.park()
                for sd in _seeds(seeds[-1] + 1, 2):
                    i = len(seeds)
                    pr = P.build_prompt(fp.key, st_key, room_key, i, prompt, evento, windows=an.windows,
                                        open_kitchen=an.open_kitchen)
                    if hasattr(eng, "embeddings"):
                        eng.embeddings([pr])
                    path, img = gen_one(i, sd, pr, i + 1)
                    seeds.append(sd); prompts.append(pr); paths.append(path); images.append(img)
                    gen_groups.append(None)
                    notas.append(score(groups_of(i), lock_f | keep_f, furnish=fp.furnish, source_openings=free_f,
                                       source_furniture=src_furn))
                m = len(seeds)
                order = sorted(range(m), key=lambda k: -notas[k]["nota"])[:n]
                timings["extras_por_corte"] = 2
            timings["escolha_s"] = round(time.perf_counter() - t0, 2)
        if do_repair:
            t0 = time.perf_counter()
            for j, k in enumerate(order):
                prog("geracao", 0.88 + 0.06 * j / len(order), "Conferindo janelas e portas")
                lost_src = None
                new = AL.new_openings(groups_of(k), open_f | lock_f, (Wf, Hf), source_groups=gr)
                if new.any():
                    prog("geracao", 0.88 + 0.06 * (j + 0.5) / len(order), "Corrigindo uma janela inventada")
                    if seg is not None and mplan.seg_off_gpu and hasattr(seg, "park"):
                        seg.park()
                    fixed, info = _repair_openings(eng, images[k], new, lock_f, control_f, room_key, seeds[k],
                                                   steps, an.windows, keep_out=ceil_f, gen_size=(W, H))
                    if info:
                        # a versão antes do reparo fica em candidatos/ (conferência), com a área refeita
                        info["antes_do_reparo"] = _save(images[k], out / "candidatos" / f"{tag}_{k + 1}_sem_reparo.jpg")
                        _save((new.astype(np.uint8) * 255), out / "candidatos" / f"{tag}_{k + 1}_aberturas_novas.png")
                        if lock_f.any():
                            fixed, _ = AL.lock_openings(fixed, base_f, lock_f)
                        images[k] = fixed
                        repairs[str(k)] = info
                        lost_src = gen_groups[k]
                        gen_groups[k] = None          # a imagem mudou: o teto é conferido de novo
                # janela/cortina da foto que sumiu (virou parede, tijolo, armário): a original volta
                lost = AL.missing_openings(lost_src if lost_src is not None else groups_of(k), free_f, (Wf, Hf))
                if lost.any():
                    images[k], _ = AL.lock_openings(images[k], base_f, lost)
                    repairs.setdefault(str(k), {})["aberturas_perdidas"] = round(float(lost.mean()), 4)
            timings["reparo_s"] = round(time.perf_counter() - t0, 2)
        if fp.event and evento == "natal" and tree_fix:
            # a árvore é o que faz o Natal ser reconhecido: versão escolhida sem árvore ganha uma num canto livre
            from .quality import tree_score

            t0 = time.perf_counter()
            photo_green = _green_mask(base_f)
            for j, k in enumerate(order):
                t = tree_score(images[k], exclude=photo_green | lock_f)
                if t["arvore"] >= TREE_MIN_SCORE:
                    continue
                prog("geracao", 0.9 + 0.04 * j / len(order), "Colocando a árvore de Natal")
                if seg is not None and mplan.seg_off_gpu and hasattr(seg, "park"):
                    seg.park()
                new_img, info = _insert_tree(eng, images[k], groups_of(k), lock_f, seeds[k], steps, gen_size=(W, H))
                if info.get("inserida"):
                    if lock_f.any():
                        new_img, _ = AL.lock_openings(new_img, base_f, lock_f)
                    info["arvore_antes"] = t["arvore"]
                    info["arvore_depois"] = tree_score(new_img, exclude=photo_green | lock_f)["arvore"]
                    images[k] = new_img
                    gen_groups[k] = None
                repairs.setdefault(str(k), {})["arvore"] = info
            timings["arvore_s"] = round(time.perf_counter() - t0, 2)
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
           "passos": steps, "refino": do_refine, "refino_desligado_por_memoria": spilled, "tom": tone,
           "plano_memoria": mplan.to_dict(), "preset": asdict(fp),
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


EDIT_OLD_GROW = 0.012     # quanto o objeto antigo cresce (× diagonal) para sair com a sombra e os pés finos
EDIT_CONTEXT = 0.6        # margem de contexto do recorte (× o lado da área): o gerador precisa "ver" a sala
EDIT_MIN_CROP = 0.5       # o recorte cobre pelo menos esta fração do lado maior da foto


def edit_gen_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Tamanho da geração da "Editar área" com a proporção da FOTO (múltiplos de 8): numa foto 3:2 (960×640) o
    recorte saía 4:3 e era gerado a 1024×768 — a área esticada/achatada na volta. Proporção fiel em até ~1%."""
    s = long_side / max(width, height)
    w = max(64, int(round(width * s / 8)) * 8)
    h = max(64, int(round(height * s / 8)) * 8)
    return w, h


def edit_crop_box(mask: np.ndarray, aspect: float, margin_frac: float = EDIT_CONTEXT, min_side: int = 0
                  ) -> tuple[int, int, int, int]:
    """Recorte da "Editar área": a área pintada + contexto (``margin_frac`` × o lado dela, no mínimo ``min_side``
    no lado maior), com a proporção ``aspect`` (largura/altura) da geração e dentro da foto. A área pintada fica
    sempre inteira dentro do recorte (se não couber na proporção, o recorte é a foto toda)."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0, W, H
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cw = (bx1 - bx0) * (1 + 2 * margin_frac)
    ch = (by1 - by0) * (1 + 2 * margin_frac)
    cw = max(cw, ch * aspect)
    ch = cw / aspect
    if 0 < max(cw, ch) < min_side:
        k = min_side / max(cw, ch)
        cw, ch = cw * k, ch * k
    if cw > W or ch > H:                              # não cabe: o maior recorte com a proporção da foto
        k = min(W / cw, H / ch)
        cw, ch = cw * k, ch * k
    cw, ch = int(round(min(W, cw))), int(round(min(H, ch)))
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    X0 = int(np.clip(round(cx - cw / 2), 0, W - cw))
    Y0 = int(np.clip(round(cy - ch / 2), 0, H - ch))
    X1, Y1 = X0 + cw, Y0 + ch
    if bx0 < X0 or by0 < Y0 or bx1 > X1 or by1 > Y1:   # a área não coube: a foto inteira
        return 0, 0, W, H
    return X0, Y0, X1, Y1


OBJECT_GROUPS = ("furniture", "fixture", "ceiling_fixture", "window_covering")


def _objects(groups: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    g = np.asarray(groups)
    if (g.shape[1], g.shape[0]) != tuple(size):
        g = cv2.resize(g, tuple(size), interpolation=cv2.INTER_NEAREST)
    return np.isin(g, [GROUP_IDS_[k] for k in OBJECT_GROUPS])


def _split_target(objs: np.ndarray, area: np.ndarray, inside: float = 0.6, ids: np.ndarray | None = None,
                  clip: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Objetos da foto -> (alvo, vizinhos). Alvo = o que está majoritariamente (``inside``) dentro da área pintada
    (a mesa que o usuário quer trocar); vizinhos = o que só encosta na área (o sofá e a poltrona que o retângulo
    pegou de raspão) — esses ficam intactos."""
    a = np.asarray(area, bool)
    o = np.asarray(objs, bool)
    # móveis encostados (a mesa no sofá) viram um componente só: separados por uma erosão antes de classificar
    r = max(1, int(0.012 * float(np.hypot(*a.shape))))
    target = np.zeros(a.shape, bool)
    protect = np.zeros(a.shape, bool)
    # com os rótulos finos, cada rótulo (sofá, mesa de centro...) separado; sem eles, os grupos erodidos
    parts = ([o & (ids == v) for v in np.unique(ids[o])] if ids is not None else [o])
    for part in parts:
        n, lab = cv2.connectedComponents(cv2.erode(part.astype(np.uint8), AL._disk(r)), connectivity=8)
        for k in range(1, n):
            c = lab == k
            if float((c & a).sum()) >= inside * float(c.sum()):
                target |= c
            else:
                protect |= c
    grow = AL._disk(r + 1)
    protect = cv2.dilate(protect.astype(np.uint8), grow).astype(bool) & o
    target = cv2.dilate(target.astype(np.uint8), grow).astype(bool) & o & ~protect
    return (target & a if clip else target), protect


def _object_only(g: np.ndarray, photo: np.ndarray, plate: np.ndarray, area: np.ndarray, old_obj: np.ndarray,
                 new_obj: np.ndarray, min_frac: float = 0.03, max_frac: float = 0.85,
                 protect: np.ndarray | None = None, leftover: list | None = None
                 ) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Só o OBJETO novo entra (com a sombra): numa área retangular o gerador refazia também o tapete/piso em volta
    e a borda do retângulo aparecia (o tapete estampado de fora contra um piso liso por dentro). Aqui, dentro da
    área: o objeto novo (segmentado na imagem gerada, crescido ~3% para a sombra de contato) entra por Poisson
    (a luz do piso em volta casa); onde estava o objeto antigo e o novo não cobre, entra a foto sem ele
    (``plate``; essa área vai em ``leftover``, se dado, para a passada de textura); o resto fica a FOTO. Objeto novo pequeno demais ou ocupando quase toda a área (troca de estilo
    de uma região, parede, piso) = a área inteira gerada, como antes. Devolve (imagem, peso da foto ou None, modo)."""
    a = np.asarray(area, bool)
    dw = float(np.hypot(*a.shape))
    keep = np.zeros(a.shape, bool)
    if protect is not None and protect.any():
        # móveis da foto que a área só pegou de raspão (o braço do sofá): ficam os da foto, em QUALQUER modo — no
        # teste na GPU a remoção borrava a área inteira, braço do sofá incluído
        keep = cv2.dilate(protect.astype(np.uint8), AL._disk(2)).astype(bool)
        a = a & ~keep
    new = _main_object(new_obj & a, a)
    frac = float(new.sum()) / max(1.0, float(a.sum()))
    if frac < min_frac or frac > max_frac:
        kp = (AL.feather_alpha(keep & np.asarray(area, bool), max(1.5, 0.004 * dw), inside=1.0)
              if keep.any() else None)
        return g, kp, "area"
    region = cv2.dilate(new.astype(np.uint8), AL._disk(max(2, int(0.02 * dw)))).astype(bool) & a
    # só o objeto antigo (segmentado) crescido um POUCO sai: o crescimento de 3,5% pegava o braço do sofá
    old = cv2.dilate(old_obj.astype(np.uint8), AL._disk(max(2, int(EDIT_OLD_GROW * dw)))).astype(bool) & a & ~region
    top = photo
    a_old = AL.feather_alpha(old, max(2.0, 0.01 * dw), inside=0.5) if old.any() else np.zeros(a.shape, np.float32)
    if old.any():
        top = AL.composite(top, plate, a_old)
        if leftover is not None:
            leftover.append(old)
    # margem da sombra: Poisson MISTO (a estampa do tapete da foto continua por ela, só a luz/sombra do gerador
    # entra: sem o "halo" de piso liso em volta do objeto); o objeto em si: Poisson normal
    pasted = AL.poisson_blend(top, g, region, mixed=True)
    # o objeto em si entra OPACO (pixels do gerador, só a borda suavizada): o Poisson no objeto puxava as cores
    # dele para as do tapete e a mesa saía semitransparente (o "tronco" translúcido do teste na GPU)
    core = cv2.dilate(new.astype(np.uint8), AL._disk(max(1, int(0.004 * dw)))).astype(bool) & region
    pasted = AL.composite(pasted, g, AL.feather_alpha(core, max(1.5, 0.004 * dw), inside=1.0))
    a_new = AL.feather_alpha(region, max(2.0, 0.012 * dw), inside=1.0)
    # peso da FOTO original (em resolução cheia, sem a ida e volta pelo tamanho da geração): 1 - (novo ∪ antigo)
    return AL.composite(top, pasted, a_new), 1.0 - np.maximum(a_new, a_old), "objeto"


def _main_object(new: np.ndarray, area: np.ndarray, min_part: float = 0.15) -> np.ndarray:
    """O objeto novo = as partes grandes (≥ ``min_part`` da maior) dos móveis segmentados na área: pedacinhos
    soltos (um resto de sofá redesenhado na borda, um vaso) não entram colados por cima da foto."""
    m = np.asarray(new, bool)
    if not m.any():
        return m
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    areas = st[1:, cv2.CC_STAT_AREA]
    big = 1 + np.nonzero(areas >= min_part * areas.max())[0]
    return np.isin(lab, big)


def _edit_prefill(img: np.ndarray, hole: np.ndarray, inpainter=None, auto: bool = False, device: str = "cpu"
                  ) -> tuple[np.ndarray, str]:
    """Ponto de partida da "Editar área": a área sem o objeto antigo. Com o removedor de objetos do modo B (pesos
    Apache-2.0 já baixados pelo instalador) o tapete/piso CONTINUA dentro do retângulo e, com força < 1, a geração
    parte dele: o objeto novo pousa no mesmo tapete (no teste em CPU o retângulo virava piso de madeira com borda
    reta no tapete). Sem ele (pesos ausentes, testes), Telea + desfoque: só a cor e a luz do entorno.
    ``inpainter(rgb, máscara) -> rgb``; ``auto``: carrega o removedor se nenhum foi dado."""
    dw = float(np.hypot(*hole.shape))
    if inpainter is None and auto:
        try:
            from ..removal.plate import load_lama

            lama = load_lama(device if device.startswith("cuda") else "cpu", offline=True)
            try:
                return lama(img, hole), "remocao"
            finally:
                lama.release()
                free_cuda_cache(device)
        except Exception:  # noqa: BLE001 - sem os pesos: Telea
            pass
    elif inpainter is not None:
        return inpainter(img, hole), "remocao"
    base = cv2.inpaint(img, hole.astype(np.uint8) * 255, max(3, int(0.01 * dw)), cv2.INPAINT_TELEA)
    return AL.composite(img, cv2.GaussianBlur(base, (0, 0), 0.01 * dw), hole.astype(np.float32)), "telea"


EDIT_FOOTPRINT_MIN = 0.25  # objeto antigo cobrindo isto da área pintada: o novo nasce na pegada dele
EDIT_FOOTPRINT_GROW = 0.03 # folga da pegada (× diagonal): o novo pode ser um pouco maior que o antigo
EDIT_SAM_MIN = 0.002      # componentes do objeto antigo a partir desta fração do recorte vão ao SAM


def _sam_complete(rgb: np.ndarray, obj: np.ndarray, allowed: np.ndarray, sam_fn) -> tuple[np.ndarray, int]:
    """Completa o objeto antigo com o SAM 2.1 (caixa de cada parte, um pouco maior para BAIXO: pés finos e a base
    cromada que a segmentação semântica deixa de fora e que apareciam embaixo do objeto novo na GPU). Das 3
    máscaras do SAM fica a maior que cobre a parte segmentada (≥ 85%) sem passar de 3× o tamanho dela; tudo
    limitado a ``allowed`` (a área pintada sem os móveis vizinhos). Devolve (máscara, quantas partes cresceram)."""
    o = np.asarray(obj, bool)
    H, W = o.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(o.astype(np.uint8), connectivity=8)
    comps, boxes = [], []
    for k in range(1, n):
        x, y, w, h, a = (int(v) for v in st[k])
        if a < EDIT_SAM_MIN * H * W:
            continue
        mx, my = 0.04 * w + 2, 0.04 * h + 2
        boxes.append([max(0.0, x - mx), max(0.0, y - my), min(W - 1.0, x + w + mx), min(H - 1.0, y + h + 0.12 * h + my)])
        comps.append(lab == k)
    if not boxes:
        return o, 0
    try:
        probs = sam_fn(rgb, boxes)
    except Exception:  # noqa: BLE001 - sem SAM (pesos ausentes): fica a segmentação
        return o, 0
    out, grown = o.copy(), 0
    for c, pr in zip(comps, probs):
        best = None
        for m in np.asarray(pr):
            m = m > 0.5
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            cov = float((m & c).sum()) / max(1.0, float(c.sum()))
            if cov >= 0.85 and m.sum() <= 3.0 * c.sum() and (best is None or m.sum() > best.sum()):
                best = m
        if best is not None and (best & ~out).any():
            out |= best & np.asarray(allowed, bool)
            grown += 1
    return out, grown


def _auto_sam(device: str):
    from ..removal.instances import sam2_box_fn

    return sam2_box_fn(device if device.startswith("cuda") else "cpu", offline=True)


PLATE_REFINE_STRENGTH = 0.5     # textura nova sobre o preenchimento do removedor (a cor e o desenho ficam)
PLATE_REFINE_MIN = 0.002        # sobra do objeto antigo (fração do recorte) a partir da qual vale a passada


def _refine_plate(eng, img: np.ndarray, left: np.ndarray, control, room_key: str, seed: int, steps: int,
                  fp) -> tuple[np.ndarray, dict]:
    """Onde o objeto antigo estava e o novo não cobre, o removedor deixa um borrão liso (o "borrão cinza no
    tapete" visto na GPU). Uma passada leve do gerador SÓ ali (força 0,5, máscara suave): o desenho e a cor do
    preenchimento ficam, a textura do tapete/piso volta. Entra com rampa curta; fora da sobra nada muda."""
    m = np.asarray(left, bool)
    if float(m.mean()) < PLATE_REFINE_MIN:
        return img, {}
    dw = float(np.hypot(*m.shape))
    md = cv2.dilate(m.astype(np.uint8), AL._disk(max(2, int(0.006 * dw)))).astype(bool)
    weight = AL.feather_alpha(md, max(3.0, 0.012 * dw), inside=0.5)
    r = P.ROOMS[room_key]
    pr = P.DesignPrompt(f"seamless continuation of the surrounding floor and rug of the {r['en']}, same pattern, "
                        "same colors, same texture and lighting, photorealistic, sharp focus, high detail",
                        "furniture, table, object, blur, smudge, blurry patch, grey stain, seam, visible border, "
                        + P.NEGATIVE_BASE)
    g = eng.generate(img, weight, control, pr, int(seed) + 202, PLATE_REFINE_STRENGTH, steps, 1.8,
                     fp.control_scale, fp.control_end, None)
    g = _resize(g, (img.shape[1], img.shape[0]))
    a = AL.feather_alpha(md, max(2.0, 0.006 * dw), inside=1.0)
    return AL.composite(img, g, a), {"fracao": round(float(m.mean()), 4)}


def edit_area(photo, mask, prompt: str = "", estilo: str | None = None, ambiente: str = "sala", n: int = 4,
              seed: int | None = None, out_dir=None, quality: str = "final", engine: Generator | None = None,
              depth: np.ndarray | None = None, progress: Progress | None = None, device: str = "auto",
              overrides: dict | None = None, feather_px: float | None = None, grow_frac: float = 0.01,
              inpainter=None, seg_fn=None, object_only: bool = True, sam_fn=None,
              refine_plate: bool = True) -> DesignResult:
    """Refaz só a área pintada pelo usuário (``mask``: PNG ou array, branco = editar), guiada por ``prompt`` (em
    português ou inglês: ``glossary.to_english``) e/ou ``estilo``.

    A região é recortada com bastante contexto (``EDIT_CONTEXT``: o gerador vê o piso, as paredes e os móveis em
    volta e põe o objeto novo na mesma perspectiva e luz), gerada na resolução do SDXL e colada de volta na foto
    ORIGINAL (resolução cheia). A geração usa uma máscara SUAVE nos latentes (peso 1 no miolo, rampa na borda):
    a cada passo a borda mistura a foto com o que está sendo gerado, então o tapete/piso continua pela área nova
    em vez de terminar num "remendo oval". A área é pré-preenchida com o entorno (o objeto antigo não semeia). A
    colagem final usa uma rampa por dentro da máscara crescida ``grow_frac`` da diagonal: todo pixel fora dela fica
    idêntico à foto.

    Travas do teste na GPU (uma "mini-sala" inteira com sofá, cortina e janela dentro do retângulo, gerada a
    1024×768 para uma foto 3:2): o recorte e a geração têm a proporção da FOTO (``edit_crop_box``/``edit_gen_size``,
    nada esticado na volta), a profundidade do entorno guia (``control_scale`` do preset, igual na GPU e na CPU) e
    o prompt é só do objeto pedido, com "sala inteira"/móveis não pedidos no negativo (``presets.build_prompt``).

    0.9.1 (teste na GPU: base cromada da mesa antiga aparecendo e borrão cinza no tapete): o objeto antigo é
    completado pelo SAM 2.1 (``sam_fn(rgb, caixas)``; ``None`` = o do disco quando o motor é o real) antes de sair,
    e onde ele estava e o novo não cobre o gerador refaz a textura por cima do preenchimento (``refine_plate``)."""
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
    W, H = edit_gen_size(Wf, Hf, qp.long_side)
    x0, y0, x1, y1 = edit_crop_box(ue, W / H, EDIT_CONTEXT, min_side=int(EDIT_MIN_CROP * max(Wf, Hf)))
    crop, cm = rgb[y0:y1, x0:x1], ue[y0:y1, x0:x1]
    crop_ref = _resize(crop, (W, H))
    mw = cv2.resize(cm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    dw = float(np.hypot(W, H))
    # peso da geração nos latentes: 1 dentro da área, rampa de ~3% da diagonal metade por dentro, metade por fora
    weight = AL.feather_alpha(mw, max(4.0, 0.03 * dw), inside=0.5)
    weight[mw & (weight < 1.0) & (cv2.erode(mw.astype(np.uint8), AL._disk(2)) > 0)] = 1.0
    hole = weight > 0.5
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
    seg, sfn = None, seg_fn
    old_obj, plate, base, fill_kind = None, None, None, ""
    protect = None
    sam_grown = 0
    area_grown = 0
    footprint = False
    plate_refined: list = []
    modes: list[str] = []
    try:
        if object_only and (prompt or "").strip():
            if sfn is None and engine is None:
                try:
                    from .quality import Segmenter

                    seg = Segmenter(dev)
                    sfn = seg.groups
                except Exception:  # noqa: BLE001 - sem segmentação: a área inteira
                    sfn = None
            if sfn:
                # o objeto antigo da área e a foto SEM ele (só ele sai: o tapete em volta fica nítido; o
                # preenchimento da área inteira é um borrão)
                ids = None
                if seg is not None:              # rótulos finos: mesa e sofá encostados têm rótulos diferentes
                    from ..analysis.segmentation import labels_to_groups

                    lab0 = seg.labels(crop_ref)
                    g0 = labels_to_groups(lab0, seg.id2label)
                    ids = cv2.resize(lab0, (W, H), interpolation=cv2.INTER_NEAREST)
                else:
                    g0 = sfn(crop_ref)
                old_full, protect = _split_target(_objects(g0, (W, H)), mw, ids=ids, clip=False)
                keep = cv2.dilate(protect.astype(np.uint8), AL._disk(2)).astype(bool)
                old_obj = old_full & mw
                if old_full.any():
                    sfn_sam = sam_fn if sam_fn is not None else (_auto_sam(dev) if engine is None else None)
                    if sfn_sam is not None:
                        old_full, sam_grown = _sam_complete(crop_ref, old_full, ~keep, sfn_sam)
                    # o objeto antigo sai INTEIRO, mesmo a parte que a pincelada não cobriu (a perna cromada fora do
                    # retângulo ficava na foto): a área cresce até ele, se o que falta não passa da metade dela
                    og = cv2.dilate(old_full.astype(np.uint8), AL._disk(max(2, int(EDIT_OLD_GROW * dw)))).astype(bool)
                    ext = og & ~mw & ~keep
                    if ext.any() and ext.sum() <= 0.5 * mw.sum():
                        mw = mw | ext
                        ext_full = cv2.resize(ext.astype(np.uint8), (x1 - x0, y1 - y0),
                                              interpolation=cv2.INTER_NEAREST).astype(bool)
                        ue[y0:y1, x0:x1] |= ext_full
                        area_grown = int(ext_full.sum())
                    old_obj = old_full & mw
                    # o objeto novo nasce na pegada do antigo (mesma escala e lugar), não no retângulo inteiro: com a
                    # área toda o gerador enchia o retângulo (mesa enorme por cima da poltrona, colagem "area")
                    if old_obj.sum() >= EDIT_FOOTPRINT_MIN * mw.sum():
                        foot = cv2.dilate(old_obj.astype(np.uint8),
                                          AL._disk(max(2, int(EDIT_FOOTPRINT_GROW * dw)))).astype(bool) & mw & ~keep
                        weight = AL.feather_alpha(foot, max(4.0, 0.03 * dw), inside=0.5)
                        weight[foot & (cv2.erode(foot.astype(np.uint8), AL._disk(2)) > 0)] = 1.0
                        hole = weight > 0.5
                        footprint = True
                    ring = cv2.dilate(mw.astype(np.uint8), AL._disk(max(2, int(0.025 * dw)))).astype(bool) & ~mw
                if old_obj is not None and old_obj.any():
                    # SÓ o objeto antigo sai (crescido um pouco para os pés finos e a sombra), nunca o móvel vizinho
                    # que o retângulo pegou: o removedor na área inteira deixava um borrão por cima do braço do sofá
                    od = cv2.dilate(old_obj.astype(np.uint8),
                                    AL._disk(max(2, int(EDIT_OLD_GROW * dw)))).astype(bool) & mw & ~keep
                    plate, fill_kind = _edit_prefill(crop_ref, od, inpainter, auto=engine is None, device=dev)
                    # a geração parte da foto SEM o objeto antigo (nítida no resto): a borda suave dos latentes
                    # mistura a foto de verdade, não um borrão
                    base = plate
    except Exception:  # noqa: BLE001
        sfn = None
    if base is None:
        # sem objeto antigo reconhecido: a área é pré-preenchida com o entorno (só a luz e a cor ficam)
        base, fill_kind = _edit_prefill(crop_ref, hole, inpainter, auto=engine is None, device=dev)
    if plate is None:
        plate = base
    if area_grown:
        alpha = AL.feather_alpha(ue, fr, inside=1.0)
    t_gen = time.perf_counter()
    try:
        for i, (sd, pr) in enumerate(zip(seeds, prompts)):
            def cb(k, tot, i=i):
                prog("geracao", (i + k / max(tot, 1)) / len(seeds), f"Gerando a versão {i + 1} de {len(seeds)}")

            cb(0, 1)
            if seg is not None and hasattr(seg, "park"):
                seg.park()
            g = eng.generate(base, weight, control, pr, sd, fp.strength, steps, fp.guidance, fp.control_scale,
                             fp.control_end, cb)
            g = _resize(g, (W, H))
            keep_photo, mode = None, "area"
            if sfn and old_obj is not None:
                left: list = []
                g, keep_photo, mode = _object_only(g, crop_ref, plate, mw, old_obj, _objects(sfn(g), (W, H)),
                                                   protect=protect, leftover=left)
                modes.append(mode)
                if refine_plate and left and mode == "objeto":
                    if seg is not None and hasattr(seg, "park"):
                        seg.park()
                    g, pinfo = _refine_plate(eng, g, left[0], control, room_key, sd, steps, fp)
                    if pinfo:
                        plate_refined.append(pinfo)
            if mode == "area":
                g = _match_edge(g, crop_ref, mw, ring)    # sem halo: a luz da borda nova casa com a do entorno
            g = _resize(g, (x1 - x0, y1 - y0))
            if keep_photo is not None:
                kp = cv2.resize(keep_photo, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                g = AL.composite(g, crop, np.clip(kp, 0, 1))
            top = rgb.copy()
            top[y0:y1, x0:x1] = g
            img = AL.composite(rgb, top, alpha)
            path = _save(img, out / f"editar_area_{i + 1}.jpg")
            outputs.append(path)
            images.append(img)
            prog.emit({"event": "preview", "path": path, "index": i})
    finally:
        if seg is not None:
            seg.close()
    timings["geracao_s"] = round(time.perf_counter() - t_gen, 2)
    timings["por_imagem_s"] = round(timings["geracao_s"] / len(seeds), 2)
    prog("final")
    before = _save(rgb, out / "antes.jpg")
    timings["total_s"] = round(time.perf_counter() - t_all, 2)
    rep = {"funcao": "editar_area", "estilo": st_key, "ambiente": room_key, "recorte": [x0, y0, x1, y1],
           "proporcao": {"foto": round(Wf / Hf, 4), "recorte": round((x1 - x0) / (y1 - y0), 4),
                         "geracao": round(W / H, 4)},
           "margem_px": grow, "preenchimento": fill_kind, "colagem": modes or None,
           "sam_partes": sam_grown, "textura_refeita": plate_refined or None, "area_crescida_px": area_grown,
           "pegada_do_objeto_antigo": footprint,
           "tamanho_geracao": [W, H], "tamanho": [Wf, Hf], "sementes": seeds, "passos": steps,
           "preset": asdict(fp), "prompts": [asdict(p) for p in prompts], "saidas": outputs, "tempos": timings}
    report = out / "design.json"
    report.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    return DesignResult(outputs, before, str(report), "editar_area", st_key, room_key, seeds, (Wf, Hf), timings,
                        images)


# ====================================================================================== texto para ambiente
ASPECTS = P.ASPECTS


def text_to_design(prompt: str = "", estilo: str | None = "moderno", ambiente: str = "sala", n: int = 4,
                   seed: int | None = None, out_dir=None, quality: str = "final", aspect: str = "paisagem",
                   engine: Generator | None = None, progress: Progress | None = None, device: str = "auto",
                   overrides: dict | None = None) -> DesignResult:
    """Cria o ambiente do zero a partir da descrição + tipo de ambiente + estilo (sem foto)."""
    t_all = time.perf_counter()
    fp = P.with_overrides(P.function("texto"), **(overrides or {}))
    room_key = P.norm_room(ambiente)
    # o ambiente citado no texto vence o escolhido na lista ("quarto aconchegante" com --ambiente sala = quarto)
    room_said = P.infer_room(prompt)
    room_asked = room_key
    if room_said and room_said != room_key:
        room_key = room_said
    aspect = P.norm_aspect(aspect)
    st_key = P.norm_style(estilo) if estilo else None
    prog = _Prog(progress, (("preparo", "Preparando", 0.03), ("geracao", "Criando o ambiente", 0.92),
                            ("final", "Finalizando", 0.05)))
    out = _out_dir(out_dir)
    prog("preparo")
    eng = engine or get_engine(device)
    dev = getattr(eng, "device", pick_device(device))
    qp = P.quality_profile(quality, dev)
    aw, ah = ASPECTS[aspect]
    W, H = edit_gen_size(aw * 1000, ah * 1000, qp.long_side)        # proporção exata do formato (múltiplos de 8)
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
    rep = {"funcao": "texto", "estilo": st_key, "ambiente": room_key, "ambiente_pedido": room_asked,
           "ambiente_do_texto": room_said, "formato": aspect, "tamanho": [W, H], "sementes": seeds,
           "passos": steps, "preset": asdict(fp), "prompts": [asdict(p) for p in prompts], "saidas": outputs,
           "tempos": timings}
    report = out / "design.json"
    report.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    return DesignResult(outputs, None, str(report), "texto", st_key, room_key, seeds, (W, H), timings, images)


__all__ = ["redesign", "edit_area", "text_to_design", "DesignResult", "depth_control", "estimate_depth",
           "estimate_groups", "furniture_coverage", "STAGES", "ASPECTS", "EMPTY_NOTICE"]
