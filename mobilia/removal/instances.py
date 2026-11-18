"""Instâncias de móveis da foto mobiliada (modo B, PLANO_V3 §3.1 e §3.6) e detecção automática do modo.

Etapas (todas em numpy, exceto as duas chamadas de modelo, que podem ser trocadas por stubs nos testes):

1. **Candidatas:** OneFormer ADE20K *panoptic* (sem fundir rótulos) ∪ componentes da segmentação
   *semântica* que o panoptic perdeu (no teste de Nice, a mesa de jantar só saiu na semântica).
   Uma instância panóptica feita de partes grandes e distantes é separada (2 tapetes fundidos).
2. **Refino por caixa com SAM 2.1:** a melhor das 3 máscaras do SAM é aceita se o IoU contra a máscara do
   OneFormer for >= 0,6; senão fica a do OneFormer (suavizada). Fica o maior componente conexo (mais os
   pedaços confirmados pelo OneFormer, como pernas de cadeira) e o alpha < 0,05 fora da máscara dilatada
   é zerado.
3. **Fusão de fragmentos:** dois pedaços com o mesmo rótulo (um deles vindo da semântica) cujo fecho
   convexo está coberto por móveis viram uma peça só (perna de mesa separada pela cadeira).
4. **Regra solto x planejado (§3.6, decisão do usuário):** no modo B saem os móveis soltos **e** os fixos
   (armários planejados, ilha, bancada, coifa, eletros embutidos, louças, gabinete, espelho, box).
   Ficam portas, janelas, cortinas, nichos, tomadas, teto, luminárias de teto e de parede, lareira,
   radiador, escadas e acabamentos de piso e parede (essas classes nunca viram instância). A distinção
   solto/planejado continua necessária porque o portão de qualidade refaz a placa **mantendo os
   planejados** (só os soltos saem). Como o ADE20K não separa armário embutido de solto, as classes
   ambíguas passam por uma regra geométrica: até o teto, suspenso na parede, largura da parede inteira
   ou encostado num conjunto de cozinha/banheiro significa planejado.
5. **Modo:** foto com móveis (área e número de peças removíveis acima do limite) vai para o modo B; foto
   vazia vai para o modo A. Uma chave manual pode forçar o modo.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import ndimage

SEG_MODEL = "shi-labs/oneformer_ade20k_swin_large"

# --- Classes ADE20K pelo nome base (antes da 1ª vírgula). Tudo o que não está aqui fica na foto. ---------------
LOOSE_CLASSES = frozenset({
    "bed", "table", "chair", "sofa", "armchair", "seat", "desk", "lamp", "cushion", "pillow", "rug",
    "coffee table", "swivel chair", "bench", "ottoman", "stool", "pool table", "tv", "crt screen", "computer",
    "monitor", "box", "plant", "flower", "vase", "pot", "basket", "painting", "poster", "clock", "bag",
    "trash can", "clothes", "blanket", "book", "towel", "bicycle", "plaything", "bottle", "sculpture", "tray",
    "glass", "plate", "food", "cradle", "ball", "barrel", "arcade machine", "base",
})
# Fixos que também saem no modo B (cenário cru); voltam no vídeo antes dos soltos.
BUILTIN_CLASSES = frozenset({
    "counter", "countertop", "kitchen island", "bar", "sink", "toilet", "tub", "shower", "stove", "oven",
    "refrigerator", "dishwasher", "microwave", "hood", "washer",
})
# Podem ser soltos ou planejados: decide a regra geométrica (padrão: solto; espelho: planejado).
AMBIGUOUS_CLASSES = frozenset({
    "cabinet", "shelf", "wardrobe", "chest of drawers", "bookcase", "buffet", "case", "mirror",
})
REMOVABLE_CLASSES = LOOSE_CLASSES | BUILTIN_CLASSES | AMBIGUOUS_CLASSES
# Ficam sempre na foto (§3.6): o SAM não pode "comer" estes pixels fora da máscara do OneFormer (ex.: a caixa da
# cama pega a cortina ao lado e o SAM a inclui com IoU ainda aceitável).
PROTECTED_CLASSES = frozenset({
    "curtain", "blind", "window", "door", "double door", "screen door", "sconce", "chandelier", "light", "fan",
    "fireplace", "radiator", "stairs", "stairway", "step", "bannister", "rail", "column", "person", "animal",
})
# Peças "grandes" que, sozinhas, já caracterizam a foto como mobiliada.
MAJOR_CLASSES = frozenset({
    "bed", "sofa", "armchair", "table", "coffee table", "desk", "chair", "swivel chair", "bench", "wardrobe",
    "chest of drawers", "bookcase", "cabinet", "buffet", "kitchen island", "counter", "countertop", "tub",
    "toilet", "sink", "refrigerator", "stove", "pool table", "rug", "ottoman", "shelf",
})

KIND_LOOSE = "solto"
KIND_BUILTIN = "planejado"

# --- Limites (frações da área da imagem, salvo indicação) ------------------------------------------------------
MIN_AREA_PANOPTIC = 0.0008
MIN_AREA_SEMANTIC = 0.002
SPLIT_MIN_PART = 0.001          # parte mínima para separar instância panóptica fundida
SPLIT_MIN_GAP = 0.04            # distância mínima entre as partes (fração da diagonal)
SAM_MIN_IOU = 0.6
SAM_MAX_GROW = 0.015            # o SAM só acrescenta pixels até esta distância da máscara do OneFormer (fração da diagonal)
ALPHA_CUT = 0.05
MERGE_HULL_COVER = 0.7
MERGE_VERTICAL_OVERLAP = 0.5
# Detecção do modo
MODE_B_AREA = 0.04              # área removível total >= 4% → modo B
MODE_B_PIECES = 3               # ou >= 3 peças com área >= 0,5%
MODE_B_PIECE_AREA = 0.005
MODE_B_MAJOR_AREA = 0.015       # ou uma peça "grande" >= 1,5% com área total >= 2%
MODE_B_MAJOR_TOTAL = 0.02
MODE_B_LOOSE_AREA = 0.01        # sem móvel solto (>= 1%), só planejados não bastam ...
MODE_B_BUILTIN_ONLY = 0.10      # ... a não ser que cubram >= 10% (cozinha/banheiro completos)
MAX_REMOVAL_AREA = 0.75         # acima disto quase toda a sala vazia é gerada (aviso)
# Painel fixo na parede (cabeceira-painel de hotel, painel de TV/ripado): pontos no plano da parede (<= 15 cm) e
# acima do tampo/colchão (> 75 cm). Tirar isso obriga a placa a inventar a parede inteira atrás dele.
PANEL_WALL_DIST = 0.15          # m
PANEL_MIN_Z = 0.75              # m acima do piso
PANEL_PARENTS = frozenset({"bed", "desk"})   # o painel vem colado na máscara destas peças: é recortado e fica
PANEL_MIN_FRAC = 0.10           # ... se for >= 10% da peça
PANEL_MIN_AREA = 0.005          # ... e >= 0,5% da foto
INSIDE_BUILTIN_MAX_AREA = 0.06  # objeto guardável até 6% da foto ...
INSIDE_BUILTIN_RING = 0.5       # ... com >= 50% do anel em volta num planejado está guardado nele (livro na estante)
STORED_CLASSES = frozenset({    # o que se guarda num armário/estante (assento e mesa nunca: saem sempre)
    "book", "vase", "plant", "flower", "pot", "box", "basket", "bottle", "glass", "plate", "tray", "food",
    "clock", "sculpture", "towel", "clothes", "bag", "plaything", "blanket",
})
DESK_PANEL_Z = 1.1              # m: uma mesa com >= 20% da máscara acima disto está presa a um painel alto
DESK_PANEL_FRAC = 0.2
DESK_WALL_DEPTH = 0.8           # m: profundidade de uma bancada de trabalho encostada
DESK_WALL_MIN_LEN = 2.0         # m: só paredes longas (mesa solta de 1,2 m numa parede curta continua solta)
PANEL_WHOLE_FRAC = 0.5          # classe ambígua (armário) com >= 50% no plano da parede: é um painel → planejado


def base_name(label: str) -> str:
    return label.split(",")[0].strip().lower()


@dataclass
class PanopticResult:
    """Saída panóptica compacta: ``ids`` (H, W) uint8 (0 = nada, k = linha k-1) e ``segments`` (N, 4) com
    [k, label_id ADE20K, score, área em px] (o mesmo formato de ``tests/fixtures/rooms/*/panoptic.npz``)."""
    ids: np.ndarray
    segments: np.ndarray

    def save(self, path) -> None:
        np.savez_compressed(path, ids=self.ids, segments=self.segments, model_id=np.array(SEG_MODEL))

    @staticmethod
    def load(path) -> "PanopticResult":
        d = np.load(path, allow_pickle=False)
        return PanopticResult(d["ids"], d["segments"])


@dataclass
class Candidate:
    label: str
    label_id: int
    source: str                 # "panoptic" | "panoptic-split" | "semantic"
    score: float
    mask: np.ndarray            # (H, W) bool


@dataclass
class FurnitureInstance:
    id: int
    label: str                  # nome base ADE20K (ex.: "coffee table")
    label_id: int
    kind: str                   # "solto" | "planejado"
    source: str                 # "panoptic" | "panoptic-split" | "semantic"
    score: float
    mask: np.ndarray            # (H, W) bool, máscara dura
    alpha: np.ndarray           # (H, W) float32 0..1 (suave na borda)
    sam_iou: float | None = None
    sam_used: bool = False
    merged_from: list[int] = field(default_factory=list)
    kind_reason: str = ""
    touches: tuple[str, ...] = ()   # bordas da foto tocadas: "esquerda", "direita", "topo", "base"
    wall_panel: np.ndarray | None = None   # painel fixo recortado da peça: fica na foto (nem efeito/sombra sai)
    inside_builtin: str = ""        # peça solta pequena dentro de um planejado (livro na estante): fica com ele

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) com x1/y1 exclusivos."""
        return _bbox(self.mask)

    def summary(self, image_area: int) -> dict:
        return {"id": self.id, "rotulo": self.label, "tipo": self.kind, "origem": self.source,
                "score": round(self.score, 3), "area_pct": round(100.0 * self.area / image_area, 3),
                "bbox": list(self.bbox), "sam_iou": None if self.sam_iou is None else round(self.sam_iou, 3),
                "sam_usado": self.sam_used, "fundidos": list(self.merged_from), "motivo_tipo": self.kind_reason,
                "dentro_de_planejado": self.inside_builtin,
                "bordas": list(self.touches)}


@dataclass
class ModeDecision:
    mode: str                   # "A" (sala vazia) | "B" (foto mobiliada)
    reason: str                 # texto em português para o relatório / a interface
    features: dict
    uncertain: bool = False     # perto do limite: a interface pode pedir confirmação
    forced: bool = False


@dataclass
class InstanceSet:
    instances: list[FurnitureInstance]
    shape: tuple[int, int]
    mode: ModeDecision | None = None
    warnings: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def removal(self, keep_builtins: bool = False) -> list[FurnitureInstance]:
        """Peças que saem da placa. ``keep_builtins`` é o recuo do portão de qualidade (só os soltos saem; os
        soltos pequenos guardados dentro de um planejado, como livros na estante embutida, ficam com ele: tirá-los
        abre buracos no móvel que fica)."""
        return [i for i in self.instances
                if not (keep_builtins and (i.kind == KIND_BUILTIN or i.inside_builtin))]

    def removal_mask(self, keep_builtins: bool = False) -> np.ndarray:
        m = np.zeros(self.shape, bool)
        for i in self.removal(keep_builtins):
            m |= i.mask | (i.alpha > 0)
        return m

    def report(self) -> dict:
        area = int(self.shape[0] * self.shape[1])
        return {
            "modo": None if self.mode is None else {"modo": self.mode.mode, "motivo": self.mode.reason,
                                                   "incerto": self.mode.uncertain, "forcado": self.mode.forced,
                                                   "medidas": self.mode.features},
            "n_pecas": len(self.instances),
            "n_soltos": sum(i.kind == KIND_LOOSE for i in self.instances),
            "n_planejados": sum(i.kind == KIND_BUILTIN for i in self.instances),
            "area_remocao_pct": round(100.0 * float(self.removal_mask().mean()), 2),
            "area_remocao_so_soltos_pct": round(100.0 * float(self.removal_mask(True).mean()), 2),
            "pecas": [i.summary(area) for i in self.instances],
            "avisos": list(self.warnings),
            "tempos_s": {k: round(v, 2) for k, v in self.timings.items()},
        }


# ============================================================================================================
# Utilitários
# ============================================================================================================
def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def _touches(mask: np.ndarray, margin: int = 2) -> tuple[str, ...]:
    out = []
    if mask[:, :margin].any():
        out.append("esquerda")
    if mask[:, -margin:].any():
        out.append("direita")
    if mask[:margin].any():
        out.append("topo")
    if mask[-margin:].any():
        out.append("base")
    return tuple(out)


def _ring(mask: np.ndarray, px: int) -> np.ndarray:
    return ndimage.binary_dilation(mask, iterations=px) & ~mask


def _label_lut(id2label: dict[int, str]) -> dict[int, str]:
    return {int(k): base_name(v) for k, v in id2label.items()}


# ============================================================================================================
# 1. Candidatas: panoptic ∪ semântica
# ============================================================================================================
def _split_far_parts(mask: np.ndarray) -> list[np.ndarray]:
    """Partes grandes e distantes de uma mesma instância panóptica (2 tapetes fundidos) viram instâncias.
    Partes próximas (cadeira atrás da perna da mesa) continuam juntas."""
    lab, n = ndimage.label(mask)
    if n < 2:
        return [mask]
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    big = [k + 1 for k in range(n) if sizes[k] >= 0.25 * sizes.max() and sizes[k] >= SPLIT_MIN_PART * mask.size]
    if len(big) < 2:
        return [mask]
    gap = SPLIT_MIN_GAP * float(np.hypot(*mask.shape))
    # agrupa partes cuja distância (borda a borda) é menor que ``gap``
    dist = [ndimage.distance_transform_edt(lab != k) for k in big]
    group = list(range(len(big)))

    def find(i):
        while group[i] != i:
            i = group[i]
        return i

    for i in range(len(big)):
        for j in range(i + 1, len(big)):
            if dist[i][lab == big[j]].min() < gap:
                group[find(j)] = find(i)
    roots = sorted({find(i) for i in range(len(big))})
    if len(roots) < 2:
        return [mask]
    parts = []
    for r in roots:
        m = np.isin(lab, [big[i] for i in range(len(big)) if find(i) == r])
        parts.append(m)
    # pedaços pequenos ficam com a parte mais próxima
    rest = mask & ~np.logical_or.reduce(parts)
    if rest.any():
        d = np.stack([ndimage.distance_transform_edt(~p) for p in parts])
        owner = np.argmin(d, axis=0)
        for i in range(len(parts)):
            parts[i] = parts[i] | (rest & (owner == i))
    return parts


def is_ceiling_light(mask: np.ndarray, labels: np.ndarray, ceiling_ids: Sequence[int]) -> bool:
    """Pendente/plafon que o ADE20K chama de "lamp": encosta no teto ou pende do topo da foto sem chegar
    perto do chão. Luminárias de teto ficam na foto (§3.6)."""
    x0, y0, x1, y1 = _bbox(mask)
    H = mask.shape[0]
    if y0 <= 2 and y1 <= 0.45 * H:
        return True
    if ceiling_ids and y1 <= 0.6 * H:
        ring = _ring(mask, 4)
        if np.isin(labels[ring], list(ceiling_ids)).sum() >= 0.15 * max(1, x1 - x0):
            return True
    return False


def gather_candidates(seg, pan: PanopticResult) -> list[Candidate]:
    """Candidatas a peça removível: instâncias panópticas das classes removíveis (com separação de partes
    distantes) mais os componentes semânticos que o panoptic perdeu."""
    names = _label_lut(seg.id2label)
    H, W = pan.ids.shape
    cands: list[Candidate] = []
    covered = np.zeros((H, W), bool)
    ceiling_ids = [k for k, n in names.items() if n == "ceiling"]
    for row in pan.segments:
        k, lid, score = int(row[0]), int(row[1]), float(row[2])
        name = names.get(lid, "")
        if name not in REMOVABLE_CLASSES:
            continue
        m = pan.ids == k
        if m.sum() < MIN_AREA_PANOPTIC * m.size:
            continue
        covered |= m
        if name == "lamp" and is_ceiling_light(m, seg.labels, ceiling_ids):
            continue
        parts = _split_far_parts(m)
        src = "panoptic" if len(parts) == 1 else "panoptic-split"
        for p in parts:
            cands.append(Candidate(name, lid, src, score, p))
    labels = seg.labels
    for lid in np.unique(labels):
        name = names.get(int(lid), "")
        if name not in REMOVABLE_CLASSES:
            continue
        sem = ndimage.binary_opening(labels == lid, iterations=2)
        lab, n = ndimage.label(sem)
        for k in range(1, n + 1):
            comp = lab == k
            a = int(comp.sum())
            if a < MIN_AREA_SEMANTIC * comp.size or (comp & covered).sum() > 0.5 * a:
                continue
            rest = comp & ~covered
            if rest.sum() < MIN_AREA_SEMANTIC * comp.size:
                continue
            if name == "lamp" and is_ceiling_light(rest, labels, ceiling_ids):
                continue
            cands.append(Candidate(name, int(lid), "semantic", 0.0, rest))
    return cands


# ============================================================================================================
# 2. Refino com SAM 2.1 por caixa
# ============================================================================================================
SamFn = Callable[[np.ndarray, list[list[float]]], list[np.ndarray]]


def _cleanup(prob: np.ndarray, m0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maior componente conexo (mais pedaços confirmados pelo OneFormer) e alpha suave limpo."""
    hard = prob > 0.5
    lab, n = ndimage.label(hard)
    if n > 1:
        sizes = ndimage.sum(hard, lab, range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        keep = lab == largest
        for k in range(1, n + 1):
            if k == largest:
                continue
            comp = lab == k
            c = int(comp.sum())
            if c > 30 and (comp & m0).sum() > 0.6 * c:
                keep |= comp
        hard = keep
    if not hard.any():
        hard = m0.copy()
    region = ndimage.binary_dilation(hard, iterations=3)
    alpha = np.where(region, prob, 0.0).astype(np.float32)
    alpha[alpha < ALPHA_CUT] = 0.0
    alpha[ndimage.binary_erosion(hard, iterations=2)] = 1.0
    return hard, alpha


def refine(rgb: np.ndarray, cands: Sequence[Candidate], sam: SamFn | None,
           protected: np.ndarray | None = None) -> list[tuple[np.ndarray, np.ndarray, float | None, bool]]:
    """Para cada candidata: (máscara dura, alpha, IoU do SAM, SAM usado?).

    Quando o SAM é aceito, o que ele acrescenta é limitado a uma faixa em volta da máscara do OneFormer
    (``SAM_MAX_GROW``) e nunca entra em ``protected`` (cortina, janela, porta... da semântica): sem isso a
    cama "levava" a cortina e uma auréola de carpete, que a placa teria de inventar."""
    import cv2

    grow = max(4, int(round(SAM_MAX_GROW * float(np.hypot(*rgb.shape[:2])))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    probs: list[np.ndarray | None] = [None] * len(cands)
    if sam is not None and cands:
        boxes = []
        for c in cands:
            x0, y0, x1, y1 = _bbox(c.mask)
            boxes.append([float(x0), float(y0), float(x1), float(y1)])
        probs = list(sam(rgb, boxes))
    out = []
    for c, p in zip(cands, probs):
        m0 = c.mask
        best, biou = None, None
        if p is not None:
            p = np.asarray(p, np.float32)
            if p.ndim == 2:
                p = p[None]
            for k in range(p.shape[0]):
                iou = _iou(p[k] > 0.5, m0)
                if biou is None or iou > biou:
                    best, biou = p[k], iou
        used = best is not None and biou >= SAM_MIN_IOU
        if used:
            allowed = cv2.dilate(m0.astype(np.uint8), kernel) > 0
            if protected is not None:
                allowed &= ~protected | m0
            best = np.where(allowed, best, 0.0).astype(np.float32)
        else:
            best = ndimage.gaussian_filter(m0.astype(np.float32), 1.0)
            best[m0] = np.maximum(best[m0], 0.51)
        hard, alpha = _cleanup(best, m0)
        out.append((hard, alpha, biou, used))
    return out


def sam2_box_fn(device: str = "cpu", chunk: int = 16, offline: bool = False) -> SamFn:
    """SAM 2.1 base+ (Apache-2.0) via ``transformers.Sam2Model``; devolve, por caixa, as 3 probabilidades
    (multimask) na resolução da foto. Roda em CPU se não houver CUDA."""
    from ..analysis.weights import sam2_dir

    folder = sam2_dir(offline=offline)

    def run(rgb: np.ndarray, boxes: list[list[float]]) -> list[np.ndarray]:
        import torch
        from PIL import Image
        from transformers import Sam2Model, Sam2Processor

        proc = Sam2Processor.from_pretrained(str(folder))
        model = Sam2Model.from_pretrained(str(folder)).to(device).eval()
        img = Image.fromarray(rgb)
        res: list[np.ndarray] = []
        try:
            with torch.inference_mode():
                emb = None
                for s in range(0, len(boxes), chunk):
                    part = boxes[s:s + chunk]
                    inp = proc(images=img, input_boxes=[part], return_tensors="pt").to(device)
                    if emb is None:
                        emb = model.get_image_embeddings(inp["pixel_values"])
                    kw = {k: v for k, v in inp.items() if k not in ("pixel_values",)}
                    o = model(image_embeddings=emb, input_boxes=kw["input_boxes"], multimask_output=True)
                    logits = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                                     binarize=False)[0]
                    res.extend(torch.sigmoid(logits.float()).numpy())
        finally:
            del model
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        return res

    return run


# ============================================================================================================
# 3. Fusão de fragmentos (regra do fecho convexo)
# ============================================================================================================
def _hull_mask(mask: np.ndarray) -> np.ndarray:
    import cv2

    pts = cv2.findNonZero(mask.astype(np.uint8))
    hull = np.zeros(mask.shape, np.uint8)
    if pts is not None:
        cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    return hull > 0


def merge_fragments(insts: list[FurnitureInstance]) -> list[FurnitureInstance]:
    if len(insts) < 2:
        return insts
    allm = np.logical_or.reduce([i.mask for i in insts])
    gone: set[int] = set()
    for a in insts:
        for b in insts:
            if a.id >= b.id or a.label != b.label or "semantic" not in (a.source, b.source):
                continue
            if a.id in gone or b.id in gone:
                continue
            hull = _hull_mask(a.mask | b.mask)
            cov = float((allm & hull).sum()) / max(1, int(hull.sum()))
            ya, yb = a.bbox, b.bbox
            vov = (min(ya[3], yb[3]) - max(ya[1], yb[1])) / max(1, min(ya[3] - ya[1], yb[3] - yb[1]))
            if cov > MERGE_HULL_COVER and vov > MERGE_VERTICAL_OVERLAP:
                big, small = (a, b) if a.area >= b.area else (b, a)
                big.mask = big.mask | small.mask
                big.alpha = np.maximum(big.alpha, small.alpha)
                big.merged_from.append(small.id)
                gone.add(small.id)
    return [i for i in insts if i.id not in gone]


# ============================================================================================================
# 4. Solto x planejado
# ============================================================================================================
def _room_points(mask: np.ndarray, geometry: dict, room) -> np.ndarray | None:
    """Pontos 3D da peça no referencial da sala (z para cima, piso em 0)."""
    if geometry is None or room is None:
        return None
    pts = np.asarray(geometry["points"], np.float32)
    s = int(geometry.get("stride", 1))
    valid = np.asarray(geometry.get("mask", np.ones(pts.shape[:2], bool)), bool)
    sub = mask[::s, ::s][:pts.shape[0], :pts.shape[1]]
    sel = sub & valid[:sub.shape[0], :sub.shape[1]] & np.isfinite(pts[:sub.shape[0], :sub.shape[1]]).all(-1)
    p = pts[:sub.shape[0], :sub.shape[1]][sel]
    if len(p) < 20:
        return None
    M = np.asarray(room.cam_to_room, np.float64)
    return (p.astype(np.float64) @ M[:3, :3].T) + M[:3, 3]


def builtin_reason(inst: FurnitureInstance, seg, others: Sequence[FurnitureInstance], geometry: dict | None = None,
                   room=None) -> str:
    """Motivo para tratar a peça como planejada ("" = solta)."""
    if inst.label in BUILTIN_CLASSES:
        return "classe fixa (cozinha/banheiro)"
    if inst.label == "desk":            # mesa presa a um painel alto (divisória com TV, estante embutida)
        P = _room_points(inst.mask, geometry, room)
        if P is not None and (P[:, 2] > DESK_PANEL_Z).mean() >= DESK_PANEL_FRAC:
            return "mesa presa a painel alto"
        if P is not None:               # bancada de trabalho de parede a parede (nicho de escritório)
            for w in room.walls:
                near = np.abs(w.signed_distance(P[:, :2])) <= DESK_WALL_DEPTH
                if w.length >= DESK_WALL_MIN_LEN and near.mean() >= 0.5:
                    along = w.along(P[near, :2])
                    if np.percentile(along, 98) - np.percentile(along, 2) >= 0.8 * w.length:
                        return f"mesa de parede a parede ({w.id})"
        # segue as regras das classes ambíguas (bancada embutida no nicho, de parede a parede, junto da cozinha)
    elif inst.label not in AMBIGUOUS_CLASSES:
        return ""
    names = _label_lut(seg.id2label)
    ring = _ring(inst.mask, 4)
    x0, y0, x1, y1 = inst.bbox
    width = max(1, x1 - x0)
    # encosta no teto (imagem)
    ceil_ids = [k for k, n in names.items() if n == "ceiling"]
    if ceil_ids and np.isin(seg.labels[ring], ceil_ids).sum() >= 0.15 * width:
        return "encosta no teto"
    # junto de um conjunto de cozinha/banheiro (outra peça fixa ou bancada/pia na semântica)
    for o in others:
        if o is not inst and o.label in BUILTIN_CLASSES and (o.mask & ring).sum() >= 0.1 * width:
            return f"junto de {o.label}"
    fixed_ids = [k for k, n in names.items() if n in BUILTIN_CLASSES]
    if fixed_ids and np.isin(seg.labels[ring], fixed_ids).sum() >= 0.2 * width:
        return "junto de bancada/louça"
    P = _room_points(inst.mask, geometry, room)
    if P is not None:
        z = P[:, 2]
        ch = float(room.ceiling_height)
        if np.percentile(z, 98) >= 0.85 * ch:
            return "vai até o teto"
        xy = P[:, :2]
        for w in room.walls:
            d = np.abs(w.signed_distance(xy))
            near = d <= 0.3
            if near.mean() < 0.5:
                continue
            if np.percentile(z, 5) >= 1.0:
                return f"suspenso na parede {w.id}"
            along = w.along(xy[near])
            if w.length > 0.5 and (np.percentile(along, 98) - np.percentile(along, 2)) >= 0.8 * w.length:
                return f"ocupa a parede {w.id} inteira"
        if _on_wall(P, room).mean() >= PANEL_WHOLE_FRAC:
            return "painel na parede"
    if inst.label == "mirror":
        return "espelho (preso na parede)"
    return ""


def _on_wall(P: np.ndarray, room) -> np.ndarray:
    """Pontos (referencial da sala) no plano de alguma parede e acima de :data:`PANEL_MIN_Z`."""
    if not room.walls:
        return np.zeros(len(P), bool)
    d = np.min([np.abs(w.signed_distance(P[:, :2])) for w in room.walls], 0)
    return (d <= PANEL_WALL_DIST) & (P[:, 2] > PANEL_MIN_Z)


def wall_panel_mask(inst: FurnitureInstance, geometry: dict | None, room) -> np.ndarray | None:
    """Parte da máscara da peça que é painel fixo na parede (cabeceira-painel, painel de TV), ou None."""
    if geometry is None or room is None or not getattr(room, "walls", None):
        return None
    pts = np.asarray(geometry["points"], np.float32)
    s = int(geometry.get("stride", 1))
    valid = np.asarray(geometry.get("mask", np.ones(pts.shape[:2], bool)), bool)
    h, w = pts.shape[:2]
    sub = inst.mask[::s, ::s][:h, :w]
    sel = sub & valid[:sub.shape[0], :sub.shape[1]] & np.isfinite(pts[:sub.shape[0], :sub.shape[1]]).all(-1)
    if sel.sum() < 20:
        return None
    M = np.asarray(room.cam_to_room, np.float64)
    P = (pts[:sub.shape[0], :sub.shape[1]][sel].astype(np.float64) @ M[:3, :3].T) + M[:3, 3]
    on = np.zeros(sub.shape, bool)
    on[sel] = _on_wall(P, room)
    on = ndimage.binary_opening(on, iterations=1)
    H, W = inst.mask.shape
    up = np.repeat(np.repeat(on, s, 0), s, 1)[:H, :W]
    full = np.zeros((H, W), bool)
    full[:up.shape[0], :up.shape[1]] = up
    full = ndimage.binary_fill_holes(ndimage.binary_closing(full, iterations=max(1, s))) & inst.mask
    lab, n = ndimage.label(full)
    if n == 0:
        return None
    sizes = ndimage.sum(full, lab, range(1, n + 1))
    keep = [k + 1 for k, a in enumerate(sizes) if a >= 0.002 * H * W]
    return np.isin(lab, keep) if keep else None


def split_wall_panels(insts: list[FurnitureInstance], geometry: dict | None, room) -> list[str]:
    """Recorta das camas e mesas o painel fixo na parede que veio junto na máscara (cabeceira-painel de hotel,
    painel da TV): ele fica na foto, como acabamento de parede (§3.6), e a peça entra na frente dele. Sem isso a
    sala vazia precisa inventar a parede inteira atrás do painel (parede derretida)."""
    notes = []
    for i in insts:
        if i.label not in PANEL_PARENTS or i.kind != KIND_LOOSE:
            continue
        panel = wall_panel_mask(i, geometry, room)
        if panel is None:
            continue
        a, area = int(panel.sum()), i.area
        if a < PANEL_MIN_FRAC * area or a < PANEL_MIN_AREA * panel.size or area - a < PANEL_MIN_AREA * panel.size:
            continue
        rem = i.mask & ~panel           # filetes finos da peça na borda do painel (contorno do OneFormer) ficam com ele
        sliver = rem & ~ndimage.binary_opening(rem, iterations=3) & ndimage.binary_dilation(panel, iterations=6)
        panel = panel | sliver
        i.mask = rem & ~sliver
        i.wall_panel = panel
        i.alpha = np.where(panel, 0.0, i.alpha).astype(np.float32)
        notes.append(f"painel fixo na parede atrás de {i.label} ({100 * a / panel.size:.1f}% da foto) fica na foto")
    return notes


def classify(insts: list[FurnitureInstance], seg, geometry: dict | None = None, room=None) -> None:
    # 2 passadas: um armário colado num armário já planejado também é planejado (fileira de módulos)
    for _ in range(2):
        for i in insts:
            r = builtin_reason(i, seg, insts, geometry, room)
            if not r and (i.label in AMBIGUOUS_CLASSES or i.label == "desk"):   # mesa embutida entre módulos
                ring = _ring(i.mask, 4)
                for o in insts:
                    if o is not i and o.kind == KIND_BUILTIN and o.label in AMBIGUOUS_CLASSES \
                            and (o.mask & ring).sum() >= 0.1 * max(1, i.bbox[2] - i.bbox[0]):
                        r = f"junto de {o.label} planejado"
                        break
            i.kind, i.kind_reason = (KIND_BUILTIN, r) if r else (KIND_LOOSE, "")
    built = [o for o in insts if o.kind == KIND_BUILTIN]
    for i in insts:
        i.inside_builtin = ""
        if i.kind != KIND_LOOSE or i.label not in STORED_CLASSES or not built \
                or i.area > INSIDE_BUILTIN_MAX_AREA * i.mask.size:
            continue
        ring = _ring(i.mask, 6)
        n = max(1, int(ring.sum()))
        for o in built:
            if (o.mask & ring).sum() >= INSIDE_BUILTIN_RING * n:
                i.inside_builtin = o.label
                break


# ============================================================================================================
# 5. Modo A ou B
# ============================================================================================================
def mode_features(masks_labels: Sequence[tuple[np.ndarray, str]], shape: tuple[int, int]) -> dict:
    n = int(shape[0] * shape[1])
    union = np.zeros(shape, bool)
    loose = np.zeros(shape, bool)
    sig = 0
    major = 0.0
    major_label = ""
    for m, lab in masks_labels:
        union |= m
        if lab in LOOSE_CLASSES:
            loose |= m
        a = float(m.sum()) / n
        if a >= MODE_B_PIECE_AREA:
            sig += 1
        if lab in MAJOR_CLASSES and lab in LOOSE_CLASSES and a > major:
            major, major_label = a, lab
    return {"area_removivel": round(float(union.mean()), 4), "area_soltos": round(float(loose.mean()), 4),
            "pecas_significativas": sig,
            "maior_peca_grande": round(major, 4), "rotulo_maior_peca": major_label, "n_pecas": len(masks_labels)}


def decide_mode(features: dict, override: str | None = None) -> ModeDecision:
    if override in ("A", "B"):
        return ModeDecision(override, f"modo {override} escolhido manualmente", features, False, True)
    a, n, mj = features["area_removivel"], features["pecas_significativas"], features["maior_peca_grande"]
    loose = features.get("area_soltos", a)
    rules = []
    if a >= MODE_B_AREA and (loose >= MODE_B_LOOSE_AREA or a >= MODE_B_BUILTIN_ONLY):
        rules.append(f"móveis cobrem {100 * a:.1f}% da foto")
    if n >= MODE_B_PIECES:
        rules.append(f"{n} peças significativas")
    if mj >= MODE_B_MAJOR_AREA and a >= MODE_B_MAJOR_TOTAL:  # maior peça grande e solta
        rules.append(f"{features['rotulo_maior_peca']} com {100 * mj:.1f}% da foto")
    # perto do limite: área entre metade e o dobro do limite e no máximo uma regra satisfeita
    uncertain = (0.5 * MODE_B_AREA <= a < 2 * MODE_B_AREA) and len(rules) <= 1
    if rules:
        return ModeDecision("B", "foto mobiliada: " + "; ".join(rules), features, uncertain)
    return ModeDecision("A", f"sala vazia: móveis cobrem só {100 * a:.1f}% da foto", features, uncertain)


def detect_mode(seg, pan: PanopticResult, override: str | None = None) -> ModeDecision:
    """Detecção rápida (só OneFormer, sem SAM): usada para rotear a foto antes do resto do pipeline."""
    cands = gather_candidates(seg, pan)
    return decide_mode(mode_features([(c.mask, c.label) for c in cands], pan.ids.shape), override)


# ============================================================================================================
# OneFormer (panoptic + semântica numa carga só)
# ============================================================================================================
def load_oneformer(device: str = "cpu", model_id: str = SEG_MODEL, use_fp16: bool = False):
    """Carrega o OneFormer uma vez (para processar várias fotos com :func:`oneformer_infer`)."""
    from ..analysis.geometry import _ensure_weights
    from ..analysis.weights import ensure_local, hub_offline

    _ensure_weights(ensure_local, model_id)
    from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

    with hub_offline(model_id):
        proc = OneFormerProcessor.from_pretrained(model_id)
        model = OneFormerForUniversalSegmentation.from_pretrained(model_id, use_safetensors=False)
    model = model.to(device).eval()
    if use_fp16 and device != "cpu":
        model = model.half()
    return proc, model


def oneformer_infer(proc, model, rgb8: np.ndarray, device: str = "cpu", timings: dict | None = None):
    """Tarefas semântica e panóptica (``label_ids_to_fuse`` vazio). Retorna (SegmentationResult, PanopticResult)."""
    import time

    import torch
    from PIL import Image

    from ..analysis.segmentation import SegmentationResult, labels_to_groups

    half = next(model.parameters()).dtype == torch.float16
    H, W = rgb8.shape[:2]
    img = Image.fromarray(rgb8)
    outs = {}
    with torch.inference_mode():
        for task in ("semantic", "panoptic"):
            t = time.perf_counter()
            inp = proc(images=img, task_inputs=[task], return_tensors="pt")
            inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
            if half:
                inp["pixel_values"] = inp["pixel_values"].half()
            o = model(**inp)
            if task == "semantic":
                outs[task] = proc.post_process_semantic_segmentation(o, target_sizes=[(H, W)])[0]
            else:
                outs[task] = proc.post_process_panoptic_segmentation(
                    o, target_sizes=[(H, W)], threshold=0.5, label_ids_to_fuse=set())[0]
            if timings is not None:
                timings[f"oneformer_{task}"] = time.perf_counter() - t
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    labels = outs["semantic"].cpu().numpy().astype(np.uint8)
    seg = SegmentationResult(labels, labels_to_groups(labels, id2label), id2label, SEG_MODEL)
    segm = outs["panoptic"]["segmentation"].cpu().numpy().astype(np.int32)
    infos = list(outs["panoptic"]["segments_info"])
    # o mapa é uint8 (no máximo 254 segmentos): as classes removíveis vêm primeiro
    names = _label_lut(id2label)
    infos.sort(key=lambda s: (names.get(int(s["label_id"]), "") not in REMOVABLE_CLASSES, -float(s["score"])))
    ids = np.zeros((H, W), np.uint8)
    rows = []
    for k, s in enumerate(infos[:254], start=1):
        m = segm == int(s["id"])
        ids[m] = k
        rows.append((k, int(s["label_id"]), float(s["score"]), int(m.sum())))
    return seg, PanopticResult(ids, np.array(rows, np.float64).reshape(-1, 4))


def run_oneformer(rgb8: np.ndarray, device: str = "cpu", model_id: str = SEG_MODEL, use_fp16: bool = False,
                  timings: dict | None = None):
    """Carrega o OneFormer, roda as duas tarefas numa carga só e libera a memória."""
    import time

    t0 = time.perf_counter()
    proc, model = load_oneformer(device, model_id, use_fp16)
    if timings is not None:
        timings["oneformer_carga"] = time.perf_counter() - t0
    try:
        return oneformer_infer(proc, model, rgb8, device, timings)
    finally:
        del model
        gc.collect()
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


# ============================================================================================================
# Ponto de entrada
# ============================================================================================================
def detect_instances(rgb8: np.ndarray, seg=None, pan: PanopticResult | None = None, sam: SamFn | None | str = "auto",
                     device: str = "cpu", geometry: dict | None = None, room=None,
                     mode_override: str | None = None) -> InstanceSet:
    """Instâncias de móveis da foto + decisão de modo.

    ``seg``/``pan`` já calculados são reaproveitados (senão roda o OneFormer). ``sam="auto"`` usa o SAM 2.1
    se os pesos estiverem no disco (sem baixar) e cai para as máscaras do OneFormer se não estiverem;
    ``sam=None`` desliga o refino. ``geometry`` (pontos da câmera, opcionalmente com ``stride``) e ``room``
    ativam a parte 3D da regra solto x planejado.
    """
    import time

    timings: dict = {}
    warnings: list[str] = []
    if seg is None or pan is None:
        seg, pan = run_oneformer(rgb8, device=device, timings=timings)
    t = time.perf_counter()
    cands = gather_candidates(seg, pan)
    timings["candidatas"] = time.perf_counter() - t
    mode = decide_mode(mode_features([(c.mask, c.label) for c in cands], pan.ids.shape), mode_override)
    if sam == "auto":
        from ..analysis.weights import WeightsError

        try:
            sam = sam2_box_fn(device=device, offline=True)
        except WeightsError:
            sam = None
            warnings.append("SAM 2.1 não instalado: bordas das peças vêm só do OneFormer")
    t = time.perf_counter()
    names = _label_lut(seg.id2label)
    prot_ids = [k for k, n in names.items() if n in PROTECTED_CLASSES]
    protected = np.isin(seg.labels, prot_ids) if prot_ids else None
    refined = refine(rgb8, cands, sam, protected) if cands else []
    timings["sam"] = time.perf_counter() - t
    insts = []
    for k, (c, (hard, alpha, iou, used)) in enumerate(zip(cands, refined)):
        insts.append(FurnitureInstance(k, c.label, c.label_id, KIND_LOOSE, c.source, c.score, hard, alpha,
                                       iou, used))
    insts = merge_fragments(insts)
    classify(insts, seg, geometry, room)
    warnings.extend(split_wall_panels(insts, geometry, room))
    for i in insts:
        i.touches = _touches(i.mask)
    for new_id, i in enumerate(insts):
        i.id = new_id
    s = InstanceSet(insts, tuple(pan.ids.shape), mode, warnings, timings)
    if s.removal_mask().mean() > MAX_REMOVAL_AREA:
        warnings.append(f"os móveis cobrem {100 * s.removal_mask().mean():.0f}% da foto: quase toda a sala vazia "
                        "será reconstruída; confira o primeiro quadro")
    return s
