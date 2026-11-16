"""Assets 3D do 7xAi.

- ``procedural.py`` — gera no Blender (``bpy``) os itens que a biblioteca CC0 da
  Poly Haven não tem: tapete, luminária de piso, abajur, cama, cortina, escrivaninha,
  manta e pilha de livros deitados. Cada asset é um EMPTY raiz (origem no centro da base, frente para -Y,
  metros) com as peças como filhos, prontas para a coreografia de montagem.
- ``materials.py`` — materiais PBR procedurais (sem texturas externas).

Este ``__init__`` não importa ``bpy``: pode ser usado por qualquer processo (layout,
interface) para consultar os tipos e movimentos válidos e, via ``support_surface``, a
superfície de apoio dos procedurais (topo do edredom da cama, tampo da escrivaninha) para
itens "on_top".
"""
from __future__ import annotations

# Tipos procedurais (asset_id "proc:<tipo>" no catálogo)
PROCEDURAL_KINDS: tuple[str, ...] = (
    "rug", "floor_lamp", "table_lamp", "bed", "curtain", "desk", "throw_blanket", "book_stack",
)

# Movimentos de montagem que uma peça pode declarar em obj["assembly_motion"]
ASSEMBLY_MOTIONS: tuple[str, ...] = ("drop", "rise", "slide", "unroll", "unfold", "grow", "settle")

# Propriedades customizadas presentes em TODA peça (filho do EMPTY raiz)
PART_PROPERTIES: tuple[str, ...] = ("mobilia_part", "assembly_order", "assembly_motion")

# --------------------------------------------------------------------------- cama (medidas)
# Fonte única das medidas da cama procedural: usada pelo construtor (procedural.py) e pelo
# layout (superfície de apoio para itens "on_top", ex.: manta sobre o edredom).
MATTRESS_HEIGHT = 0.24     # altura do colchão (m)
MATTRESS_EDGE_RADIUS = 0.05  # raio das quinas do colchão (m)
DUVET_THICKNESS = 0.038    # espessura do edredom (m)
DUVET_GAP = 0.004          # folga entre colchão e edredom (m)
DUVET_TURN = 0.56          # raio da dobra do edredom (fração da espessura)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bed_dimensions(size, params: dict | None = None) -> dict:
    """Medidas internas da cama procedural para o tamanho (l, p, a) e os parâmetros dados.

    Chaves: ``wood_frame``, ``hb_t`` (espessura da cabeceira), ``leg_h``, ``deck_z`` (topo
    do estrado), ``mattress_top``, ``mx`` (meia largura do colchão), ``my0``/``my1``
    (pé/cabeceira do colchão), ``y_fold`` (dobra do edredom), ``cuff`` (comprimento da aba
    dobrada, de ``y_fold`` para o pé) e ``duvet_top`` (topo do edredom, onde uma manta repousa).
    """
    W, D, H = (float(v) for v in size)
    p = params or {}
    wood_frame = str(p.get("frame", "wood" if "frame_color" in p else "upholstered")) == "wood"
    hb_dflt = 0.045 if wood_frame else _clip(0.055 * W, 0.07, 0.1)
    hb_t = _clip(float(p.get("headboard_depth", hb_dflt)), 0.03, 0.15)
    leg_h = 0.12 if wood_frame else 0.08
    deck_z = _clip(float(p.get("mattress_top", 0.54)) - MATTRESS_HEIGHT, leg_h + 0.1, 0.6 * H)
    side_out = DUVET_GAP + DUVET_THICKNESS + 2 * DUVET_TURN * DUVET_THICKNESS + 0.008
    my0, my1 = -D / 2 + 0.06, D / 2 - hb_t - 0.006
    pillow_zone = _clip(0.28 * (my1 - my0), 0.42, 0.62)
    mattress_top = deck_z + MATTRESS_HEIGHT
    y_fold = my1 - pillow_zone
    cuff = min(0.3, 0.5 * (y_fold - (my0 + MATTRESS_EDGE_RADIUS)))
    return dict(wood_frame=wood_frame, hb_t=hb_t, leg_h=leg_h, deck_z=deck_z, mattress_top=mattress_top,
                mx=W / 2 - side_out, my0=my0, my1=my1, y_fold=y_fold, cuff=cuff,
                duvet_top=mattress_top + DUVET_GAP + DUVET_THICKNESS)


def support_surface(kind: str, size, params: dict | None = None) -> dict | None:
    """Superfície de apoio de um asset procedural (para itens "on_top"), no referencial local
    (origem no centro da base, frente para -Y): {"kind", "z", "rects": [(x0, y0, x1, y1)]}.

    - bed: topo do edredom entre o pé e a dobra (a manta repousa ali);
    - desk: tampo inteiro (margem de 3 cm).
    Outros tipos não servem de apoio (None).
    """
    W, D, H = (float(v) for v in size)
    if kind == "desk":
        m = 0.03
        return {"kind": "top", "z": H, "rects": [(-W / 2 + m, -D / 2 + m, W / 2 - m, D / 2 - m)]}
    if kind == "bed":
        b = bed_dimensions(size, params)
        x = b["mx"] - 0.08
        y1 = b["y_fold"] - b["cuff"] - 0.05   # antes da aba dobrada (camada dupla, mais alta)
        return {"kind": "top", "z": b["duvet_top"], "rects": [(-x, b["my0"] + 0.08, x, y1)]}
    return None
