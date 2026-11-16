"""Testes do motor de layout automático (mobilia/layout).

Verifica, para a sala real de teste (stafford) e para salas retangulares sintéticas
construídas aqui (tamanhos e giros de câmera diferentes), em todos os tipos de ambiente:
- todas as peças inteiras dentro do quadro da foto (projeção 3D independente);
- nenhuma sobreposição entre peças de piso (tapete pode ficar por baixo);
- bases só sobre piso livre (nunca BLOCKED/OUTSIDE; no máximo 25% em UNKNOWN);
- decoração "on_top" apoiada na altura correta e dentro do móvel de apoio;
- quadros rentes à parede, sem invadir janelas/portas;
- distância mínima da câmera, tamanhos reais (escala 1) dos assets Poly Haven;
- ordem da coreografia coerente e determinismo para a mesma semente.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mobilia.analysis.geometry import Intrinsics
from mobilia.analysis.room import BLOCKED, CELL, FREE, OUTSIDE, UNKNOWN, Opening, RoomModel, Wall
from mobilia.layout import catalog
from mobilia.layout.context import LayoutContext
from mobilia.layout.geom import box_corners_3d, sample_obb, world_to_local
from mobilia.layout.model import Layout
from mobilia.layout.rules import validate_layout
from mobilia.layout.solver import plan_layout
from mobilia.layout.style import analyze_style
from mobilia.layout.surfaces import support_for

FIX = Path(__file__).parent / "fixtures"
ROOM_TYPES = ["sala", "quarto", "escritorio", "sala_jantar", "cozinha", "banheiro"]


# ----------------------------------------------------------------------------- salas sintéticas (dublê de teste)

def make_room(x0=-2.0, x1=2.5, y0=-0.4, y1=5.0, cam_h=1.5, yaw=0.0, pitch=10.0, hfov=70.0, size=(800, 600),
              ceiling=2.6, windows=(), doors=(), blocked=(), floor_rgb=(150, 110, 75), wall_rgb=(225, 222, 215)):
    """Sala retangular x0..x1 por y0..y1 com a câmera em (0, 0, cam_h).

    ``yaw`` > 0 gira a câmera para a esquerda; ``pitch`` inclina para baixo (graus).
    windows/doors: (parede, s0, s1, z0, z1) com parede em fundo|esquerda|direita|tras.
    blocked: retângulos (xa, ya, xb, yb) marcados como BLOCKED (bancadas, ilhas).
    Retorna (RoomModel, grade, foto RGB sintética).
    """
    W, H = size
    fx = (W / 2) / math.tan(math.radians(hfov) / 2)
    intr = Intrinsics(W, H, fx, fx, W / 2, H / 2)
    ps, th = math.radians(yaw), math.radians(pitch)
    f = np.array([-math.sin(ps) * math.cos(th), math.cos(ps) * math.cos(th), -math.sin(th)])
    r = np.array([math.cos(ps), math.sin(ps), 0.0])
    M = np.eye(4)
    M[:3, :3] = np.column_stack([r, np.cross(f, r), f])
    M[:3, 3] = [0, 0, cam_h]
    defs = {"fundo": ((x0, y1), (x1, y1), (0.0, -1.0)), "esquerda": ((x0, y0), (x0, y1), (1.0, 0.0)),
            "direita": ((x1, y1), (x1, y0), (-1.0, 0.0)), "tras": ((x1, y0), (x0, y0), (0.0, 1.0))}
    names = list(defs)
    walls = [Wall(i, list(defs[n][0]), list(defs[n][1]), list(defs[n][2]), ceiling, 1000, 0.01)
             for i, n in enumerate(names)]
    room = RoomModel(intrinsics=intr.to_dict(), cam_to_room=M.tolist(), camera_height=cam_h, ceiling_height=ceiling,
                     ceiling_observed=True, walls=walls, openings=[], grid_origin=[x0 - 0.5, y0 - 0.5],
                     grid_cell=CELL, grid_shape=[0, 0])
    for kind, lst in (("window", windows), ("door", doors)):
        for wname, s0, s1, z0, z1 in lst:
            w = walls[names.index(wname)]
            p0 = np.asarray(w.p0)
            dd = (np.asarray(w.p1) - p0) / w.length
            pts = np.array([[*(p0 + dd * s), z] for s in (s0, s1) for z in (z0, z1)])
            uv, _ = room.project_room_points(pts)
            uv = np.clip(uv, 0, [W - 1, H - 1])
            bbox = [int(uv[:, 0].min()), int(uv[:, 1].min()), int(uv[:, 0].max()), int(uv[:, 1].max())]
            room.openings.append(Opening(kind, w.id, s0, s1, z0, z1, bbox, [0.8, 0.8, 0.8], 0.8))
    nx = int(math.ceil((x1 - x0 + 1.0) / CELL))
    ny = int(math.ceil((y1 - y0 + 1.0) / CELL))
    grid = np.full((ny, nx), UNKNOWN, np.uint8)
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx = x0 - 0.5 + (xx + 0.5) * CELL
    cy = y0 - 0.5 + (yy + 0.5) * CELL
    inside = (cx > x0) & (cx < x1) & (cy > y0) & (cy < y1)
    uv, z = room.project_room_points(np.column_stack([cx.ravel(), cy.ravel(), np.zeros(cx.size)]))
    vis = ((z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)).reshape(ny, nx)
    grid[inside & vis] = FREE
    grid[~inside] = OUTSIDE
    for xa, ya, xb, yb in blocked:
        grid[(cx >= xa) & (cx <= xb) & (cy >= ya) & (cy <= yb)] = BLOCKED
    room.grid_shape = [ny, nx]
    # foto: piso em tábuas, paredes lisas
    rays = intr.rays().reshape(-1, 3) @ M[:3, :3].T
    down = rays[:, 2] < -1e-6
    t = np.where(down, -cam_h / np.where(down, rays[:, 2], -1.0), 0.0)
    hit = rays[:, :2] * t[:, None]
    img = np.empty((H * W, 3), np.float32)
    img[:] = wall_rgb
    plank = np.floor(hit[:, 0] / 0.12)
    fac = 1 + (((np.sin(plank * 12.9898) * 43758.5453) % 1.0) - 0.5) * 0.2 + np.sin(hit[:, 1] * 60 + plank) * 0.04
    floor = down & (hit[:, 0] > x0) & (hit[:, 0] < x1) & (hit[:, 1] > y0) & (hit[:, 1] < y1)
    img[floor] = np.array(floor_rgb, np.float32)[None] * fac[floor, None]
    photo = np.clip(img, 0, 255).astype(np.uint8).reshape(H, W, 3)
    return room, grid, photo


SYNTH = {
    "ampla_frontal": dict(x0=-2.6, x1=2.6, y0=-0.3, y1=5.6, cam_h=1.5, yaw=0.0, pitch=12.0, hfov=72.0,
                          windows=[("fundo", 3.4, 4.6, 0.9, 2.2)], doors=[("direita", 0.6, 1.5, 0.0, 2.1)]),
    "estreita_canto": dict(x0=-3.2, x1=1.0, y0=-0.3, y1=4.6, cam_h=1.4, yaw=30.0, pitch=10.0, hfov=68.0,
                           windows=[("esquerda", 1.0, 2.2, 0.9, 2.2)], floor_rgb=(205, 190, 165)),
    "pequena_direita": dict(x0=-1.2, x1=3.0, y0=-0.3, y1=4.0, cam_h=1.55, yaw=-25.0, pitch=8.0, hfov=75.0,
                            doors=[("esquerda", 2.4, 3.3, 0.0, 2.1)], floor_rgb=(70, 50, 38),
                            wall_rgb=(200, 205, 210)),
    "cozinha_ilha": dict(x0=-2.2, x1=2.8, y0=-0.3, y1=5.5, cam_h=1.5, yaw=5.0, pitch=12.0, hfov=72.0,
                         blocked=[(-2.2, 4.85, 2.8, 5.5), (-0.9, 2.7, 1.1, 3.6)], floor_rgb=(190, 190, 188),
                         wall_rgb=(240, 240, 238)),
}


# ----------------------------------------------------------------------------- dados (memorizados por módulo)

_ROOMS: dict[str, tuple] = {}
_LAYOUTS: dict[tuple, Layout] = {}


def get_room(name: str):
    if name not in _ROOMS:
        if name == "stafford":
            room = RoomModel.from_json((FIX / "rooms/stafford/room.json").read_text(encoding="utf-8"))
            grid = np.load(FIX / "rooms/stafford/grid.npy")
            photo = np.asarray(Image.open(FIX / "photos/sala_vazia_stafford.jpg").convert("RGB"))
        else:
            room, grid, photo = make_room(**SYNTH[name])
        _ROOMS[name] = (room, grid, photo)
    return _ROOMS[name]


def get_layout(name: str, room_type: str, seed: int = 0) -> Layout:
    k = (name, room_type, seed)
    if k not in _LAYOUTS:
        room, grid, photo = get_room(name)
        _LAYOUTS[k] = plan_layout(room, grid, room_type, photo, seed=seed)
    return _LAYOUTS[k]


CASES = [("stafford", t) for t in ROOM_TYPES] + [(n, t) for n in SYNTH for t in ROOM_TYPES]


# ----------------------------------------------------------------------------- verificações independentes

def _states(room: RoomModel, grid: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ij = np.floor((pts - np.asarray(room.grid_origin)) / room.grid_cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < grid.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < grid.shape[0])
    st = np.full(len(pts), OUTSIDE, np.uint8)
    st[ok] = grid[ij[ok, 1], ij[ok, 0]]
    return st


def _poly_overlap_area(a: np.ndarray, b: np.ndarray) -> float:
    from shapely.geometry import Polygon

    return float(Polygon(a).intersection(Polygon(b)).area)


@pytest.mark.parametrize("name,room_type", CASES)
def test_hard_constraints(name, room_type):
    room, grid, photo = get_room(name)
    lay = get_layout(name, room_type)
    assert lay.items, f"layout vazio para {name}/{room_type}"
    ctx = LayoutContext(room, grid)
    assert validate_layout(lay, ctx) == []
    intr = room.intr()
    by_id = {it.id: it for it in lay.items}
    assert len(by_id) == len(lay.items), "ids repetidos"
    for it in lay.items:
        # 1) inteiro dentro do quadro (caixa 3D projetada) e à frente da câmera
        box = box_corners_3d(*it.position, *it.size, it.yaw_deg)
        uv, z = room.project_room_points(box)
        assert np.all(z > 0.2), f"{it.id} atrás da câmera"
        assert np.all(uv[:, 0] >= 1) and np.all(uv[:, 0] <= intr.width - 1), f"{it.id} cortado na lateral"
        assert np.all(uv[:, 1] >= 1) and np.all(uv[:, 1] <= intr.height - 1), f"{it.id} cortado em cima/baixo"
        # 2) distância mínima da câmera
        fp = np.asarray(it.footprint())
        assert np.min(np.hypot(fp[:, 0], fp[:, 1])) >= 1.15 or it.mount == "on_top", f"{it.id} perto da câmera"
        # 3) tamanhos reais dos assets Poly Haven (escala 1)
        assert it.scale == 1.0
        if it.asset_id.startswith("ph:"):
            assert np.allclose(it.size, catalog.get(it.asset_id).size, atol=1e-3), f"{it.id} distorcido"
        # 4) base sobre piso livre
        if it.mount == "floor":
            assert abs(it.position[2]) < 1e-6
            pts = sample_obb(it.position[0], it.position[1], it.size[0], it.size[1], it.yaw_deg, step=0.025,
                             shrink=0.012)
            st = _states(room, grid, pts)
            assert not np.any((st == BLOCKED) | (st == OUTSIDE)), f"{it.id} sobre área proibida"
            assert np.mean(st == UNKNOWN) <= 0.25 + 1e-6, f"{it.id} sobre piso não observado"
    # 5) sem sobreposição entre peças de piso (tapete pode ficar por baixo)
    floor = [it for it in lay.items if it.mount == "floor" and it.role != "rug"]
    for i in range(len(floor)):
        for j in range(i + 1, len(floor)):
            a, b = np.asarray(floor[i].footprint()), np.asarray(floor[j].footprint())
            assert _poly_overlap_area(a, b) < 2e-3, f"{floor[i].id} sobrepõe {floor[j].id}"
    # 6) decoração apoiada no móvel, na altura da superfície
    for it in lay.items:
        if it.mount != "on_top":
            continue
        par = by_id[it.parent]
        sup = support_for(par.asset_id, par.role, par.size, par.params)
        assert sup is not None
        assert abs(it.position[2] - (par.position[2] + sup.z)) < 5e-3, f"{it.id} flutuando/afundado"
        lx, ly = world_to_local(par.position[0], par.position[1], par.yaw_deg, np.array([it.position[:2]]))[0]
        assert abs(lx) <= par.size[0] / 2 and abs(ly) <= par.size[1] / 2, f"{it.id} fora do apoio"
        assert par.order < it.order, f"{it.id} surge antes do apoio"
    # 7) quadros rentes à parede, sem tocar janelas/portas
    for it in lay.items:
        if it.mount != "wall":
            continue
        w = room.walls[it.wall_id]
        p0, n = np.asarray(w.p0), np.asarray(w.normal)
        d = (np.asarray(w.p1) - p0) / w.length
        c = np.asarray(it.position[:2])
        assert 0 <= (c - p0) @ n - it.size[1] / 2 <= 0.03
        s = (c - p0) @ d
        for o in room.openings:
            if o.wall != it.wall_id:
                continue
            sep = s + it.size[0] / 2 <= o.s0 or s - it.size[0] / 2 >= o.s1
            sep |= it.position[2] >= o.z1 or it.position[2] + it.size[2] <= o.z0
            assert sep, f"{it.id} sobre abertura"


@pytest.mark.parametrize("name,room_type", CASES)
def test_choreography_order(name, room_type):
    lay = get_layout(name, room_type)
    orders = sorted(it.order for it in lay.items)
    assert orders == list(range(len(lay.items)))
    by_order = sorted(lay.items, key=lambda it: it.order)
    rugs = [it for it in by_order if it.role == "rug"]
    if rugs:
        assert by_order[0].role == "rug", "o tapete entra primeiro"
    # decoração pequena depois de todos os móveis de piso
    last_floor = max((it.order for it in lay.items if it.mount == "floor"), default=-1)
    for it in lay.items:
        if it.mount == "on_top" and it.role != "table_lamp":
            assert it.order > last_floor
    # quadros depois dos móveis de piso grandes
    for it in lay.items:
        if it.mount == "wall":
            assert all(it.order > o.order for o in lay.items if o.role in ("sofa", "bed", "dining_table", "desk"))


@pytest.mark.parametrize("name,room_type", [("stafford", "sala"), ("estreita_canto", "quarto"),
                                            ("ampla_frontal", "sala_jantar")])
def test_determinism(name, room_type):
    room, grid, photo = get_room(name)
    a = plan_layout(room, grid, room_type, photo, seed=3)
    b = plan_layout(room, grid, room_type, photo, seed=3)
    assert a.to_json() == b.to_json()
    # JSON ida e volta preserva o contrato
    assert Layout.from_json(a.to_json()).to_json() == a.to_json()


def test_stafford_sala_is_polished():
    """A sala de teste precisa do grupo de conversa completo e bem composto."""
    room, grid, photo = get_room("stafford")
    lay = get_layout("stafford", "sala")
    roles = [it.role for it in lay.items]
    for r in ("sofa", "coffee_table", "rug", "wall_art", "floor_lamp", "side_table"):
        assert r in roles, f"falta {r}"
    assert any(r.startswith("plant") for r in roles)
    sofa = next(it for it in lay.items if it.role == "sofa")
    ct = next(it for it in lay.items if it.role == "coffee_table")
    rug = next(it for it in lay.items if it.role == "rug")
    f = np.array(sofa.front)
    # mesa de centro ~45 cm à frente do sofá e centrada
    ct_poly = np.asarray(ct.footprint())
    gap = float(np.min((ct_poly - np.array(sofa.position[:2])) @ f)) - sofa.size[1] / 2
    assert 0.35 <= gap <= 0.55
    u = np.array([math.cos(math.radians(sofa.yaw_deg)), math.sin(math.radians(sofa.yaw_deg))])
    assert abs((np.array(ct.position[:2]) - np.array(sofa.position[:2])) @ u) < 0.2
    # sofá mostra a frente (ou o perfil) para a câmera
    to_cam = -np.array(sofa.position[:2]) / np.linalg.norm(sofa.position[:2])
    assert f @ to_cam > 0.2
    # pés dianteiros do sofá sobre o tapete e mesa de centro inteira sobre ele
    from shapely.geometry import Point, Polygon

    rp = Polygon(rug.footprint())
    for lx in (-0.6, 0.0, 0.6):
        p = np.array(sofa.position[:2]) + u * lx * sofa.size[0] / 2 + f * (sofa.size[1] / 2 - 0.08)
        assert rp.contains(Point(*p)), "pés do sofá fora do tapete"
    assert rp.contains(Polygon(ct.footprint()))
    # almofadas no sofá
    assert any(it.role == "pillows" and it.parent == sofa.id for it in lay.items)


def test_style_analysis_stafford():
    room, grid, photo = get_room("stafford")
    ctx = LayoutContext(room, grid)
    prof = analyze_style(photo, None, ctx)
    assert prof.floor_material == "carpete"
    assert prof.floor_tone == "claro"
    assert prof.style == "moderno"
    # a textura é medida em escala física fixa: a mesma foto reduzida dá o mesmo resultado
    import cv2

    half = cv2.resize(photo, (photo.shape[1] // 2, photo.shape[0] // 2), interpolation=cv2.INTER_AREA)
    from mobilia.analysis.segmentation import SegmentationResult

    seg = SegmentationResult.load(FIX / "rooms/stafford/seg.npz")
    for ph, sg in ((half, None), (half, seg), (photo, seg)):
        p2 = analyze_style(ph, sg, ctx)
        assert (p2.style, p2.floor_material, p2.floor_tone) == ("moderno", "carpete", "claro")
    # sala clara (parede creme): estofado claro, nunca o sofá quase preto (PT9b)
    from mobilia.layout.style import is_bright_room, rank_assets

    assert is_bright_room(prof)
    assert rank_assets("sofa", prof)[0].tone == "claro"
    for key in ("accent", "rug", "textiles", "wood", "metal"):
        assert key in prof.palette


def test_procedural_params_and_rug_sizes():
    for name in ("stafford", "ampla_frontal"):
        lay = get_layout(name, "sala")
        rug = next(it for it in lay.items if it.role == "rug")
        assert rug.asset_id == "proc:rug"
        assert tuple(sorted(rug.size[:2], reverse=True)) in {(2.0, 1.4), (2.4, 1.7), (3.0, 2.0)}
        for k in ("color", "pattern_color", "pattern"):
            assert k in rug.params
        assert all(0.0 <= c <= 1.0 for c in rug.params["color"])


@pytest.mark.parametrize("room_type", ["sala", "quarto", "banheiro"])
def test_degenerate_room_without_walls(room_type):
    """Sem paredes detectadas (análise falhou parcialmente): o layout ainda sai válido."""
    room, grid, photo = make_room(**SYNTH["ampla_frontal"])
    room.walls, room.openings = [], []
    lay = plan_layout(room, grid, room_type, photo, seed=1)
    assert lay.items
    assert validate_layout(lay, LayoutContext(room, grid)) == []
    assert all(it.mount != "wall" for it in lay.items)


def test_unknown_room_type_is_rejected():
    room, grid, photo = get_room("pequena_direita")
    with pytest.raises(ValueError):
        plan_layout(room, grid, "garagem", photo)


# ----------------------------------------------------------------------------- PT9: ativos sem cara de CG

def test_bright_room_prefers_light_upholstery():
    """Sala clara: sofá/poltrona claros; os estofados de albedo quase preto ficam fora; no máximo
    uma massa escura em qualquer layout. Sala de paredes escuras mantém o contraste antigo."""
    from mobilia.layout.rules import dark_masses
    from mobilia.layout.style import DARK_UPH_TAG, is_bright_room, rank_assets
    from mobilia.layout.state import Placement

    room, grid, photo = get_room("stafford")
    prof = analyze_style(photo, None, LayoutContext(room, grid))
    assert is_bright_room(prof)
    for role in ("sofa", "armchair", "ottoman"):
        assert all(DARK_UPH_TAG not in a.tags for a in rank_assets(role, prof))
    lay = get_layout("stafford", "sala")
    for it in lay.items:
        if it.role in ("sofa", "armchair", "ottoman"):
            assert DARK_UPH_TAG not in catalog.get(it.asset_id).tags, it.asset_id
    sofa = next(it for it in lay.items if it.role == "sofa")
    assert catalog.get(sofa.asset_id).tone in ("claro", "neutro")
    for name, rt in [("stafford", "sala"), ("ampla_frontal", "sala"), ("pequena_direita", "quarto")]:
        items = [Placement.from_item(it) for it in get_layout(name, rt).items]
        assert dark_masses(items) <= 1
    # paredes escuras: não é sala clara, o sofá escuro volta a ser opção
    room2, grid2, photo2 = make_room(**dict(SYNTH["ampla_frontal"], wall_rgb=(70, 66, 62)))
    prof2 = analyze_style(photo2, None, LayoutContext(room2, grid2))
    assert not is_bright_room(prof2)
    assert any(DARK_UPH_TAG in a.tags for a in rank_assets("sofa", prof2))


def test_no_placeholder_art_in_catalog():
    """Nenhum quadro com cartão de modelo/arte esticada; a sala recebe a pintura gerada no tamanho
    das regras de projeto (~2/3 do móvel abaixo, ou tamanhos padrão em parede livre)."""
    ids = {a.asset_id for a in catalog.CATALOG}
    assert not set(catalog.PLACEHOLDER_ART) & ids
    for a in catalog.by_role("wall_art"):
        assert a.asset_id == "proc:canvas_art" or a.asset_id.startswith("ph:fancy_picture_frame_"), a.asset_id
    assert catalog.get("proc:canvas_art").scalable
    for name in ("stafford", "ampla_frontal"):
        lay = get_layout(name, "sala")
        arts = [it for it in lay.items if it.role == "wall_art"]
        assert arts, f"{name}: sem quadro"
        for it in arts:
            assert it.asset_id == "proc:canvas_art"
            assert it.params.get("frame") in ("tela", "moldura")
            assert it.params.get("colors") and it.params.get("ground") and "seed" in it.params
            w, d, h = it.size
            assert 0.3 <= w <= 1.8 and 0.3 <= h <= 1.0 and abs(d - 0.035) < 1e-6
            assert (it.params["frame"] == "tela") == (max(w, h) >= 0.9)
        if len(arts) == 2:      # díptico: mesmo tamanho, mesmo estilo, pinturas diferentes
            a, b = arts
            assert a.size == b.size and a.params["art_style"] == b.params["art_style"]
            assert a.params["seed"] != b.params["seed"]


def test_art_sizes_follow_design_rules():
    from mobilia.layout.solver import ART_PAIR_GAP, Solver
    from mobilia.layout.state import Placement

    room, grid, photo = get_room("stafford")
    ctx = LayoutContext(room, grid)
    S = Solver(ctx, analyze_style(photo, None, ctx), "sala")
    sofa = Placement("sofa_0", "sofa", "ph:Sofa_01", 1.8, 0.8, 0.8, 0.0, 3.0, 0.0, 0.0, "floor", None, None, {}, "")
    single, pair = S._art_sizes(sofa)
    assert abs(single[0][0] - 1.2) < 0.011                       # 2/3 de 1,8 m
    assert abs(single[0][0] / single[0][2] - 1.5) < 0.03         # paisagem 3:2 a partir de 1,2 m
    assert abs(2 * pair[0][0] + ART_PAIR_GAP - 1.2) < 0.011
    assert abs(pair[0][2] / pair[0][0] - 4 / 3) < 0.03           # retratos 3:4
    small = Placement("t", "tv_console", "x", 0.6, 0.4, 0.6, 0.0, 3.0, 0.0, 0.0, "floor", None, None, {}, "")
    assert abs(S._art_sizes(small)[0][0][0] - 0.5) < 1e-6        # nunca menor que 50 cm
    assert S._art_sizes(None)                                    # parede livre: tamanhos padrão
