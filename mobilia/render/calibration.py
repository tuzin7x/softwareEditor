"""Ajuste da luz do CG à foto (PT8), em NumPy puro: o worker do Blender renderiza a sala vazia
uma vez por fonte de luz e este módulo decide as intensidades e a cor das luzes.

Modelo (sala-proxy com emissão da foto, ver ``blender_worker.proxy_material``):

    predição = E + Σ kᵢ · Rᵢ

- ``E`` = render só com a emissão da foto nos raios não-câmera (termo fixo, força 1: a sala
  "devolve" exatamente a luz que a câmera viu);
- ``Rᵢ`` = render com uma fonte de cada vez, força 1: janelas, ambiente (paredes não
  fotografadas) e sol;
- ``kᵢ ≥ 0`` ajustados por mínimos quadrados na luminância dos pixels de piso/parede/teto
  (máscara de calibração).

Diferenças em relação ao ajuste antigo (um NNLS só, só pixels não saturados):
- pixels quase saturados da foto entram como **limite inferior** (a predição tem de ser pelo
  menos o valor recortado): ``nnls_censored``;
- o **sol é ajustado à parte**, só nas manchas de sol da foto (``analysis/sunlight.py``) que o sol
  do render também ilumina. No ajuste único o sol zerava: as manchas são poucas e quase sempre
  estouradas (eram descartadas);
- o limite da parcela do ambiente sobe de 0,5 para até 0,7 em salas claras;
- a cor das luzes recebe um ganho global por canal (antes era só diagnóstico).

Para o sol do render bater com o da foto, a geometria do sol também mora aqui: o telhado só de
sombra (``roof_polygons``: o sol não entra por cima das paredes) e as aberturas de sol das janelas
tiradas das manchas da foto (``sun_apertures``/``blocker_rects``: persianas e caixilhos que a
sala-proxy não tem).
"""
from __future__ import annotations

import numpy as np

LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)

# parcela máxima da luz vinda da "sala não fotografada" (paredes de fechamento): 0,5 em salas
# escuras/médias até 0,7 em salas claras (mais luz rebatida pelas paredes atrás da câmera)
AMBIENT_SHARE_MIN = 0.5
AMBIENT_SHARE_MAX = 0.7
BRIGHT_LUM = (0.20, 0.45)          # luminância linear mediana da sala: 0,5 → 0,7 nesse intervalo
COLOR_GAIN_LIMITS = (0.85, 1.15)   # ganho de cor por canal nas luzes
SUN_RENDER_FRAC = 0.10             # pixel "com sol no render": > 10% do sol direto (percentil 99,9)
SUN_BG_SIGMA_FRAC = 0.025          # raio da foto "sem sol" interpolada dentro das manchas
SUN_MIN_PX = 20                    # pixels mínimos (normais ou estourados) para ajustar o sol
RMS_SIGMA_FRAC = 0.04              # escala do erro da luz reportado (``rel_rms``): 4% do lado maior
CENSOR_ITERS = 12

# máscara de calibração (PNG de 3 níveis, ver ``scene_spec.calibration_mask``)
MASK_NORMAL = 255
MASK_LOWER = 128


def closure_box(extent: list[float], margin: float = 0.35) -> tuple[float, float, float, float]:
    """Retângulo das paredes de fechamento (lados não fotografados), o mesmo de
    ``blender_worker.closure_walls``: a extensão da sala + margem, e nunca passando à frente da
    câmera (y ≥ −0,6 m)."""
    xmin, ymin, xmax, ymax = extent
    return xmin - margin, min(ymin, -0.6) - margin, xmax + margin, ymax + margin


def roof_polygons(room: dict, margin: float = 0.35, reach: float = 60.0) -> list[list[tuple[float, float]]]:
    """Planta do "telhado" que só faz sombra (acima do teto da sala-proxy).

    O teto visível não projeta sombra (estendido além das paredes, ele taparia o sol que entra
    pela janela). Sem nada por cima, porém, o sol entra por cima das paredes e ilumina quase
    todo o piso no render, onde a foto não tem sol nenhum. O telhado é o retângulo das paredes
    de fechamento menos as faixas atrás de cada parede fotografada (no vão de cada parede, para
    fora dela): cobre a sala inteira, inclusive salas não convexas (bay windows), e nunca fica no
    caminho do sol depois que ele sai pela janela. Devolve os anéis externos (x, y)."""
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    x0, y0, x1, y1 = closure_box(room["extent"], margin)
    strips = []
    for w in room["walls"]:
        p0, p1 = np.asarray(w["p0"], float)[:2], np.asarray(w["p1"], float)[:2]
        L = float(np.linalg.norm(p1 - p0))
        if L < 1e-6:
            continue
        d = (p1 - p0) / L
        n = np.asarray(w["normal"], float)[:2]
        n = n / max(1e-9, float(np.linalg.norm(n)))
        ext = w.get("ext", 0.35)
        e0, e1 = (float(ext), float(ext)) if np.isscalar(ext) else (float(ext[0]), float(ext[1]))
        a, b = p0 - d * e0, p1 + d * e1
        out = -n * reach                              # lado de fora da parede (a normal aponta para dentro)
        strips.append(Polygon([tuple(a), tuple(b), tuple(b + out), tuple(a + out)]))
    roof = box(x0, y0, x1, y1)
    if strips:
        roof = roof.difference(unary_union(strips))
    parts = [roof] if roof.geom_type == "Polygon" else list(getattr(roof, "geoms", []))
    return [[(float(x), float(y)) for x, y in list(p.exterior.coords)[:-1]]
            for p in parts if p.geom_type == "Polygon" and p.area > 1e-4]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / max(1e-12, float(np.linalg.norm(v)))


def window_frame(room: dict, o: dict):
    """(P0, direção, normal) da parede da abertura ``o`` em 3D (z = 0 no piso)."""
    wl = next(w for w in room["walls"] if w["id"] == o["wall"])
    p0, p1 = np.asarray(wl["p0"], float)[:2], np.asarray(wl["p1"], float)[:2]
    d = _unit(np.r_[p1 - p0, 0.0])
    n = _unit(np.r_[np.asarray(wl["normal"], float)[:2], 0.0])
    return np.r_[p0, 0.0], d, n


def project_points(spec: dict, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pontos da sala → pixels (u, v) da placa e profundidade na câmera (OpenCV)."""
    M = np.asarray(spec["cam_to_room"], np.float64)
    K = spec["intrinsics"]
    Pc = (P - M[:3, 3]) @ M[:3, :3]
    z = Pc[:, 2]
    zz = np.where(np.abs(z) > 1e-9, z, 1e-9)
    return np.stack([K["fx"] * Pc[:, 0] / zz + K["cx"], K["fy"] * Pc[:, 1] / zz + K["cy"]], 1), z


def sun_apertures(spec: dict, patch: np.ndarray, min_hits: int = 20, frame_frac: float = 0.05,
                  margin: float = 0.04, grid: int = 12) -> list[dict]:
    """Parte de cada janela por onde o sol da FOTO de fato entra.

    No render, o sol passa pela abertura inteira; na foto, persianas, caixilhos e telas deixam só
    uma parte (as manchas de ``sunlight.py`` costumam ser bem menores que a projeção da janela).
    Cada pixel de mancha é levado ao piso pelo raio da câmera e dali, na direção do sol, até o
    plano da parede: os pontos que caem numa janela dão a "abertura de sol" dela (percentis 2–98 +
    margem). Janelas voltadas para o sol sem nenhuma mancha e com a projeção no enquadramento
    ficam fechadas para o sol; sem evidência (projeção fora da foto), ficam como estão.

    Devolve, por janela: ``{"janela", "abertura": [s0, s1, z0, z1], "estado", "acertos",
    "sol": [s0, s1, z0, z1] | None}``; ``estado`` ∈ {"parcial", "fechada", "sem_evidencia",
    "sem_sol"}."""
    room, sun = spec["room"], (spec.get("lighting") or {}).get("sun") or {}
    if not sun.get("direction"):
        return []
    d = _unit(sun["direction"])
    W, H = int(spec["width"]), int(spec["height"])
    K = spec["intrinsics"]
    M = np.asarray(spec["cam_to_room"], np.float64)
    patch = np.asarray(patch, bool)
    ph, pw = patch.shape
    ys, xs = np.nonzero(patch)
    u, v = (xs + 0.5) * W / pw, (ys + 0.5) * H / ph
    rays = np.stack([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], np.ones_like(u)], 1) @ M[:3, :3].T
    c = M[:3, 3]
    down = rays[:, 2] < -1e-6
    F = c + rays[down] * (-c[2] / rays[down, 2])[:, None]          # pontos das manchas no piso
    out = []
    for i, o in enumerate(room["openings"]):
        if o["kind"] != "window":
            continue
        s0, s1, z0, z1 = (float(o[k]) for k in ("s0", "s1", "z0", "z1"))
        P0, dw, n = window_frame(room, o)
        rec = {"janela": i, "abertura": [s0, s1, z0, z1], "acertos": 0, "sol": [s0, s1, z0, z1]}
        den = float(d @ n)
        if den > -0.05:                       # o sol não está do lado de fora desta janela
            rec.update(estado="sem_sol", sol=None)
            out.append(rec)
            continue
        uu = ((P0 - F) @ n) / den
        Hp = F + uu[:, None] * d
        s, z = (Hp - P0) @ dw, Hp[:, 2]
        tol = 0.05
        hit = (uu > 0) & (s > s0 - tol) & (s < s1 + tol) & (z > z0 - tol) & (z < z1 + tol)
        rec["acertos"] = int(hit.sum())
        if rec["acertos"] >= min_hits:
            ps, pz = s[hit], z[hit]
            rec["sol"] = [max(s0, float(np.percentile(ps, 2)) - margin), min(s1, float(np.percentile(ps, 98)) + margin),
                          max(z0, float(np.percentile(pz, 2)) - margin), min(z1, float(np.percentile(pz, 98)) + margin)]
            rec["estado"] = "parcial"
        else:
            # pegada do feixe no piso: fração que cai dentro da foto
            gs, gz = np.meshgrid(np.linspace(s0, s1, grid), np.linspace(z0, z1, grid))
            Pw = P0 + gs.reshape(-1, 1) * dw + gz.reshape(-1, 1) * np.array([0.0, 0.0, 1.0])
            Q = Pw - d * (Pw[:, 2] / d[2])[:, None]
            uv, zc = project_points(spec, Q)
            inside = (zc > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if float(inside.mean()) >= frame_frac:
                rec.update(estado="fechada", sol=None)
            else:
                rec["estado"] = "sem_evidencia"
        out.append(rec)
    return out


BLOCKER_OUT = 0.05     # m: as "persianas" passam da abertura da parede (sem fresta de sol na borda)
BLOCKER_SEAM = 0.002   # m: sobreposição entre persianas vizinhas (sem fresta na emenda)


def blocker_rects(ap: dict) -> list[list[float]]:
    """Retângulos [s0, s1, z0, z1] da janela que não deixam o sol passar: o complemento da abertura
    de sol dentro da abertura da parede. Nas bordas da abertura da parede eles passam ``BLOCKER_OUT``
    para trás da parede (um raio rasante que entra pelo vão entre a parede e a persiana deixaria
    uma "lâmina" de sol), e entre si se sobrepõem ``BLOCKER_SEAM``; nunca invadem a abertura de sol."""
    s0, s1, z0, z1 = ap["abertura"]
    sol = ap.get("sol")
    if ap.get("estado") in ("sem_sol", "sem_evidencia"):
        return []
    O, E = BLOCKER_OUT, BLOCKER_SEAM
    if sol is None:
        return [[s0 - O, s1 + O, z0 - O, z1 + O]]
    a0, a1, b0, b1 = sol
    edge = lambda v, ref: abs(v - ref) < 1e-9  # noqa: E731
    out = []
    if a0 - s0 > 1e-3:
        out.append([s0 - O, a0, z0 - O, z1 + O])
    if s1 - a1 > 1e-3:
        out.append([a1, s1 + O, z0 - O, z1 + O])
    left = a0 - (O if edge(a0, s0) else E)
    right = a1 + (O if edge(a1, s1) else E)
    if b0 - z0 > 1e-3:
        out.append([left, right, z0 - O, b0])
    if z1 - b1 > 1e-3:
        out.append([left, right, b1, z1 + O])
    return out


def masks_from_png(m8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(normal, limite_inferior) da máscara gravada: 255 = pixel usado no ajuste, 128 = quase
    saturado (só limite inferior), 0 = fora. Máscaras antigas (0/255) não têm limites."""
    m8 = np.asarray(m8)
    normal = m8 >= 192
    lower = (m8 >= 64) & (m8 < 192)
    return normal, lower


def nnls_censored(A: np.ndarray, b: np.ndarray, A_lo: np.ndarray | None = None,
                  b_lo: np.ndarray | None = None, iters: int = CENSOR_ITERS,
                  lower_weight: float = 1.0) -> np.ndarray:
    """Mínimos quadrados não negativos com linhas de limite inferior (``A_lo·x ≥ b_lo``).

    Minimiza ``‖A·x − b‖² + w·Σ max(0, b_lo − A_lo·x)²`` com ``x ≥ 0`` por conjunto ativo: as
    linhas de limite que a solução ainda viola entram como equações (peso ``w``) e o NNLS é
    refeito, até o conjunto parar de mudar. Um pixel estourado só "puxa" a predição para cima
    quando ela fica abaixo do valor recortado; nunca para baixo."""
    A = np.atleast_2d(np.asarray(A, np.float64))
    b = np.asarray(b, np.float64).ravel()
    n = A.shape[1]
    A = A.reshape(len(b), n)
    if A_lo is None or b_lo is None or len(b_lo) == 0 or lower_weight <= 0:
        return _nnls(A, b)
    b_lo = np.asarray(b_lo, np.float64).ravel()
    A_lo = np.asarray(A_lo, np.float64).reshape(len(b_lo), n)
    sw = float(np.sqrt(lower_weight))

    def loss(x):
        return float(np.sum((A @ x - b) ** 2) + lower_weight * np.sum(np.maximum(0.0, b_lo - A_lo @ x) ** 2))

    x = _nnls(A, b)
    best, best_loss = x, loss(x)
    active = None
    for _ in range(iters):            # Newton semissuave da perda quadrática "hinge"
        viol = A_lo @ x < b_lo - 1e-12
        if active is not None and np.array_equal(viol, active):
            break
        active = viol
        x = _nnls(np.vstack([A, sw * A_lo[active]]), np.concatenate([b, sw * b_lo[active]]))
        lx = loss(x)
        if lx < best_loss:
            best, best_loss = x, lx
    return best


def _nnls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NNLS do SciPy, com solução fechada por casos quando ele não existe (até 2 incógnitas)."""
    if len(b) == 0:
        return np.zeros(A.shape[1])
    try:
        from scipy.optimize import nnls

        return nnls(A.astype(np.float64), b.astype(np.float64))[0]
    except ImportError:
        if A.shape[1] <= 2:
            return _nnls_small(A, b)
        raise


def _nnls_small(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NNLS por enumeração dos suportes (1 ou 2 incógnitas)."""
    n = A.shape[1]
    best, best_err = np.zeros(n), float(np.sum(b ** 2))
    supports = [[j] for j in range(n)] + ([list(range(n))] if n > 1 else [])
    for sup in supports:
        try:
            xs = np.linalg.lstsq(A[:, sup], b, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        if np.all(xs >= 0):
            x = np.zeros(n)
            x[sup] = xs
            err = float(np.sum((A @ x - b) ** 2))
            if err < best_err:
                best, best_err = x, err
    return best


def ambient_share_limit(photo_lum: np.ndarray) -> float:
    """Limite da parcela do ambiente pela claridade da sala (luminância linear mediana das
    superfícies da máscara): 0,5 numa sala média/escura até 0,7 numa sala branca."""
    if photo_lum.size == 0:
        return AMBIENT_SHARE_MIN
    med = float(np.median(photo_lum))
    t = float(np.clip((med - BRIGHT_LUM[0]) / (BRIGHT_LUM[1] - BRIGHT_LUM[0]), 0.0, 1.0))
    return AMBIENT_SHARE_MIN + t * (AMBIENT_SHARE_MAX - AMBIENT_SHARE_MIN)


def sun_pixels(sun_render_lum: np.ndarray, photo_patch: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(afetados, ajuste): pixels que o sol muda (ficam fora do ajuste das outras luzes) e pixels
    usados para ajustar o sol (manchas da foto que o sol do render também ilumina; sem a máscara da
    foto, os pixels iluminados no render)."""
    pos = sun_render_lum[sun_render_lum > 0]
    lit = np.zeros(sun_render_lum.shape, bool)
    if pos.size:
        # sol direto ≫ luz do sol rebatida (paredes de fechamento): limiar pelo topo da distribuição
        lit = sun_render_lum > SUN_RENDER_FRAC * float(np.percentile(pos, 99.9))
    if photo_patch is None:
        return lit, lit
    patch = np.asarray(photo_patch, bool)
    return lit | patch, lit & patch


def sunless_background(b: np.ndarray, m: np.ndarray, sigma_frac: float = SUN_BG_SIGMA_FRAC) -> np.ndarray:
    """A foto "sem o sol" dentro das manchas: a luminância dos pixels ``m`` (sem sol) em volta,
    interpolada por desfoque normalizado de raio curto (a mancha é preenchida pela vizinhança)."""
    from scipy import ndimage

    ok = np.asarray(m, bool)
    if not ok.any():
        return np.zeros(b.shape, np.float32)
    sig = max(1.0, sigma_frac * max(b.shape))
    wgt = ok.astype(np.float32)
    num = ndimage.gaussian_filter(np.where(ok, b, 0.0).astype(np.float32), sig)
    den = ndimage.gaussian_filter(wgt, sig)
    out = num / np.maximum(den, 1e-6)
    # onde nem o raio curto alcança pixels sem sol, cai no desfoque largo
    far = den < 1e-3
    if far.any():
        sig2 = 4 * sig
        out = np.where(far, ndimage.gaussian_filter(np.where(ok, b, 0.0).astype(np.float32), sig2)
                       / np.maximum(ndimage.gaussian_filter(wgt, sig2), 1e-6), out)
    return out.astype(np.float32)


def fit_lights(renders: dict[str, np.ndarray], photo: np.ndarray, normal: np.ndarray,
               lower: np.ndarray | None = None, emit: np.ndarray | None = None,
               sun_patch: np.ndarray | None = None, ambient_max: float | None = None) -> dict:
    """Intensidades e cor das luzes que fazem a sala vazia renderizada reproduzir a foto.

    ``renders``: {"windows", "ambient"[, "sun"]} → (h, w, 3) linear, cada um com a fonte em força 1
    (e a emissão da foto desligada); ``emit``: o render só com a emissão da foto (termo fixo);
    ``photo``: a foto linear no mesmo tamanho; ``normal``/``lower``: máscaras de calibração;
    ``sun_patch``: manchas de sol da foto. Devolve as forças, os ganhos de cor (já aplicados em
    ``pred``), a predição RGB da sala vazia e as métricas."""
    h, w = photo.shape[:2]
    lower = np.zeros((h, w), bool) if lower is None else np.asarray(lower, bool)
    normal = np.asarray(normal, bool)
    E = np.zeros((h, w, 3), np.float32) if emit is None else np.asarray(emit, np.float32)
    names = [n for n in ("windows", "ambient") if n in renders]
    R = {n: np.asarray(renders[n], np.float32) for n in renders}
    lum = {n: R[n] @ LUM for n in R}
    e_l = E @ LUM
    b_l = photo @ LUM
    has_sun = "sun" in R and float(lum["sun"].max()) > 0
    affected, fit_px = (sun_pixels(lum["sun"], sun_patch) if has_sun
                        else (np.zeros((h, w), bool), np.zeros((h, w), bool)))
    if ambient_max is None:
        ambient_max = ambient_share_limit(b_l[normal | lower])

    # 1) janelas + ambiente fora da área do sol (emissão como termo fixo). Os limites inferiores
    #    nunca pesam mais que os pixels normais no total: numa sala branca quase toda estourada
    #    eles dominariam o ajuste e inflariam as luzes por causa de pixels que o modelo não alcança
    P, Lo = normal & ~affected, lower & ~affected
    wl = min(1.0, float(P.sum()) / max(1.0, float(Lo.sum())))
    cols = lambda m: np.stack([lum[n][m] for n in names], 1)  # noqa: E731
    x = nnls_censored(cols(P), b_l[P] - e_l[P], cols(Lo), b_l[Lo] - e_l[Lo], lower_weight=wl)
    k = dict(zip(names, (float(v) for v in x)))
    pred_P = e_l[P] + cols(P) @ x
    amb_share = (float((lum["ambient"][P] * k["ambient"]).sum() / max(1e-9, pred_P.sum()))
                 if "ambient" in k and P.any() else 0.0)
    capped = False
    if "ambient" in k and amb_share > ambient_max and "windows" in k and lum["windows"][P].any():
        # a luz da parte não fotografada não pode dominar: ambiente fixo no limite, refaz as janelas
        k["ambient"] = ambient_max * float(b_l[P].mean()) / max(1e-9, float(lum["ambient"][P].mean()))
        rest = e_l + k["ambient"] * lum["ambient"]
        kw = nnls_censored(lum["windows"][P][:, None], b_l[P] - rest[P],
                           lum["windows"][Lo][:, None], b_l[Lo] - rest[Lo], lower_weight=wl)
        k["windows"] = float(kw[0])
        capped = True

    # 2) sol à parte, só nas manchas: a força vem do EXCESSO da mancha sobre a própria foto em
    #    volta dela (a foto "sem sol" interpolada da vizinhança), não do resíduo do modelo. Perto da
    #    janela a luz de área costuma passar da foto, e o resíduo zeraria o sol; o excesso local é
    #    a luz do sol que o piso de fato recebe (o resto do nível fica com o mapa de ganho).
    #    Pixels normais da mancha: mínimos quadrados. Pixels estourados: o sol tem de estourar ao
    #    menos a METADE deles (mediana da força exigida no miolo da mancha), nunca o pior pixel da
    #    borda. A vizinhança inclui os pixels estourados em volta (com o valor recortado): se ela
    #    também estoura, a foto não diz nada da força do sol e o limite some.
    k_sun, sun_info = 0.0, {"ajustado_a_parte": False, "pixels": 0, "pixels_limite_inferior": 0}
    if has_sun:
        Sn, Sl = fit_px & normal, fit_px & lower
        bg = sunless_background(b_l, (normal | lower) & ~affected)
        sun_info.update(pixels=int(Sn.sum()), pixels_limite_inferior=int(Sl.sum()),
                        mascara_foto=sun_patch is not None,
                        excesso_mancha=round(float(np.median((b_l - bg)[Sn | Sl])), 4) if (Sn | Sl).any() else None)
        rs = lum["sun"]
        s_n = s_lo = 0.0
        if Sn.sum() >= SUN_MIN_PX:
            s_n = float(nnls_censored(rs[Sn][:, None], b_l[Sn] - bg[Sn])[0])
        if Sl.sum() >= SUN_MIN_PX:
            core = Sl & (rs >= 0.5 * float(np.median(rs[Sn | Sl])))
            need = (b_l[core] - bg[core]) / np.maximum(rs[core], 1e-6)
            s_lo = max(0.0, float(np.median(need))) if need.size >= SUN_MIN_PX else 0.0
        if Sn.sum() >= SUN_MIN_PX or Sl.sum() >= SUN_MIN_PX:
            k_sun = max(s_n, s_lo)
            sun_info.update(ajustado_a_parte=True, forca_pixels_normais=round(s_n, 4),
                            forca_minima_estourados=round(s_lo, 4))
    k["sun"] = k_sun

    # 3) cor global das luzes (por canal), sem mudar a luminância ajustada
    light = np.zeros((h, w, 3), np.float32)
    for n in R:
        if k.get(n, 0.0) > 0:
            light += np.float32(k[n]) * R[n]
    gains = np.ones(3)
    m = normal
    if m.any() and float((light[m] @ LUM).sum()) > 1e-6:
        num = (photo[m] - E[m]).sum(0).astype(np.float64)
        den = np.maximum(light[m].sum(0).astype(np.float64), 1e-9)
        g = np.maximum(num / den, 1e-3)
        lsum = light[m].sum(0).astype(np.float64)
        g = g * float(lsum @ LUM) / max(1e-12, float((g * lsum) @ LUM))
        gains = np.clip(g, *COLOR_GAIN_LIMITS)
    pred = (E + light * gains.astype(np.float32)).astype(np.float32)

    # métricas (luminância, pixels normais da máscara). Com albedo uniforme por superfície o erro
    # por pixel inclui a textura (rodapés, manchas do carpete); o erro da LUZ é o de baixa frequência
    pl = pred @ LUM
    viol = float(np.mean(pl[lower] < b_l[lower] - 1e-3)) if lower.any() else 0.0
    total = float(pl[normal].sum()) if normal.any() else 0.0
    return {
        "window_power_per_m2": k.get("windows", 0.0), "ambient_strength": k.get("ambient", 0.0),
        "sun_strength": k_sun, "color_ratio": [float(v) for v in gains], "pred": pred,
        "rel_rms": lowpass_rel_rms(pl, b_l, normal), "rel_rms_pixel": lowpass_rel_rms(pl, b_l, normal, 0.0),
        "rel_rms_sem_sol": lowpass_rel_rms(pl, b_l, normal & ~affected),
        "emissao_fracao": round(float(e_l[normal].sum()) / max(1e-9, total), 4) if emit is not None else 0.0,
        "ambiente_fracao": round(float((k.get("ambient", 0.0) * lum["ambient"])[normal].sum()) / max(1e-9, total), 4)
        if "ambient" in lum else 0.0,
        "ambiente_limite": round(float(ambient_max), 3), "ambiente_limitado": capped,
        "limite_inferior_violado": round(viol, 4), "sol": sun_info,
    }


def lowpass_rel_rms(pred_lum: np.ndarray, photo_lum: np.ndarray, m: np.ndarray,
                    sigma_frac: float = RMS_SIGMA_FRAC) -> float:
    """Erro relativo (RMS / média da foto) entre a predição e a foto nos pixels ``m``, depois de um
    desfoque normalizado de raio ``sigma_frac`` do lado maior (0 = por pixel). É o erro da luz na
    escala da sala, sem a textura do albedo (o render da calibração usa albedo uniforme)."""
    from scipy import ndimage

    m = np.asarray(m, bool)
    if not m.any():
        return 0.0
    if sigma_frac > 0:
        sig = sigma_frac * max(m.shape)
        den = np.maximum(ndimage.gaussian_filter(m.astype(np.float32), sig), 1e-6)
        pred_lum = ndimage.gaussian_filter(np.where(m, pred_lum, 0.0).astype(np.float32), sig) / den
        photo_lum = ndimage.gaussian_filter(np.where(m, photo_lum, 0.0).astype(np.float32), sig) / den
    d = pred_lum[m] - photo_lum[m]
    return float(np.sqrt(np.mean(d.astype(np.float64) ** 2)) / max(1e-6, float(photo_lum[m].mean())))
