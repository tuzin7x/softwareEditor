"""Sombra de contato analítica (PT10): a linha escura onde o móvel encosta no piso.

A emissão da foto (PT8) deixa a luz rebatida do render igual à da sala, mas suaviza as sombras do
*shadow catcher*: some a faixa escura de poucos centímetros que toda foto real tem sob pés, bases e
vasos. Ela é somada aqui, sem segunda camada de render:

- **Modelo** (gravado pelo worker do Blender na importação, ``frames.json["contact"]`` e evento
  ``contact``): para cada peça de um item de piso, a "silhueta ortográfica de baixo" numa grade de
  ``celula_m``: a altura do primeiro ponto da malha acima de cada célula (raio vertical de baixo para
  cima, na pose final), até ``altura_max_m``. Base, pés e o fundo de um sofá entram; tampo alto não.
  Junto vão a câmera (intrínsecos + ``cam_to_room``) e o tamanho do quadro.
- **Pose por quadro** (``entry["contato"]`` de cada quadro renderizado): ``[peça, fade, D(3×4)]``,
  com ``D = M_quadro · M_repouso⁻¹`` (matriz de mundo da peça). Os pontos da silhueta seguem a peça
  em qualquer movimento (queda, giro, deslize, surgir do chão). Abaixo do piso (surgindo do chão,
  PT11) o ponto pesa pela profundidade ``|z|`` como se estivesse à mesma altura acima: a sombra de
  contato só aparece quando a peça chega perto do repouso, nunca uma mancha da base inteira antes de
  a peça aparecer.
- **Campo de oclusão** ``c`` (0..1) no piso: cada ponto pesa ``fade · 1/(1 + (h/H0)²)`` e é
  desfocado por ``σ = σ0 + κ·h`` (faixas de altura): colado no piso, linha escura e nítida;
  subindo, mais fraca e mais larga. O campo é lido em cada pixel pela homografia piso→imagem.

Na composição: ``S' = S · (1 − k·c)`` antes do peso da região de influência. Fora da vizinhança
dos móveis o peso é 0, então a regra da placa não muda (a verificação independente de
``verify.py`` continua valendo, testada).
"""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

VERSION = 1
CELL_M = 0.015              # lado da célula da silhueta de baixo (m)
HEIGHT_MAX_M = 0.6          # acima disso a peça não faz sombra de contato
EMPTY_MM = 0xFFFF           # célula sem malha até HEIGHT_MAX_M
H0_M = 0.10                 # altura em que a oclusão cai à metade
SIGMA0_M = 0.012            # desfoque da linha de contato com a peça no piso
KAPPA = 0.5                 # alargamento do desfoque por metro de altura
BANDS_M = (0.0, 0.025, 0.06, 0.12, 0.22, 0.36, math.inf)   # faixas de altura (um desfoque por faixa)
MAX_GRID = 1400             # lado máximo da grade do piso (células)
MIN_AMP = 1e-3


def attenuation(h):
    """Peso da oclusão de um ponto a ``h`` m do piso (1 colado no piso)."""
    h = np.maximum(np.asarray(h, np.float64), 0.0)
    return 1.0 / (1.0 + (h / H0_M) ** 2)


def blur_sigma(h: float) -> float:
    """Desfoque (m) da sombra de um ponto a ``h`` m do piso."""
    return SIGMA0_M + KAPPA * max(0.0, float(h))


def encode_heights(h_m: np.ndarray) -> str:
    """Alturas (m, NaN = vazio) → base64 de uint16 em mm (little endian)."""
    mm = np.where(np.isfinite(h_m), np.clip(np.round(np.nan_to_num(h_m) * 1000.0), 0, EMPTY_MM - 1), EMPTY_MM)
    return base64.b64encode(mm.astype("<u2").tobytes()).decode("ascii")


def decode_heights(s: str, n: int) -> np.ndarray:
    mm = np.frombuffer(base64.b64decode(s), dtype="<u2")
    if mm.size != n:
        raise ValueError(f"silhueta de contato com {mm.size} células (esperado {n})")
    return np.where(mm == EMPTY_MM, np.nan, mm.astype(np.float64) / 1000.0)


def part_record(pid: str, item: str, origin, cell: float, heights: np.ndarray) -> dict | None:
    """Registro de uma peça para ``frames.json["contact"]["pecas"]``. ``heights`` (ny, nx) em m
    acima do piso, NaN = vazio; ``origin`` = centro da célula [0, 0] (x, y do mundo)."""
    h = np.asarray(heights, np.float64)
    if not np.isfinite(h).any():
        return None
    ny, nx = h.shape
    return {"id": pid, "item": item, "origem": [round(float(origin[0]), 5), round(float(origin[1]), 5)],
            "celula_m": float(cell), "nx": int(nx), "ny": int(ny), "alturas_mm": encode_heights(h)}


def floor_homography(intr: dict, cam_to_room) -> np.ndarray:
    """H (3×3): (x, y, 1) do piso z = 0 → (u, v, 1) da imagem (pixel contínuo, câmera OpenCV)."""
    M = np.asarray(cam_to_room, np.float64)
    Rt = M[:3, :3].T
    t = M[:3, 3]
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64)
    return K @ np.column_stack([Rt[:, 0], Rt[:, 1], -Rt @ t])


@dataclass
class _Part:
    pts: np.ndarray            # (N, 4) pontos homogêneos da silhueta na pose de repouso (mundo)
    cell: float


class ContactModel:
    """Silhuetas de baixo + câmera; gera o campo de oclusão de cada quadro (``occlusion``)."""

    def __init__(self, meta: dict, shape: tuple[int, int] | None = None):
        self.meta = meta
        cam = meta["camera"]
        self.H = int(shape[0] if shape else meta["height"])
        self.W = int(shape[1] if shape else meta["width"])
        intr = dict(cam["intrinsics"])
        sx, sy = self.W / float(meta.get("width", self.W)), self.H / float(meta.get("height", self.H))
        for k, s in (("fx", sx), ("cx", sx), ("fy", sy), ("cy", sy)):
            intr[k] = float(intr[k]) * s
        self.Hf = floor_homography(intr, cam["cam_to_room"])
        self.Hinv = np.linalg.inv(self.Hf)
        self.parts: list[_Part | None] = []
        for p in meta.get("pecas") or []:
            nx, ny, c = int(p["nx"]), int(p["ny"]), float(p["celula_m"])
            h = decode_heights(p["alturas_mm"], nx * ny).reshape(ny, nx)
            iy, ix = np.nonzero(np.isfinite(h))
            x0, y0 = (float(v) for v in p["origem"])
            pts = np.column_stack([x0 + ix * c, y0 + iy * c, h[iy, ix], np.ones(ix.size)])
            self.parts.append(_Part(pts, c) if pts.size else None)

    @classmethod
    def from_meta(cls, meta: dict | None, shape: tuple[int, int] | None = None) -> "ContactModel | None":
        if not meta or not meta.get("pecas") or not meta.get("camera"):
            return None
        return cls(meta, shape)

    @property
    def n_parts(self) -> int:
        return sum(1 for p in self.parts if p is not None)

    def frame(self, poses) -> "ContactFrame | None":
        """Quadro com as poses ``entry["contato"]`` (None = nenhuma peça de piso visível)."""
        if not poses:
            return None
        return ContactFrame(self, poses)

    # ------------------------------------------------------------------ conta
    def samples(self, poses) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pontos (x, y), alturas (m ≥ 0), pesos e lado da célula de todas as peças nas ``poses``."""
        xy, hh, aa, cc = [], [], [], []
        for rec in poses:
            k, fade = int(rec[0]), float(rec[1])
            if not (0 <= k < len(self.parts)) or self.parts[k] is None or fade <= MIN_AMP:
                continue
            part = self.parts[k]
            D = np.asarray(rec[2:14], np.float64).reshape(3, 4)
            P = part.pts @ D.T
            h = np.abs(P[:, 2])                   # abaixo do piso (surgir do chão): pela profundidade
            a = fade * attenuation(h)
            keep = a > MIN_AMP
            xy.append(P[keep, :2])
            hh.append(h[keep])
            aa.append(a[keep])
            cc.append(np.full(int(keep.sum()), part.cell))
        if not xy:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0)
        return np.concatenate(xy), np.concatenate(hh), np.concatenate(aa), np.concatenate(cc)

    def floor_field(self, poses) -> tuple[np.ndarray, tuple[float, float, float]] | None:
        """Oclusão ``c`` numa grade do piso e (x0, y0, passo) do centro da célula [0, 0]."""
        xy, h, a, cell = self.samples(poses)
        if not a.size:
            return None
        smax = blur_sigma(float(h.max()))
        # passo da grade: a maior célula das silhuetas (sem buracos entre amostras vizinhas)
        cmax = float(cell.max())
        pad = 4.0 * smax + 2 * cmax
        lo, hi = xy.min(0) - pad, xy.max(0) + pad
        g = max(cmax, float((hi - lo).max()) / MAX_GRID)
        nx, ny = int(math.ceil((hi[0] - lo[0]) / g)) + 2, int(math.ceil((hi[1] - lo[1]) / g)) + 2
        dens = (cell / g) ** 2                    # área de cada amostra em células da grade
        gx, gy = (xy[:, 0] - lo[0]) / g, (xy[:, 1] - lo[1]) / g
        ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
        fx, fy = gx - ix, gy - iy
        occ = np.zeros((ny, nx), np.float64)      # oclusão acumulada (união das faixas)
        for b0, b1 in zip(BANDS_M[:-1], BANDS_M[1:]):
            sel = (h >= b0) & (h < b1)
            if not sel.any():
                continue
            grid = np.zeros((ny, nx), np.float64)
            w = a[sel] * dens[sel]
            for dy, dx, wt in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)),
                               (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
                np.add.at(grid, (iy[sel] + dy, ix[sel] + dx), w * wt[sel])
            np.minimum(grid, 1.0, out=grid)
            hm = float(np.median(h[sel]))
            grid = ndimage.gaussian_filter(grid, blur_sigma(hm) / g, mode="constant")
            occ = 1.0 - (1.0 - occ) * (1.0 - grid)
        return np.clip(occ, 0.0, 1.0), (float(lo[0]), float(lo[1]), g)

    def occlusion(self, poses, box, where: np.ndarray | None = None) -> np.ndarray | None:
        """Oclusão ``c`` (float32, h×w) nos pixels do retângulo ``box`` = [x0, y0, x1, y1].

        Só os pixels que podem cair na grade do campo são calculados: o retângulo da imagem que
        contém a projeção da grade (com uma célula de folga) e, com ``where`` (bool h×w, p. ex. a
        região de influência), só os pixels marcados. Nos demais ``c`` = 0; onde é calculado, o
        valor é o mesmo da conta no recorte inteiro (amostragem bilinear ponto a ponto)."""
        ff = self.floor_field(poses)
        if ff is None:
            return None
        field, (gx0, gy0, g) = ff
        X0, Y0, X1, Y1 = (int(v) for v in box)
        out = np.zeros((max(0, Y1 - Y0), max(0, X1 - X0)), np.float32)
        # retângulo da imagem que contém a grade projetada (todos os cantos à frente da câmera:
        # a imagem de um retângulo do piso é um quadrilátero convexo); senão, o recorte inteiro
        ny, nx = field.shape
        cx_ = (gx0 - g, gx0 + nx * g)
        cy_ = (gy0 - g, gy0 + ny * g)
        corners = np.array([[x, y, 1.0] for x in cx_ for y in cy_]) @ self.Hf.T
        sx0, sy0, sx1, sy1 = X0, Y0, X1, Y1
        if (corners[:, 2] > 1e-9).all():
            uv = corners[:, :2] / corners[:, 2:]
            umin, vmin = np.floor(uv.min(0)) - 2
            umax, vmax = np.ceil(uv.max(0)) + 2
            sx0, sx1 = int(np.clip(umin, X0, X1)), int(np.clip(umax, X0, X1))
            sy0, sy1 = int(np.clip(vmin, Y0, Y1)), int(np.clip(vmax, Y0, Y1))
        if sx1 <= sx0 or sy1 <= sy0:
            return out
        sub = (slice(sy0 - Y0, sy1 - Y0), slice(sx0 - X0, sx1 - X0))
        if where is not None:
            iy, ix = np.nonzero(np.asarray(where, bool)[sub])
            if not iy.size:
                return out
            uu, vv = ix + (sx0 + 0.5), iy + (sy0 + 0.5)
        else:
            uu, vv = (a.ravel() for a in np.meshgrid(np.arange(sx0, sx1, dtype=np.float64) + 0.5,
                                                     np.arange(sy0, sy1, dtype=np.float64) + 0.5))
        Hi = self.Hinv
        q = Hi[:, 0, None] * uu + Hi[:, 1, None] * vv + Hi[:, 2, None]
        front = q[2] > 1e-12                      # raio do pixel desce até o piso à frente da câmera
        wz = np.where(front, q[2], 1.0)
        x, y = q[0] / wz, q[1] / wz
        cx, cy = (x - gx0) / g, (y - gy0) / g
        cx = np.where(front, cx, -10.0)
        val = ndimage.map_coordinates(field, [cy, cx], order=1, mode="constant", cval=0.0)
        val = np.where(front, np.clip(val, 0.0, 1.0), 0.0).astype(np.float32)
        if where is not None:
            out[sub][iy, ix] = val
        else:
            out[sub] = val.reshape(sy1 - sy0, sx1 - sx0)
        return out


class ContactFrame:
    """Modelo + poses de um quadro; ``occlusion(box)`` é chamado pelo compositor no seu recorte."""

    def __init__(self, model: ContactModel, poses):
        self.model = model
        self.poses = poses

    def occlusion(self, box, where: np.ndarray | None = None) -> np.ndarray | None:
        return self.model.occlusion(self.poses, box, where)


def pose_record(k: int, fade: float, D) -> list:
    """Entrada de ``entry["contato"]``: [peça, fade, D 3×4 por linhas] (5 casas)."""
    D = np.asarray(D, np.float64)[:3, :4]
    return [int(k), round(float(fade), 4)] + [round(float(v), 5) for v in D.ravel()]


IDENTITY = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
