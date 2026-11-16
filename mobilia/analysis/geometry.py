"""Geometria métrica monocular com MoGe-2 (Microsoft, licença MIT).

Saída no referencial de câmera OpenCV: x para a direita, y para baixo, z para frente.
Intrínsecos em pixels na convenção de borda (o centro do pixel i fica em i + 0.5),
que é a mesma convenção da câmera do Blender (validado em testes).
"""
from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def fov_x_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.width / (2 * self.fx))))

    @property
    def fov_y_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.height / (2 * self.fy))))

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def rays(self) -> np.ndarray:
        """Direções (não normalizadas, z=1) de cada pixel, shape (H, W, 3)."""
        u = np.arange(self.width, dtype=np.float64) + 0.5
        v = np.arange(self.height, dtype=np.float64) + 0.5
        uu, vv = np.meshgrid(u, v)
        return np.stack([(uu - self.cx) / self.fx, (vv - self.cy) / self.fy, np.ones_like(uu)], -1)

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        """Projeta pontos (..., 3) no referencial de câmera para pixels (..., 2) contínuos."""
        z = pts_cam[..., 2]
        u = self.fx * pts_cam[..., 0] / z + self.cx
        v = self.fy * pts_cam[..., 1] / z + self.cy
        return np.stack([u, v], -1)

    def scaled(self, width: int, height: int) -> "Intrinsics":
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy)

    def to_dict(self) -> dict:
        return dict(width=self.width, height=self.height, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)


@dataclass
class GeometryResult:
    points: np.ndarray      # (H, W, 3) float32, metros, referencial OpenCV
    depth: np.ndarray       # (H, W) float32 (z)
    normals: np.ndarray     # (H, W, 3) float32, unitárias, referencial OpenCV
    mask: np.ndarray        # (H, W) bool — pixels com geometria válida
    intrinsics: Intrinsics
    model_id: str

    def save(self, path) -> None:
        np.savez_compressed(path, points=self.points, depth=self.depth, normals=self.normals,
                            mask=self.mask, intrinsics=np.array(list(self.intrinsics.to_dict().values())),
                            model_id=np.array(self.model_id))

    @staticmethod
    def load(path) -> "GeometryResult":
        d = np.load(path, allow_pickle=False)
        w, h, fx, fy, cx, cy = d["intrinsics"].tolist()
        return GeometryResult(d["points"], d["depth"], d["normals"], d["mask"],
                              Intrinsics(int(w), int(h), fx, fy, cx, cy), str(d["model_id"]))


def estimate_geometry(rgb8: np.ndarray, model_id: str = "Ruicheng/moge-2-vitl-normal",
                      device: str = "cpu", use_fp16: bool = False, num_tokens: int | None = None,
                      fov_x_deg: float | None = None, timings: dict | None = None) -> GeometryResult:
    """Roda o MoGe-2 sobre a foto (resolução original) e libera a VRAM ao final.

    ``timings`` (opcional) recebe ``carga`` (import + pesos + envio ao dispositivo) e ``download``
    (rede, só se os pesos ainda não estavam no cache) em segundos; o resto da chamada é a inferência.
    """
    import time

    from .weights import ensure_local

    t0 = time.perf_counter()
    dl = _ensure_weights(ensure_local, model_id)
    import torch
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(model_id).to(device).eval()
    if timings is not None:
        timings["download"] = dl
        timings["carga"] = time.perf_counter() - t0 - dl
    try:
        img = torch.from_numpy(np.ascontiguousarray(rgb8)).to(device=device, dtype=torch.float32)
        img = img.permute(2, 0, 1) / 255.0
        kwargs = dict(use_fp16=bool(use_fp16 and device != "cpu"))
        if num_tokens:
            kwargs["num_tokens"] = int(num_tokens)
        if fov_x_deg:
            kwargs["fov_x"] = float(fov_x_deg)
        with torch.inference_mode():
            out = model.infer(img, **kwargs)
        points = out["points"].float().cpu().numpy()
        depth = out["depth"].float().cpu().numpy()
        mask = out["mask"].cpu().numpy().astype(bool)
        normals = out["normal"].float().cpu().numpy() if "normal" in out else None
        K = out["intrinsics"].float().cpu().numpy()
    finally:
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    h, w = rgb8.shape[:2]
    # MoGe devolve intrínsecos normalizados (largura/altura = 1).
    intr = Intrinsics(w, h, float(K[0, 0] * w), float(K[1, 1] * h), float(K[0, 2] * w), float(K[1, 2] * h))
    if normals is None:
        normals = normals_from_points(points, mask)
    normals = np.nan_to_num(normals).astype(np.float32)
    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mask &= depth > 0
    return GeometryResult(points, depth, normals, mask, intr, model_id)


def _ensure_weights(ensure, model_id: str) -> float:
    """Tempo de rede para completar o cache dos pesos; falhas ficam para o ``from_pretrained``
    (que dá a mensagem de erro adequada)."""
    try:
        return float(ensure(model_id))
    except Exception:
        return 0.0


def normals_from_points(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normais por diferenças finitas (fallback quando o modelo não as fornece)."""
    dx = np.zeros_like(points); dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1] = points[2:] - points[:-2]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    # orienta para a câmera
    flip = (n * points).sum(-1) > 0
    n[flip] *= -1
    n[~mask] = 0
    return n.astype(np.float32)
