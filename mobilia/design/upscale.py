"""'Ampliar 4K': Real-ESRGAN x4plus (RRDBNet, licença BSD-3-Clause) em blocos, sem a dependência basicsr.

A versão escolhida (1024 px) vira 4096 px pelo modelo e é reduzida (Lanczos) para o lado maior pedido (3840 =
4K UHD). Os blocos se sobrepõem (``pad``) e só o miolo de cada um é usado, então não há costura. GPU: fp16, blocos
de 512; CPU: fp32/bf16, blocos de 256 (lento, ~1–2 min por imagem; só para máquinas sem placa de vídeo).
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


def _build(num_block: int = 23, num_feat: int = 64, num_grow: int = 32):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class RDB(nn.Module):
        def __init__(self):
            super().__init__()
            g = num_grow
            self.conv1 = nn.Conv2d(num_feat, g, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + g, g, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * g, g, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * g, g, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * g, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1, self.rdb2, self.rdb3 = RDB(), RDB(), RDB()

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB() for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet()


class Upscaler:
    scale = 4

    def __init__(self, device: str = "auto", weights: Path | None = None, offline: bool = True):
        import torch

        from ..analysis.weights import esrgan_path
        from .engine import pick_device

        self.device = pick_device(device)
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        net = _build()
        sd = torch.load(weights or esrgan_path(offline=offline), map_location="cpu", weights_only=True)
        sd = sd.get("params_ema", sd.get("params", sd))
        net.load_state_dict(sd, strict=True)
        self.net = net.eval().to(self.device, self.dtype)
        self.tile = 512 if self.device.startswith("cuda") else 256

    def __call__(self, rgb: np.ndarray, pad: int = 16, progress=None) -> np.ndarray:
        import torch

        H, W = rgb.shape[:2]
        s, t = self.scale, self.tile
        out = np.zeros((H * s, W * s, 3), np.uint8)
        x = torch.from_numpy(np.array(rgb, dtype=np.uint8, copy=True)).permute(2, 0, 1)[None].float() / 255.0
        tiles = [(y0, x0) for y0 in range(0, H, t) for x0 in range(0, W, t)]
        for k, (y0, x0) in enumerate(tiles):
            y1, x1 = min(H, y0 + t), min(W, x0 + t)
            py0, px0, py1, px1 = max(0, y0 - pad), max(0, x0 - pad), min(H, y1 + pad), min(W, x1 + pad)
            with torch.inference_mode():
                y = self.net(x[:, :, py0:py1, px0:px1].to(self.device, self.dtype)).float().clamp_(0, 1)
            y = (y[0].permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            oy, ox = (y0 - py0) * s, (x0 - px0) * s
            out[y0 * s:y1 * s, x0 * s:x1 * s] = y[oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s]
            if progress:
                progress((k + 1) / len(tiles))
        return out


def upscale(image, out_path=None, long_side: int = 3840, device: str = "auto", upscaler: Upscaler | None = None,
            progress=None) -> tuple[np.ndarray, dict]:
    """Amplia ``image`` (caminho ou RGB uint8) 4x e reduz para ``long_side`` (0 = mantém os 4x). Salva em
    ``out_path`` (JPEG q95) se dado. Devolve a imagem e um resumo (tamanhos, tempo)."""
    from PIL import Image

    t0 = time.perf_counter()
    rgb = image if isinstance(image, np.ndarray) else np.asarray(Image.open(image).convert("RGB"))
    up = upscaler or Upscaler(device)
    big = up(rgb, progress=progress)
    if long_side and max(big.shape[:2]) != long_side:
        k = long_side / max(big.shape[:2])
        big = cv2.resize(big, (int(round(big.shape[1] * k)), int(round(big.shape[0] * k))),
                         interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(big).save(out_path, quality=95, subsampling=0)
    return big, {"de": [rgb.shape[1], rgb.shape[0]], "para": [big.shape[1], big.shape[0]],
                 "s": round(time.perf_counter() - t0, 2), "dispositivo": up.device}


__all__ = ["Upscaler", "upscale"]
