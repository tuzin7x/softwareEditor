"""Mede a composição + verificação + codificação do modo B (PT-V3-7) numa sala sintética, sem modelos.

Uso (na raiz do repositório):
    python scripts/medir_composicao_modo_b.py [--largura 1920 --altura 1440] [--backend auto|numpy|torch|cuda]
                                              [--codec auto|libx264|h264_nvenc] [--sem-mp4] [--mkv]

A sala tem 6 peças (tapete, cadeira atrás do sofá, sofá, almofada apoiada, poltrona na borda, mesa isolada), em
qualquer resolução. Imprime o relatório (tempos por etapa) e o tempo normalizado para 240 quadros. Serve para a
medição na GPU do usuário (PLANO_V3 §7.2): rodar com ``--backend cuda --codec h264_nvenc``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mobilia.layers2d import choreo as C  # noqa: E402
from mobilia.layers2d import layers as L  # noqa: E402
from mobilia.layers2d import video as VB  # noqa: E402
from mobilia.removal import plate as rp  # noqa: E402
from mobilia.video import encoder as E  # noqa: E402

# (rótulo, y0, y1, x0, x1, cor, tipo) em frações do quadro, de trás para frente
ROOM = [
    ("rug", .708, .983, .30, .825, (40, 80, 160), "solto"),
    ("chair", .292, .625, .50, .65, (90, 60, 30), "solto"),
    ("sofa", .50, .917, .40, .75, (180, 30, 30), "solto"),
    ("cushion", .433, .567, .50, .60, (230, 210, 40), "solto"),
    ("armchair", .417, .833, .85, 1.0, (60, 120, 60), "solto"),
    ("table", .25, .50, .10, .30, (200, 200, 210), "solto"),
]


def synthetic_room(H: int, W: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (150.0, 140.0, 130.0), np.float32) + rng.normal(0, 4.0, (H, W, 3)).astype(np.float32)
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img += 30 * yy                                           # gradiente (luz)
    visible = []
    for lab, y0, y1, x0, x1, col, kind in ROOM:
        m = np.zeros((H, W), bool)
        m[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = True
        tex = rng.normal(0, 6.0, (int(m.sum()), 3)).astype(np.float32)
        img[m] = np.asarray(col, np.float32) + tex
        for v in visible:
            v &= ~m
        visible.append(m)
    photo = np.clip(img, 0, 255).round().astype(np.uint8)
    pieces = [SimpleNamespace(id=k, label=r[0], kind=r[6], alpha=visible[k].astype(np.float32), mask=visible[k],
                              touches=()) for k, r in enumerate(ROOM)]
    mask = rp.core_mask(pieces, (H, W))
    plate = photo.copy()
    fl = np.full((H, W, 3), (150.0, 140.0, 130.0), np.float32) + 30 * yy + \
        rng.normal(0, 4.0, (H, W, 3)).astype(np.float32)
    pt = np.clip(fl, 0, 255).round().astype(np.uint8)
    plate[mask] = pt[mask]
    res = SimpleNamespace(plate=plate, mask=mask, shadow=np.zeros((H, W), bool), effect=np.zeros((H, W), bool))
    return photo, res, pieces


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--largura", type=int, default=1920)
    ap.add_argument("--altura", type=int, default=1440)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--codec", default="auto")
    ap.add_argument("--sem-mp4", action="store_true")
    ap.add_argument("--mkv", action="store_true")
    a = ap.parse_args(argv)
    os.environ[E.ENV_CODEC] = a.codec
    t0 = time.perf_counter()
    photo, res, pieces = synthetic_room(a.altura, a.largura)
    st = L.build_layers(photo, res, pieces)
    t_layers = time.perf_counter() - t0
    t0 = time.perf_counter()
    ch = C.plan(st)
    t_plan = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as d:
        mp4 = None if a.sem_mp4 else Path(d) / "modo_b.mp4"
        mkv = Path(d) / "auditoria.mkv" if a.mkv else None
        rep = VB.render_mode_b(st, ch, photo, res.mask, mp4, audit_mkv=mkv, backend=a.backend)
        if mp4 is not None:
            rep["video_modo_b"]["mp4_mb"] = round(mp4.stat().st_size / 1e6, 2)
    r = rep["video_modo_b"]
    n = r["quadros"]
    r["preparo_s"] = {"camadas": round(t_layers, 2), "coreografia": round(t_plan, 2)}
    r["por_240_quadros_s"] = {k: round(v * (240.0 / n if k != "preparo_compositor" else 1.0), 2)
                              for k, v in r["tempo_s"].items()}
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0 if rep["verificacao_modo_b"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
