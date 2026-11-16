"""Aquecimento feito pelo instalador (PLANO_V2, PT2): os custos únicos saem dos 2 minutos do vídeo.

``mobilia.cli aquecer`` roda, na instalação do motor:

1. o Blender instalado (``blender_worker --aquecimento``): renders de 64 × 64 com OptiX e com
   CUDA, que deixam os kernels de render e do denoiser compilados nos caches do driver NVIDIA
   (``CUDA_CACHE_MAXSIZE``/``OPTIX_CACHE_MAXSIZE`` de 4 GiB, ver ``runtime.GPU_CACHE_ENV``);
2. uma carga dos dois modelos de análise (MoGe-2 e OneFormer) com uma inferência minúscula, no
   mesmo dispositivo dos trabalhos: lê os pesos do disco uma vez (antivírus, cache de arquivos),
   inicializa o CUDA/cuDNN e confirma que tudo funciona sem internet.

Falhas não derrubam a instalação: ficam no resultado (``ok: false`` + ``erro``) e o primeiro
vídeo paga o custo (o aviso "primeira execução" da interface diz isso).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

Emit = Callable[[dict], None]
# Imagem sintética da carga dos modelos (o tamanho só precisa exercitar o caminho completo).
WARM_IMAGE = (192, 256)


def _progress(emit: Emit, fraction: float, message: str) -> None:
    emit({"event": "progress", "stage": "aquecimento", "fraction": round(max(0.0, min(1.0, fraction)), 3),
          "message": message})


def warm_blender(emit: Emit, out_dir: Path | None = None, span: tuple[float, float] = (0.0, 1.0)) -> dict:
    """Roda o worker do Blender em modo de aquecimento e devolve o resultado dele (``warmup_done``)."""
    from .pipeline import blender_worker_env, blender_worker_prefix

    t0 = time.perf_counter()
    tmp = None
    if out_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="mobilia_aquecimento_")
        out_dir = Path(tmp.name)
    cmd = blender_worker_prefix() + ["--aquecimento", str(out_dir)]
    result, tail = None, []
    a, b = span
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=blender_worker_env(),
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for line in proc.stdout:
            line = line.strip()
            tail = (tail + [line])[-40:]
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            kind = ev.get("event")
            if kind == "progress":
                _progress(emit, a + (b - a) * float(ev.get("fraction", 0.0)), ev.get("message", ""))
            elif kind == "log":
                emit({"event": "log", "level": ev.get("level", "info"), "message": ev.get("message", "")})
            elif kind == "warmup_done":
                result = ev.get("resultado")
        rc = proc.wait()
    except OSError as e:
        rc, tail = -1, [f"{e.__class__.__name__}: {e}"]
    finally:
        if tmp is not None:
            tmp.cleanup()
    if rc != 0 or not isinstance(result, dict):
        return {"ok": False, "erro": f"worker do Blender terminou com código {rc}",
                "saida": tail[-15:], "parede_s": round(time.perf_counter() - t0, 3)}
    result["processo_s"] = round(time.perf_counter() - t0, 3)
    return result


def _synthetic_image() -> np.ndarray:
    """Imagem fixa (gradiente + ruído com semente): mesma entrada em toda instalação."""
    h, w = WARM_IMAGE
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([200 - 80 * y / h, 190 - 60 * y / h + 20 * x / w, 180 - 40 * y / h], -1)
    return np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)


def warm_models(emit: Emit, prof=None, span: tuple[float, float] = (0.0, 1.0)) -> dict:
    """Carrega MoGe-2 e OneFormer no dispositivo dos trabalhos e faz uma inferência minúscula."""
    from .analysis.geometry import estimate_geometry
    from .analysis.segmentation import segment
    from .hardware import choose_profile, detect_hardware

    a, b = span
    t0 = time.perf_counter()
    if prof is None:
        prof = choose_profile(detect_hardware())
    rgb = _synthetic_image()
    out = {"dispositivo": prof.torch_device, "perfil": prof.name}
    steps = (("geometria", prof.geometry_model,
              lambda tm: estimate_geometry(rgb, model_id=prof.geometry_model, device=prof.torch_device,
                                           use_fp16=prof.use_fp16, num_tokens=prof.geometry_tokens, timings=tm)),
             ("segmentacao", "OneFormer",
              lambda tm: segment(rgb, device=prof.torch_device, use_fp16=prof.use_fp16, timings=tm)))
    for i, (key, label, fn) in enumerate(steps):
        _progress(emit, a + (b - a) * i / len(steps), f"Carregando o modelo de IA {label} ({prof.torch_device})")
        tm: dict = {}
        t = time.perf_counter()
        try:
            fn(tm)
            total = time.perf_counter() - t
            carga = min(total, float(tm.get("carga", 0.0)))
            out[key] = {"ok": True, "carga_s": round(carga, 3), "download_s": round(float(tm.get("download", 0.0)), 3),
                        "inferencia_s": round(total - carga - float(tm.get("download", 0.0)), 3)}
        except Exception as e:  # registra e segue: o modelo será carregado (e acusará) no trabalho
            out[key] = {"ok": False, "erro": f"{e.__class__.__name__}: {e}"}
            emit({"event": "log", "level": "warning", "message": f"Aquecimento do modelo {label} falhou: {e}"})
    out["ok"] = all(out[k].get("ok") for k, _, _ in steps)
    out["parede_s"] = round(time.perf_counter() - t0, 3)
    _progress(emit, b, "Modelos de IA prontos")
    return out


def run(emit: Emit, blender: bool = True, models: bool = True) -> dict:
    """Aquecimento completo (Blender primeiro: é o custo maior numa GPU nova)."""
    t0 = time.perf_counter()
    res: dict = {}
    if blender:
        res["blender"] = warm_blender(emit, span=(0.0, 0.6 if models else 1.0))
    if models:
        res["modelos"] = warm_models(emit, span=(0.6 if blender else 0.0, 1.0))
    res["ok"] = all(v.get("ok") for v in res.values() if isinstance(v, dict))
    res["parede_s"] = round(time.perf_counter() - t0, 3)
    return res
