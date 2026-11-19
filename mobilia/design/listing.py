"""'Exportar para anúncio' e 'Lote' (várias fotos de um imóvel).

* :func:`export_for_listing`: JPEGs prontos para WhatsApp/portais — lado maior 2048 px (leve e nítido no celular)
  ou 3840 px (4K; fotos menores são ampliadas pelo "Ampliar 4K"), qualidade 90, perfil sRGB simples, e o selo
  discreto "Imagem ilustrativa" (obrigatório na prática para mobília virtual: o comprador precisa saber que os
  móveis não existem) — ligado por padrão nas funções de redesenho, desligado no "Melhorar foto";
* :func:`property_folder`: nome de pasta seguro a partir do nome do imóvel ("Apto 302 – Rua X");
* :func:`run_batch`: processa as fotos de um imóvel em sequência com a mesma função e as mesmas escolhas, e junta
  as entregas numa pasta só.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

LABEL_TEXT = "Imagem ilustrativa"
LONG_SIDES = {"2048": 2048, "whatsapp": 2048, "4k": 3840, "3840": 3840}


def norm_long_side(v) -> int:
    k = str(v).strip().lower().replace(" ", "").replace("px", "")
    if k not in LONG_SIDES:
        raise ValueError(f"Tamanho inválido: {v!r} (opções: 2048, 4k)")
    return LONG_SIDES[k]


def default_label(funcao: str | None) -> bool:
    """Selo "Imagem ilustrativa" por padrão: sim nas funções que inventam móveis/decoração, não no "Melhorar foto"."""
    from .presets import ILLUSTRATIVE

    return (funcao or "") in ILLUSTRATIVE


def _font(size: int):
    from ..video.showcase import _font as f

    return f("IBMPlexSans-Medium.ttf", size)


def add_label(rgb: np.ndarray, text: str = LABEL_TEXT) -> np.ndarray:
    """Selo discreto no canto inferior direito: texto claro numa pílula escura semitransparente, ~1,6% do lado
    maior (legível no celular, sem brigar com a foto)."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.asarray(rgb)[..., :3]).convert("RGBA")
    W, H = img.size
    size = max(12, int(round(0.016 * max(W, H))))
    font = _font(size)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    tw, th = x1 - x0, y1 - y0
    pad_x, pad_y = int(size * 0.7), int(size * 0.45)
    m = int(round(0.018 * max(W, H)))
    bw, bh = tw + 2 * pad_x, th + 2 * pad_y
    bx, by = W - m - bw, H - m - bh
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    od.rounded_rectangle((bx, by, bx + bw, by + bh), radius=bh // 2, fill=(20, 18, 28, 120))
    od.text((bx + pad_x - x0, by + pad_y - y0), text, font=font, fill=(255, 255, 255, 225))
    return np.asarray(Image.alpha_composite(img, over).convert("RGB"))


def _fit(rgb: np.ndarray, long_side: int, upscaler=None, log=None) -> np.ndarray:
    H, W = rgb.shape[:2]
    if max(H, W) == long_side:
        return rgb
    if max(H, W) > long_side:
        k = long_side / max(H, W)
        return cv2.resize(rgb, (int(round(W * k)), int(round(H * k))), interpolation=cv2.INTER_AREA)
    if long_side >= 3000:
        # 4K de uma imagem menor: o "Ampliar 4K" (detalhe de verdade); sem o modelo, Lanczos
        try:
            if upscaler is not None:
                return upscaler(rgb, long_side)
            from .upscale import upscale

            return upscale(rgb, None, long_side=long_side)[0]
        except Exception as e:  # noqa: BLE001
            if log:
                log(f"Ampliação por IA indisponível ({e}); ampliado do jeito simples.")
    k = long_side / max(H, W)
    return cv2.resize(rgb, (int(round(W * k)), int(round(H * k))), interpolation=cv2.INTER_LANCZOS4)


def export_for_listing(images: list, out_dir, long_side: int = 2048, label: bool = False, quality: int = 90,
                       names: list[str] | None = None, upscaler=None, progress: Callable | None = None,
                       log: Callable | None = None) -> list[str]:
    """Grava cada imagem como JPEG de anúncio em ``out_dir`` (``<nome>_anuncio.jpg``) e devolve os caminhos."""
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, src in enumerate(images):
        rgb = np.asarray(Image.open(src).convert("RGB")) if not isinstance(src, np.ndarray) else src
        img = _fit(rgb, int(long_side), upscaler, log)
        if label:
            img = add_label(img)
        stem = (names[i] if names else Path(str(src)).stem if not isinstance(src, np.ndarray) else f"foto_{i + 1}")
        dest = out / f"{stem}_anuncio.jpg"
        k = 2
        while dest.exists():
            dest = out / f"{stem}_anuncio_{k}.jpg"
            k += 1
        Image.fromarray(img).save(dest, quality=int(quality), subsampling=0, optimize=True, progressive=True)
        paths.append(str(dest))
        if progress:
            progress((i + 1) / len(images))
    return paths


_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def property_folder(name: str, fallback: str = "imovel") -> str:
    """Nome de pasta seguro no Windows a partir do nome do imóvel (acentos ficam; ``/\\:*?"<>|`` saem)."""
    s = unicodedata.normalize("NFC", str(name or "")).strip()
    s = _BAD.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    if s.upper() in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?i)(com|lpt)\d", s or ""):
        s = f"_{s}"
    return s[:80] or fallback


# ====================================================================================== lote
def run_batch(photos: list[str], out_dir, funcao: str = "melhorar", progress: Callable | None = None,
              process: Callable | None = None, export: dict | None = None, **kw) -> dict:
    """Processa as fotos em sequência (uma falha não para o lote). ``process(foto, pasta, progresso) ->
    DesignResult`` (padrão: a função pedida, com ``kw``). As entregas (a 1ª versão de cada foto) são copiadas
    para ``out_dir`` como ``<foto>_<função>.jpg``; o trabalho de cada foto fica em ``out_dir/.trabalho``.
    ``export``: ``{"lado": 2048, "rotulo": bool}`` grava também ``out_dir/anuncio``. Eventos: ``progress`` (fração
    do lote) e ``item`` (index, total, photo, status, outputs, message)."""
    import shutil

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emit = progress or (lambda ev: None)
    n = len(photos)
    if n == 0:
        raise ValueError("Escolha ao menos uma foto.")
    if process is None:
        process = _default_process(funcao, kw)
    items, delivered = [], []
    t0 = time.perf_counter()
    for i, ph in enumerate(photos):
        stem = Path(ph).stem
        emit({"event": "item", "index": i, "total": n, "photo": str(ph), "status": "processando"})

        def sub(ev, i=i):
            if ev.get("event") == "progress":
                f = (i + float(ev.get("fraction") or 0.0)) / n
                emit({"event": "progress", "stage": "lote", "fraction": round(f, 4),
                      "message": f"Foto {i + 1} de {n}: {ev.get('message', '')}"})
            elif ev.get("event") == "log":
                emit(ev)

        try:
            res = process(ph, out / ".trabalho" / f"{i + 1:02d}_{stem}", sub)
            outs = []
            for j, o in enumerate(res.outputs):
                dest = out / (f"{stem}_{funcao}.jpg" if j == 0 else f"{stem}_{funcao}_{j + 1}.jpg")
                shutil.copyfile(o, dest)
                outs.append(str(dest))
            delivered += outs
            items.append({"photo": str(ph), "status": "pronto", "outputs": outs})
            emit({"event": "item", "index": i, "total": n, "photo": str(ph), "status": "pronto", "outputs": outs})
        except Exception as e:  # noqa: BLE001 - uma foto ruim não derruba o lote
            items.append({"photo": str(ph), "status": "erro", "message": str(e) or e.__class__.__name__})
            emit({"event": "item", "index": i, "total": n, "photo": str(ph), "status": "erro",
                  "message": str(e) or e.__class__.__name__})
    exported = []
    if export and delivered:
        emit({"event": "progress", "stage": "lote", "fraction": 0.99, "message": "Exportando para o anúncio"})
        lab = export.get("rotulo")
        exported = export_for_listing(delivered, out / "anuncio", int(export.get("lado", 2048)),
                                      default_label(funcao) if lab is None else bool(lab),
                                      log=lambda m: emit({"event": "log", "level": "warning", "message": m}))
    rep = {"funcao": funcao, "fotos": [str(p) for p in photos], "itens": items, "entregas": delivered,
           "anuncio": exported, "s": round(time.perf_counter() - t0, 2)}
    (out / "lote.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


def _default_process(funcao: str, kw: dict):
    from . import api, presets

    f = presets.function(funcao)

    def run(photo, out, progress):
        if f.key == "melhorar":
            from .enhance import enhance

            return enhance(photo, out, kw.get("opts"), progress=progress, device=kw.get("device", "auto"))
        if f.key in ("editar_area", "texto"):
            raise ValueError(f"'{f.label}' não funciona em lote.")
        return api.redesign(photo, f.key, kw.get("estilo") or ("moderno" if not f.event else None),
                            kw.get("ambiente", "sala"), n=int(kw.get("n", 1)), seed=kw.get("seed"),
                            prompt=kw.get("prompt", ""), evento=kw.get("evento"), out_dir=out,
                            quality=kw.get("quality", "final"), progress=progress, device=kw.get("device", "auto"))

    return run


__all__ = ["export_for_listing", "add_label", "default_label", "property_folder", "run_batch", "norm_long_side",
           "LABEL_TEXT"]
