"""Vídeo do modo B (PT-V3-7): composição + verificação (§3.5) + codificação, em fluxo.

Cada quadro mestre sai do :class:`compose.FrameCompositor` (torch/CUDA quando há, numpy senão), passa pelo
:class:`composite.verify.ModeBVerifier` (todas as regras, em 100% dos quadros) e vai para o FFmpeg (NVENC quando
a GPU codifica, x264 senão; o mesmo ``video/encoder.py`` do modo A). Opcionalmente grava também um MKV sem perdas
para auditoria. Qualquer falha da verificação ou do FFmpeg aborta: os arquivos parciais são apagados e NENHUM MP4
fica no disco. Depois do ``close``, o 1º e o último quadro do MP4 são decodificados e comparados (PSNR) com a placa
e a foto; o MKV, quando pedido, é decodificado inteiro e cada quadro tem de bater byte a byte (hash) com o mestre.
"""
from __future__ import annotations

import hashlib
import queue
import threading
import time
from pathlib import Path

import numpy as np

from ..composite.verify import ModeBVerificationError, ModeBVerifier
from ..video import encoder as E
from . import compose as K

QUEUE_FRAMES = 4          # quadros em trânsito entre a composição e os FFmpeg (memória: 4 × 8 MB a 1920×1440)
# piso de PSNR do 1º/último quadro do MP4 decodificado contra placa/foto. x264/NVENC normais dão 45–50 dB; abaixo
# disto o codificador estragou o vídeo (matriz de cor errada, quadro verde/preto, deslocamento) e o MP4 não sai.
MIN_PSNR_MP4 = 30.0
PROGRESS_EVERY = 12


class _Sink(threading.Thread):
    """Envia os quadros para os FFmpeg numa thread (a escrita no pipe espera o codificador; a composição segue)."""

    def __init__(self, writers):
        super().__init__(name="modo-b-encode", daemon=True)
        self.writers = writers
        self.q: queue.Queue = queue.Queue(maxsize=QUEUE_FRAMES)
        self.error: BaseException | None = None
        self.busy_s = 0.0

    def run(self):
        while True:
            fr = self.q.get()
            if fr is None:
                return
            if self.error is not None:
                continue
            t0 = time.perf_counter()
            try:
                for w in self.writers:
                    w.write(fr)
            except BaseException as e:            # FFmpeg morreu: a composição para no próximo quadro
                self.error = e
            self.busy_s += time.perf_counter() - t0

    def put(self, fr):
        if self.error is not None:
            raise RuntimeError(f"codificação falhou: {self.error}")
        self.q.put(fr)


def render_mode_b(stack, ch, photo: np.ndarray, removal_mask: np.ndarray, out_mp4: str | Path | None,
                  audit_mkv: str | Path | None = None, backend: str | None = None, gpu: bool | None = None,
                  settings: E.EncodeSettings | None = None, choreo_stats: dict | None = None,
                  check_mp4: bool = True, log_dir: str | Path | None = None, progress=None) -> dict:
    """Compõe, verifica e codifica o vídeo do modo B. Devolve o relatório (tempos, codificador, verificação).

    ``out_mp4`` None = só compõe e verifica (medição). ``gpu`` None = usa NVENC se o compositor está em CUDA.
    ``choreo_stats`` (de ``choreo.verify``) entra como regra 5. Levanta :class:`ModeBVerificationError` ou
    ``RuntimeError`` sem deixar MP4/MKV no disco. ``progress(feitos, total)`` (opcional) é chamado a cada
    :data:`PROGRESS_EVERY` quadros (barra da interface)."""
    H, W = stack.photo_shape
    if (out_mp4 is not None or audit_mkv is not None) and (H % 2 or W % 2):
        raise ValueError(f"o vídeo precisa de largura e altura pares (camadas em {W}×{H}); "
                         "recorte a foto na resolução do vídeo antes de montar as camadas")
    t_all = time.perf_counter()
    comp = K.FrameCompositor(stack, backend)
    t_up = time.perf_counter() - t_all
    use_gpu = (comp.device == "cuda") if gpu is None else bool(gpu)
    s = settings or E.EncodeSettings(codec=E.choose_codec(use_gpu))
    ver = ModeBVerifier(photo, stack, removal_mask, ch.n_frames)
    writers = []
    log = (lambda name: None if log_dir is None else Path(log_dir) / name)
    sink = None
    hashes: list[bytes] = []
    t_comp = t_ver = 0.0
    n = 0
    try:
        # dentro do try: se o 2º FFmpeg não abre, o 1º é abortado (sem processo órfão nem arquivo parcial)
        if out_mp4 is not None:
            writers.append(E.VideoWriter(out_mp4, W, H, ch.fps, s, log_path=log("encode_modo_b.log")))
        if audit_mkv is not None:
            writers.append(E.LosslessWriter(audit_mkv, W, H, ch.fps, log_path=log("encode_auditoria.log")))
        sink = _Sink(writers) if writers else None
        if sink:
            sink.start()
        it = K.master_frames(stack, ch, photo, comp, with_placements=True)
        while True:
            t0 = time.perf_counter()
            try:
                fr, pl = next(it)
            except StopIteration:
                break
            t1 = time.perf_counter()
            ver.check(fr, pl)
            if audit_mkv is not None:
                hashes.append(hashlib.blake2b(np.ascontiguousarray(fr).data, digest_size=16).digest())
            t2 = time.perf_counter()
            t_comp += t1 - t0
            t_ver += t2 - t1
            if sink:
                sink.put(fr)
            n += 1
            if progress is not None and n % PROGRESS_EVERY == 0:
                progress(n, ch.n_frames)
        rep = ver.finish(choreo_stats)
        if sink:
            sink.q.put(None)
            sink.join()
            if sink.error is not None:
                raise RuntimeError(f"codificação falhou: {sink.error}")
            t_close = time.perf_counter()
            for w in writers:
                w.close()
            t_close = time.perf_counter() - t_close
        else:
            t_close = 0.0
    except BaseException:
        if sink and sink.is_alive():
            sink.error = sink.error or RuntimeError("abortado")
            for w in writers:
                w.abort()
            sink.q.put(None)
            sink.join(timeout=10)
        for w in writers:
            w.abort()
        # o close do MP4 pode ter dado certo e o do MKV falhado: o MP4 já renomeado também sai
        _unlink(out_mp4, audit_mkv)
        raise
    t_total = time.perf_counter() - t_all
    out = {"video_modo_b": {
        "backend": comp.backend, "device": comp.device, "codificador": s.codec if out_mp4 is not None else None,
        "quadros": n, "fps": ch.fps, "resolucao": [W, H],
        "tempo_s": {"preparo_compositor": round(t_up, 3), "composicao": round(t_comp, 3), "verificacao": round(t_ver, 3),
                    "pipe_ffmpeg": round(sink.busy_s, 3) if sink else 0.0, "fechar_ffmpeg": round(t_close, 3),
                    "total": round(t_total, 3)}}, **rep}
    try:
        if out_mp4 is not None and check_mp4:
            got = dict(E.seek_frames(out_mp4, W, H, [1, n], ch.fps))
            if 1 not in got or n not in got:
                raise RuntimeError("o MP4 não devolveu o 1º/último quadro na decodificação")
            p0, p1 = E.psnr(got[1], stack.plate), E.psnr(got[n], photo)
            out["video_modo_b"]["psnr_mp4"] = {"quadro0_vs_placa": round(float(min(p0, 99.0)), 2),
                                               "ultimo_vs_foto": round(float(min(p1, 99.0)), 2)}
            if min(p0, p1) < MIN_PSNR_MP4:
                raise RuntimeError(f"MP4 decodificado longe dos quadros mestres (PSNR {min(p0, p1):.1f} dB < "
                                   f"{MIN_PSNR_MP4:.0f} dB): o codificador estragou o vídeo")
        if audit_mkv is not None:
            k = 0
            for idx, fr in E.decode_lossless(audit_mkv, W, H):
                if idx > len(hashes) or hashlib.blake2b(fr.data, digest_size=16).digest() != hashes[idx - 1]:
                    raise ModeBVerificationError(f"MKV de auditoria: quadro {idx} não volta byte a byte")
                k = idx
            if k != len(hashes):
                raise ModeBVerificationError(f"MKV de auditoria com {k} quadros, esperados {len(hashes)}")
            out["video_modo_b"]["auditoria_mkv"] = {"quadros_identicos": k, "codec": "libx264rgb qp0"}
    except BaseException:
        _unlink(out_mp4, audit_mkv)
        raise
    return out


def _unlink(*paths) -> None:
    for p in paths:
        if p is not None:
            Path(p).unlink(missing_ok=True)


__all__ = ["render_mode_b", "ModeBVerificationError"]
