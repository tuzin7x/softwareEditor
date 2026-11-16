"""Orquestração do trabalho: foto -> análise -> layout -> render 3D -> composição exata -> MP4.

Cada etapa salva seus artefatos na pasta do trabalho (permite depurar, retomar e
processar em lote). A VRAM é liberada entre etapas: os modelos de IA rodam um por
vez neste processo e o Blender roda num subprocesso próprio.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .config import AppConfig
from .hardware import choose_profile, detect_hardware
from .jobs import JobRequest, JobResult, ProgressEmitter
from .photo import load_photo

ANALYSIS_LONG_EDGE = 1536


def _quality(req: JobRequest, prof) -> dict:
    gpu = prof.torch_device == "cuda"
    q = req.quality
    if q == "auto":
        q = "final" if gpu else "previa"
    if q == "previa":
        return {"name": "previa", "long_edge": min(960, prof.max_long_edge), "samples": 24 if not gpu else 48,
                "fps": 24, "texture_res": "1k", "motion_blur": True}
    return {"name": "final", "long_edge": prof.max_long_edge, "samples": prof.render_samples, "fps": 30,
            "texture_res": "2k" if gpu else "1k", "motion_blur": True}


def _resize(rgb8: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = rgb8.shape[:2]
    if max(h, w) <= long_edge:
        return rgb8
    s = long_edge / max(h, w)
    return np.asarray(Image.fromarray(rgb8).resize((int(round(w * s)), int(round(h * s))), Image.LANCZOS))


def analyze_photo(photo_path: str | Path, out_dir: str | Path, device: str | None = None, emitter=None):
    """Etapa de análise isolada (também usada pela CLI de depuração)."""
    from .analysis.debug_viz import geometry_overlay
    from .analysis.geometry import estimate_geometry
    from .analysis.room import build_room_model
    from .analysis.segmentation import segment

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    em = emitter or ProgressEmitter()
    hw = detect_hardware()
    prof = choose_profile(hw)
    dev = device or prof.torch_device
    photo = load_photo(photo_path)
    rgb = _resize(photo.rgb8, ANALYSIS_LONG_EDGE)
    em.stage("analise", 0.05, "Estimando profundidade e câmera (MoGe-2)")
    geo = estimate_geometry(rgb, model_id=prof.geometry_model, device=dev, use_fp16=prof.use_fp16,
                            num_tokens=prof.geometry_tokens)
    geo.save(out / "geometry.npz")
    em.stage("analise", 0.55, "Identificando piso, paredes, janelas e portas (OneFormer)")
    seg = segment(rgb, device=dev, use_fp16=prof.use_fp16)
    seg.save(out / "segmentation.npz")
    em.stage("analise", 0.9, "Reconstruindo a sala em 3D")
    room, dbg = build_room_model(geo, seg, rgb)
    (out / "room.json").write_text(room.to_json(), encoding="utf-8")
    np.save(out / "grid.npy", dbg["grid"])
    Image.fromarray(rgb).save(out / "analysis_rgb.png")
    ov = geometry_overlay(rgb, room, seg)
    Image.fromarray(ov).save(out / "analise_overlay.jpg", quality=88)
    return rgb, geo, seg, room, dbg["grid"]


def run_job(req: JobRequest, em: ProgressEmitter) -> JobResult:
    from .analysis.segmentation import SegmentationResult
    from .render.scene_spec import build_scene_spec

    cfg = AppConfig()
    t0 = time.time()
    timings: dict[str, float] = {}
    em.stage("preparo", 0.0, "Detectando hardware")
    hw = detect_hardware()
    prof = choose_profile(hw)
    q = _quality(req, prof)
    em.log(f"Perfil: {prof.name} — {prof.description} | qualidade: {q['name']}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    work = Path(req.work_dir) if req.work_dir else cfg.jobs_dir / f"{stamp}_{Path(req.photo_path).stem}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "request.json").write_text(json.dumps(asdict(req), ensure_ascii=False, indent=1), encoding="utf-8")

    # placa imutável na resolução do vídeo (redução única, se a foto for maior)
    original = load_photo(req.photo_path)
    plate = load_photo(req.photo_path, max_long_edge=q["long_edge"])
    plate8 = plate.rgb8
    Image.fromarray(plate8).save(work / "plate.png")
    em.stage("preparo", 1.0, f"Foto {original.width}×{original.height} → vídeo {plate.width}×{plate.height}")
    timings["preparo"] = time.time() - t0

    # ---------------------------------------------------------------- análise
    t = time.time()
    an_dir = work / "analise"

    def sub_em_stage(stage, frac, msg=None):
        em.stage(stage, frac, msg)

    em_an = ProgressEmitter(callback=lambda ev: em._emit(ev))
    rgb, geo, seg, room, grid = analyze_photo(req.photo_path, an_dir, device=prof.torch_device, emitter=em_an)
    em.preview(an_dir / "analise_overlay.jpg")
    q_room = room.quality
    from .analysis.depth_mesh import occluder_mesh
    from .analysis.sunlight import detect_sun

    occ = occluder_mesh(geo, seg, room)
    if len(occ[1]):
        em.log(f"Objetos existentes na foto viram oclusores 3D ({len(occ[1])} triângulos)")
    sun = detect_sun(room, seg, rgb)
    if sun is not None:
        (an_dir / "sol.json").write_text(json.dumps(sun.to_dict(), indent=1), encoding="utf-8")
        em.log(f"Sol direto detectado: elevação {sun.elevation_deg:.0f}°, {sun.patches} mancha(s) no piso "
               f"(precisão {sun.precision:.0%})")
    em.log(f"Sala: câmera a {room.camera_height:.2f} m, pé-direito {room.ceiling_height:.2f} m, "
           f"{len(room.walls)} paredes, {len(room.openings)} aberturas, FOV {q_room.get('fov_x_deg', 0):.0f}°")
    timings["analise"] = time.time() - t

    # ---------------------------------------------------------------- layout
    t = time.time()
    em.stage("layout", 0.1, "Escolhendo móveis e posições ideais")
    from .layout.solver import plan_layout

    layout = plan_layout(room, grid, req.room_type, rgb, seed=req.seed, seg=seg)
    (work / "layout.json").write_text(layout.to_json(), encoding="utf-8")
    try:
        from .layout.debug import layout_overlay

        ov = layout_overlay(rgb, room, layout)
        Image.fromarray(ov).save(work / "layout_overlay.jpg", quality=88)
        em.preview(work / "layout_overlay.jpg")
    except Exception as e:  # visualização é opcional
        em.log(f"Sem prévia do layout: {e}", "warning")
    em.log(f"Layout: {len(layout.items)} itens, estilo {layout.style}")
    timings["layout"] = time.time() - t

    # ---------------------------------------------------------------- assets + cena
    t = time.time()
    em.stage("assets", 0.1, "Baixando/verificando modelos 3D (CC0)")
    seg_plate = SegmentationResult(
        np.asarray(Image.fromarray(seg.labels).resize((plate.width, plate.height), Image.NEAREST)),
        np.asarray(Image.fromarray(seg.groups).resize((plate.width, plate.height), Image.NEAREST)),
        seg.id2label, seg.model_id)
    hold_start, hold_end = cfg.video.hold_start_s, cfg.video.hold_end_s
    spec = build_scene_spec(room, layout, plate8, work / "plate.png", seg_plate, work, cfg.assets_dir,
                            fps=q["fps"], samples=q["samples"], device=prof.render_device,
                            texture_res=q["texture_res"], hold_start=hold_start, hold_end=hold_end,
                            motion_blur=q["motion_blur"], seed=req.seed,
                            sun=sun.to_dict() if sun is not None else None, occluders=occ)
    em.stage("assets", 1.0, f"{len(spec['items'])} modelos prontos")
    timings["assets"] = time.time() - t

    # ---------------------------------------------------------------- render (subprocesso Blender)
    t = time.time()
    em.stage("render", 0.0, "Montando a cena 3D e calibrando a luz pela foto")
    render_meta = _run_render_worker(work / "scene.json", em)
    timings["render"] = time.time() - t

    # ---------------------------------------------------------------- composição + vídeo
    t = time.time()
    comp = compose_video(work, plate8, render_meta, Path(req.output_path), q["fps"], req.seed, em)
    timings["composicao_video"] = comp["seconds"]
    timings["verificacao"] = comp["verify_seconds"]
    out_path = Path(req.output_path)

    report = {
        "versao": __version__,
        "data": stamp,
        "foto": {"arquivo": str(req.photo_path), "original": [original.width, original.height],
                 "placa_video": [plate.width, plate.height],
                 "reduzida": [original.width, original.height] != [plate.width, plate.height]},
        "ambiente": req.room_type,
        "qualidade": q,
        "hardware": json.loads(hw.to_json()),
        "perfil": prof.__dict__,
        "sala": {"camera_altura_m": room.camera_height, "pe_direito_m": room.ceiling_height,
                 "paredes": len(room.walls), "aberturas": [o.kind for o in room.openings],
                 "qualidade": room.quality},
        "layout": json.loads(layout.to_json()),
        "calibracao_luz": render_meta.get("calibration"),
        "sol": sun.to_dict() if sun is not None else None,
        "render": {"dispositivo": render_meta.get("device"), "tempos": render_meta.get("timings"),
                   "itens_removidos": render_meta.get("removed_items") or []},
        "preservacao": comp["preservacao"],
        "encode": comp["encode"],
        "tempos_s": {k: round(v, 1) for k, v in timings.items()},
        "licencas_assets": sorted({i.asset_id for i in layout.items}),
        "previas": comp["previas"],
        "plataforma": platform.platform(),
    }
    rp = work / "report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return JobResult(str(out_path), str(rp), str(work), timings)


def compose_video(work: Path, plate8: np.ndarray, render_meta: dict, out_path: Path, fps: int,
                  seed: int, em: ProgressEmitter, codec: str = "libx264") -> dict:
    """Compõe os passes renderizados sobre a foto, grava o MP4 e verifica o resultado.

    Usa só o que está em ``work/render`` (EXRs + frames.json): pode ser repetido sem
    renderizar de novo (comando ``recompor``).
    """
    from .composite.compositor import Compositor, LayerAccumulator, load_render_passes
    from .video.encoder import EncodeSettings, VideoWriter, decode_frames, psnr

    t = time.time()
    em.stage("composicao", 0.0, "Compondo sobre a foto original")
    H, W = plate8.shape[:2]
    comp = Compositor(plate8, seed=seed)
    out_path = Path(out_path)
    enc = VideoWriter(out_path, W, H, fps, EncodeSettings(codec=codec), log_path=work / "encode.log")
    frames = render_meta["frames"]
    stats = []
    previews = []
    keyframes_dir = work / "quadros"
    keyframes_dir.mkdir(exist_ok=True)
    n = len(frames)
    sample = sorted({1, max(1, n // 2), n})
    masters: dict[int, np.ndarray] = {}
    acc = LayerAccumulator(H, W, feather_px=max(6, W // 120),
                           shadow_feather_px=max(16, int(0.04 * float(np.hypot(W, H)))))
    loaded_src = None
    for i, fr in enumerate(frames):
        src = fr.get("source")
        if src is None:
            acc.reset()
            out8, st = comp.composite(fr["frame"], None, None, None)
        else:
            if src != loaded_src and not fr.get("reused"):
                rgba, sh = load_render_passes(work / "render" / "frames" / src)
                acc.update(rgba, sh, fr.get("border"), fr.get("mode", "full"))
                loaded_src = src
            out8, st = comp.composite(fr["frame"], acc.rgba, acc.shadow, None)
        if not st.preserved_exact:
            enc.close()
            raise RuntimeError(f"Violação da preservação de pixels no quadro {fr['frame']}")
        stats.append(st)
        enc.write(out8)
        if fr["frame"] in sample:
            masters[fr["frame"]] = out8
        if i % max(1, n // 8) == 0 or i == n - 1:
            pth = keyframes_dir / f"quadro_{fr['frame']:05d}.jpg"
            Image.fromarray(out8).save(pth, quality=92)
            previews.append(str(pth))
            em.preview(pth)
        if i % 10 == 0:
            em.stage("composicao", i / n, f"Quadro {i + 1}/{n}")
    em.stage("video", 0.5, "Finalizando o MP4")
    enc.close()
    seconds = time.time() - t

    # ---- verificação do encode (decodifica e compara com os quadros mestres)
    t = time.time()
    check = []
    for idx, dec in decode_frames(out_path, W, H, sample):
        ref = masters.get(idx)
        if ref is None:
            continue
        untouched = (ref == plate8).all(-1)
        check.append({"frame": idx, "psnr_total_db": round(psnr(dec, ref), 2),
                      "psnr_fundo_db": round(psnr(dec, plate8, untouched), 2)})
    em.stage("video", 1.0, "Vídeo pronto")
    changed = max((s.changed_pixels for s in stats), default=0)
    return {
        "seconds": seconds,
        "verify_seconds": time.time() - t,
        "previas": previews,
        "preservacao": {
            "quadros": len(stats),
            "todos_bit_exatos_fora_dos_objetos": all(s.preserved_exact for s in stats),
            "max_pixels_alterados_em_um_quadro": changed,
            "fracao_max_alterada": round(changed / plate8[..., 0].size, 4),
            "quadros_identicos_a_foto": sum(1 for s in stats if s.changed_pixels == 0),
        },
        "encode": {"codec": codec, "verificacao": check, "arquivo": str(out_path),
                   "bytes": out_path.stat().st_size},
    }


def recompose(work_dir: str | Path, out_path: str | Path | None, em: ProgressEmitter) -> dict:
    """Refaz composição + MP4 de um trabalho já renderizado (sem Blender, sem IA)."""
    work = Path(work_dir)
    plate8 = np.asarray(Image.open(work / "plate.png").convert("RGB"))
    render_meta = json.loads((work / "render" / "frames.json").read_text(encoding="utf-8"))
    req = json.loads((work / "request.json").read_text(encoding="utf-8"))
    rp = work / "report.json"
    report = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    fps = int(report.get("qualidade", {}).get("fps") or
              json.loads((work / "scene.json").read_text(encoding="utf-8"))["render"]["fps"])
    out = Path(out_path) if out_path else Path(req["output_path"])
    comp = compose_video(work, plate8, render_meta, out, fps, int(req.get("seed", 7)), em)
    report.update({"preservacao": comp["preservacao"], "encode": comp["encode"], "previas": comp["previas"]})
    report.setdefault("tempos_s", {})["recomposicao"] = round(comp["seconds"] + comp["verify_seconds"], 1)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return report



def _run_render_worker(spec_path: Path, em: ProgressEmitter) -> dict:
    env = dict(os.environ)
    pkg_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-m", "mobilia.render.blender_worker", str(spec_path)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                            encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    meta_path = None
    err_tail: list[str] = []
    import threading

    def _drain_err():
        for line in proc.stderr:
            err_tail.append(line)
            if len(err_tail) > 200:
                del err_tail[:100]

    th = threading.Thread(target=_drain_err, daemon=True)
    th.start()
    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("event")
        if kind == "render_progress":
            tot = max(1, ev.get("estimated_total", 1))
            em.stage("render", ev["done"] / tot, f"Renderizando quadro {ev['done']}/{tot} ({ev['seconds']:.1f} s/quadro)")
        elif kind == "calibration":
            em.log(f"Luz calibrada pela foto (erro relativo {ev['rel_rms']:.0%})")
        elif kind == "log":
            em.log(ev.get("message", ""), ev.get("level", "info"))
        elif kind == "render_done":
            meta_path = ev["meta"]
    proc.wait()
    th.join(timeout=5)
    if proc.returncode != 0 or not meta_path:
        raise RuntimeError("O render 3D falhou.\n" + "".join(err_tail[-40:]))
    return json.loads(Path(meta_path).read_text(encoding="utf-8"))
