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
from .render.layers import samples_by_layer
from .jobs import JobRequest, JobResult, ProgressEmitter
from .photo import load_photo
from .telemetry import (ENV_T_CLIQUE, ETAPAS, Stopwatch, VramMonitor, env_time, stages_sum, torch_peak_mb,
                        torch_reset_peak, value_total)

ANALYSIS_LONG_EDGE = 1536
# Lista JSON que substitui o comando do worker do Blender (depuração e testes automatizados).
ENV_BLENDER_WORKER = "MOBILIA_BLENDER_WORKER"
# "1" mantém os EXR dos quadros depois da composição (depuração e comando ``recompor``)
ENV_MANTER_EXR = "MOBILIA_MANTER_EXR"


def keep_exr() -> bool:
    return os.environ.get(ENV_MANTER_EXR, "").strip().lower() in ("1", "true", "sim", "yes")


def _quality(req: JobRequest, prof) -> dict:
    gpu = prof.torch_device == "cuda"
    q = req.quality
    if q == "auto":
        q = "final" if gpu else "previa"
    if q == "previa":
        s = 24 if not gpu else 48
        return {"name": "previa", "long_edge": min(960, prof.max_long_edge), "samples": s,
                "samples_layers": samples_by_layer(s, preview=True),
                "fps": 24, "texture_res": "1k", "motion_blur": True}
    # 24 fps (cinematográfico; o desfoque de movimento já está ligado): 20% menos quadros que 30
    return {"name": "final", "long_edge": prof.max_long_edge, "samples": prof.render_samples,
            "samples_layers": dict(prof.layer_samples), "fps": 24,
            "texture_res": "2k" if gpu else "1k", "motion_blur": True}


def _resize(rgb8: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = rgb8.shape[:2]
    if max(h, w) <= long_edge:
        return rgb8
    s = long_edge / max(h, w)
    return np.asarray(Image.fromarray(rgb8).resize((int(round(w * s)), int(round(h * s))), Image.LANCZOS))


def analyze_photo(photo_path: str | Path, out_dir: str | Path, device: str | None = None, emitter=None,
                  prof=None, sw: Stopwatch | None = None):
    """Etapa de análise isolada (também usada pela CLI de depuração).

    ``sw`` (opcional) recebe os tempos das subetapas em ``analise.*`` (ver ``telemetry.ETAPAS_ANALISE``).
    """
    from .analysis.debug_viz import geometry_overlay
    from .analysis.geometry import estimate_geometry
    from .analysis.room import build_room_model
    from .analysis.segmentation import segment

    sw = sw or Stopwatch()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    em = emitter or ProgressEmitter()
    with sw.measure("analise.redimensionar"):
        if prof is None:
            prof = choose_profile(detect_hardware())
        dev = device or prof.torch_device
        photo = load_photo(photo_path)
        rgb = _resize(photo.rgb8, ANALYSIS_LONG_EDGE)
    em.stage("analise", 0.05, "Estimando profundidade e câmera (MoGe-2)")
    t, tg = time.perf_counter(), {}
    geo = estimate_geometry(rgb, model_id=prof.geometry_model, device=dev, use_fp16=prof.use_fp16,
                            num_tokens=prof.geometry_tokens, timings=tg)
    total = _minus_download(sw, time.perf_counter() - t, tg)
    sw.add("analise.geometria_carga", min(total, tg.get("carga", 0.0)))
    sw.add("analise.geometria_inferencia", total - min(total, tg.get("carga", 0.0)))
    with sw.measure("analise.gravacao"):
        geo.save(out / "geometry.npz")
    em.stage("analise", 0.55, "Identificando piso, paredes, janelas e portas (OneFormer)")
    t, ts = time.perf_counter(), {}
    seg = segment(rgb, device=dev, use_fp16=prof.use_fp16, timings=ts)
    total = _minus_download(sw, time.perf_counter() - t, ts)
    sw.add("analise.segmentacao_carga", min(total, ts.get("carga", 0.0)))
    sw.add("analise.segmentacao_inferencia", total - min(total, ts.get("carga", 0.0)))
    with sw.measure("analise.gravacao"):
        seg.save(out / "segmentation.npz")
    em.stage("analise", 0.9, "Reconstruindo a sala em 3D")
    with sw.measure("analise.sala"):
        room, dbg = build_room_model(geo, seg, rgb)
    with sw.measure("analise.gravacao"):
        (out / "room.json").write_text(room.to_json(), encoding="utf-8")
        np.save(out / "grid.npy", dbg["grid"])
        Image.fromarray(rgb).save(out / "analysis_rgb.png")
        ov = geometry_overlay(rgb, room, seg)
        Image.fromarray(ov).save(out / "analise_overlay.jpg", quality=88)
    return rgb, geo, seg, room, dbg["grid"]


def _minus_download(sw: Stopwatch, total: float, timings: dict) -> float:
    """Tira da chamada do modelo o tempo de rede para baixar pesos que faltavam no cache: ele vai
    para ``tempos_s.download`` (custo de primeira execução), não para ``analise.*``."""
    dl = min(max(0.0, float(timings.get("download", 0.0) or 0.0)), total)
    if dl:
        sw.add("download", dl)
    return total - dl


def first_run_state() -> tuple[bool, Path | None, dict]:
    """(primeira execução?, arquivo de estado, situação do aquecimento do instalador). Primeira =
    nenhuma geração concluída desde a instalação, a última troca das dependências ou o último
    aquecimento (kernels e caches possivelmente frios)."""
    from . import bootstrap, runtime

    try:
        path = runtime.runtime_paths().state_file
        state = bootstrap.load_state(path)
        return bootstrap.is_first_run(state), path, bootstrap.warmup_status(state)
    except Exception:
        return False, None, {}


def run_job(req: JobRequest, em: ProgressEmitter, t_ref: float | None = None) -> JobResult:
    """Executa o trabalho inteiro e grava ``report.json`` com a telemetria completa.

    ``t_ref`` (``time.time()``) é o instante de lançamento do processo do motor, quando
    conhecido: entra em ``tempos_s.inicio_motor`` e em ``tempos_s.total``.
    """
    from .analysis.segmentation import SegmentationResult
    from .assets import polyhaven
    from .render.scene_spec import build_scene_spec

    wall0 = time.time()
    t_ref = wall0 if t_ref is None or t_ref > wall0 else float(t_ref)
    t_click = env_time(ENV_T_CLIQUE)
    t_click = t_ref if t_click is None or t_click > t_ref else t_click
    sw = Stopwatch()
    vram = VramMonitor().start()
    first, state_path, warm_state = first_run_state()
    # lançamento + imports + NVML + estado da instalação (o resto do início é medido abaixo)
    sw.set("inicio_motor", time.time() - t_ref)
    cfg = AppConfig()
    try:
        with sw.measure("inicio_motor"):
            em.stage("preparo", 0.0, "Detectando hardware")
            hw = detect_hardware()
            prof = choose_profile(hw)
            q = _quality(req, prof)
            em.log(f"Perfil: {prof.name} — {prof.description} | qualidade: {q['name']}")
            if first:
                em.log("Primeira execução depois da instalação: pode ser mais lenta (caches frios)."
                       + ("" if warm_state.get("ok") else " O aquecimento da GPU do instalador não foi "
                          "feito: os kernels de render serão preparados agora."))
        with sw.measure("carga_foto"):
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            work = Path(req.work_dir) if req.work_dir else cfg.jobs_dir / f"{stamp}_{Path(req.photo_path).stem}"
            work.mkdir(parents=True, exist_ok=True)
            (work / "request.json").write_text(json.dumps(asdict(req), ensure_ascii=False, indent=1),
                                               encoding="utf-8")
            # placa imutável na resolução do vídeo (redução única, se a foto for maior)
            original = load_photo(req.photo_path)
            plate = load_photo(req.photo_path, max_long_edge=q["long_edge"])
            plate8 = plate.rgb8
            Image.fromarray(plate8).save(work / "plate.png")
            em.stage("preparo", 1.0, f"Foto {original.width}×{original.height} → vídeo {plate.width}×{plate.height}")

        # ---------------------------------------------------------------- análise
        an_dir = work / "analise"
        em_an = ProgressEmitter(callback=lambda ev: em._emit(ev))
        with sw.measure("analise.geometria_carga"):   # na GPU cria o contexto CUDA (seria na carga)
            torch_reset_peak(prof.torch_device)
        with vram.track("analise"):
            rgb, geo, seg, room, grid = analyze_photo(req.photo_path, an_dir, device=prof.torch_device,
                                                      emitter=em_an, prof=prof, sw=sw)
            vram_torch = torch_peak_mb(prof.torch_device)
            with sw.measure("analise.gravacao"):
                em.preview(an_dir / "analise_overlay.jpg")
            q_room = room.quality
            from .analysis.depth_mesh import occluder_mesh
            from .analysis.sunlight import detect_sun

            with sw.measure("analise.oclusores"):
                occ = occluder_mesh(geo, seg, room)
            if len(occ[1]):
                em.log(f"Objetos existentes na foto viram oclusores 3D ({len(occ[1])} triângulos)")
            with sw.measure("analise.sol"):
                sun = detect_sun(room, seg, rgb)
                if sun is not None:
                    (an_dir / "sol.json").write_text(json.dumps(sun.to_dict(), indent=1), encoding="utf-8")
                    em.log(f"Sol direto detectado: elevação {sun.elevation_deg:.0f}°, {sun.patches} mancha(s) "
                           f"no piso (precisão {sun.precision:.0%})")
        em.log(f"Sala: câmera a {room.camera_height:.2f} m, pé-direito {room.ceiling_height:.2f} m, "
               f"{len(room.walls)} paredes, {len(room.openings)} aberturas, FOV {q_room.get('fov_x_deg', 0):.0f}°")

        # ---------------------------------------------------------------- layout
        with sw.measure("layout"):
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

        # ---------------------------------------------------------------- assets + cena
        dl_models = float(sw.get("download", 0.0))        # pesos dos modelos que faltavam (análise)
        t = time.perf_counter()
        net = polyhaven.reset_net_stats()
        em.stage("assets", 0.1, "Baixando/verificando modelos 3D (CC0)")
        seg_plate = SegmentationResult(
            np.asarray(Image.fromarray(seg.labels).resize((plate.width, plate.height), Image.NEAREST)),
            np.asarray(Image.fromarray(seg.groups).resize((plate.width, plate.height), Image.NEAREST)),
            seg.id2label, seg.model_id)
        hold_start, hold_end = cfg.video.hold_start_s, cfg.video.hold_end_s
        spec = build_scene_spec(room, layout, plate8, work / "plate.png", seg_plate, work, cfg.assets_dir,
                                fps=q["fps"], samples=q["samples"], samples_layers=q.get("samples_layers"),
                                device=prof.render_device,
                                texture_res=q["texture_res"], hold_start=hold_start, hold_end=hold_end,
                                motion_blur=q["motion_blur"], seed=req.seed,
                                sun=sun.to_dict() if sun is not None else None, occluders=occ)
        em.stage("assets", 1.0, f"{len(spec['items'])} modelos prontos")
        dl = min(float(net["s"]), time.perf_counter() - t)
        sw.add("download", dl)
        sw.add("ativos", time.perf_counter() - t - dl)
        download = {"s": round(dl + dl_models, 2), "ativos_s": round(dl, 2), "modelos_s": round(dl_models, 2),
                    "bytes": int(net["bytes"]), "arquivos": int(net["arquivos"]),
                    "consultas": int(net["consultas"])}
        if dl_models:
            em.log(f"Pesos de modelos de IA baixados durante o trabalho: {dl_models:.0f} s (fora do tempo de análise)")
        if net["arquivos"]:
            em.log(f"Modelos 3D baixados: {net['arquivos']} arquivo(s), {net['bytes'] / 1e6:.0f} MB em {dl:.0f} s")

        # ---------------------------------------------------------------- render (subprocesso Blender)
        # + composição em fluxo (PT5): cada quadro é composto, verificado e codificado assim que o
        # worker o anuncia, enquanto o Blender renderiza os seguintes; só a "cauda" fica depois
        em.stage("render", 0.0, "Montando a cena 3D e calibrando a luz pela foto")
        with sw.measure("composicao"):      # imports + FFmpeg aberto; a placa é estimada em paralelo
            composer = stream_composer(work, plate8, spec, Path(req.output_path), q["fps"], req.seed, em,
                                       gpu=prof.torch_device == "cuda", delete_exr=not keep_exr())
        try:
            with vram.track("render"):
                render_meta, proc = _run_render_worker(work / "scene.json", em, with_times=True,
                                                       on_frame=composer.push)
            _blender_stages(sw, render_meta.get("telemetria") or {}, proc)
            # ------------------------------------------------------------ cauda da composição + vídeo
            # cada EXR é apagado assim que o último quadro que o usa vai para o codificador (PT4)
            comp = composer.finish(render_meta)
        except BaseException:
            composer.abort()
            raise
        comp["composicao"]["exr_recortado"] = bool((render_meta.get("exr") or {}).get("recorte", False))
        sw.add("composicao", comp["tempos"]["composicao"])
        sw.add("codificacao", comp["tempos"]["codificacao"])
        sw.add("verificacao", comp["verify_seconds"])
        out_path = Path(req.output_path)

        with sw.measure("relatorio"):
            report = _build_report(req, stamp, original, plate, q, hw, prof, room, layout, sun, render_meta,
                                   comp, first, download, vram, vram_torch)
            if first and state_path is not None:
                try:
                    from . import bootstrap

                    bootstrap.record_run(state_path)
                except OSError:
                    pass
        now = time.time()
        tempos = sw.rounded()
        tempos["total"] = round(now - t_ref, 2)
        tempos["total_percebido"] = round(now - t_click, 2)
        tempos["soma_etapas"] = round(stages_sum(tempos), 2)
        report["tempos_s"] = tempos
        report["primeira_execucao_custos"] = {
            "download_s": tempos.get("download", 0.0),
            "blender_kernels_s": tempos.get("blender_kernels", 0.0),
            "kernels_cycles_s": (render_meta.get("telemetria") or {}).get("aquecimento", {}).get("kernels_s"),
            # a carga/compilação pode se repetir fora do aquecimento (OptiX recria o pipeline quando
            # a cena pede recursos novos, p. ex. desfoque de movimento): soma de todas as chamadas
            "kernels_cycles_total_s": _kernels_total(render_meta.get("telemetria") or {}),
            "modelos_carga_s": round(value_total(tempos["analise"].get("geometria_carga", 0.0))
                                     + value_total(tempos["analise"].get("segmentacao_carga", 0.0)), 2),
            # o que o instalador já pagou (PT2): kernels OptiX/CUDA e carga dos modelos
            "aquecimento_instalador": warm_state or None,
        }
        rp = work / "report.json"
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        flat = {k: value_total(v) for k, v in tempos.items()}
        em.log("Tempos: " + ", ".join(f"{k} {flat[k]:.1f} s" for k in ETAPAS if flat.get(k, 0) >= 0.05)
               + f" | total {tempos['total']:.1f} s")
        return JobResult(str(out_path), str(rp), str(work), flat)
    finally:
        vram.stop()


def _kernels_total(tele: dict) -> float:
    """Tempo de carga/compilação de kernels do Cycles em todas as chamadas de render do worker
    (aquecimento, calibração e quadros)."""
    calls = [tele.get("aquecimento") or {}] + list(tele.get("calibracao_renders") or []) \
        + list(tele.get("por_quadro") or [])
    return round(sum(float(c.get("kernels_s") or 0.0) for c in calls), 3)


def _blender_stages(sw: Stopwatch, tele: dict, proc: dict) -> None:
    """Distribui o tempo de parede do subprocesso do Blender pelas etapas ``blender_*``/``render_*``.

    ``blender_inicio`` vai do lançamento até o evento ``worker_ready`` (bpy importado, GPU
    configurada); as etapas seguintes vêm do cronômetro do próprio worker; ``blender_fim`` é o
    que sobra (frames.json, encerramento do processo, folgas entre etapas).
    """
    wt = tele.get("tempos_s") or {}
    wall = max(0.0, float(proc.get("exit", 0.0)) - float(proc.get("start", 0.0)))
    ready = proc.get("ready")
    if ready is not None:
        inicio = max(0.0, float(ready) - float(proc["start"]))
    else:
        inicio = float(wt.get("importacao_bpy", 0.0)) + float(wt.get("configuracao", 0.0))
    inicio = min(inicio, wall)
    sw.add("blender_inicio", inicio)
    used = inicio
    for key, wkey in (("blender_kernels", "kernels"), ("cena_montagem", "cena_montagem"),
                      ("calibracao", "calibracao"), ("render_final", "render_final"),
                      ("render_marcos", "render_marcos"), ("render_dinamico", "render_dinamico")):
        v = min(max(0.0, float(wt.get(wkey, 0.0))), max(0.0, wall - used))
        sw.add(key, v)
        used += v
    sw.add("blender_fim", max(0.0, wall - used))


def _build_report(req, stamp, original, plate, q, hw, prof, room, layout, sun, render_meta, comp, first,
                  download, vram, vram_torch) -> dict:
    from .telemetry import summarize_render

    tele = render_meta.get("telemetria") or {}
    W, H = int(render_meta.get("width", plate.width)), int(render_meta.get("height", plate.height))
    por_quadro = tele.get("por_quadro") or []
    summ = summarize_render(por_quadro, W, H)
    frames = render_meta.get("frames") or []
    quadros = tele.get("quadros") or {
        "total": len(frames), "renderizados": len({f.get("source") for f in frames if f.get("source")}),
        "reutilizados": sum(1 for f in frames if f.get("reused")),
        "so_foto": sum(1 for f in frames if f.get("source") is None)}
    calib = render_meta.get("calibration")
    gpu = hw.best_gpu
    calib_view = None
    if calib:
        calib_view = {**{k: v for k, v in calib.items() if k != "renders"},
                      "ganhos_canal": calib.get("color_ratio"),
                      "fracao_mascara": (round(calib["pixels"] / calib["pixels_total"], 4)
                                         if calib.get("pixels_total") else None),
                      "sol_ajustado": bool(calib.get("sun_strength", 0.0) > 0.0),
                      "curva_tom_ok": None,
                      "renders": calib.get("renders")}
    return {
        "versao": __version__,
        "data": stamp,
        "primeira_execucao": bool(first),
        "foto": {"arquivo": str(req.photo_path), "original": [original.width, original.height],
                 "placa_video": [plate.width, plate.height],
                 "megapixels": round(original.width * original.height / 1e6, 3),
                 "megapixels_placa": round(plate.width * plate.height / 1e6, 3),
                 "reduzida": [original.width, original.height] != [plate.width, plate.height]},
        "ambiente": req.room_type,
        "qualidade": q,
        "hardware": json.loads(hw.to_json()),
        "perfil": prof.__dict__,
        "gpu": {"nome": gpu.name if gpu else None, "driver": gpu.driver if gpu else None,
                "vram_total_mb": gpu.vram_total_mb if gpu else None,
                "vram_pico_mb": {"analise": vram.peak("analise"), "render": vram.peak("render"),
                                 "analise_torch": vram_torch,
                                 "render_cycles": summ["desempenho"]["mem_cycles_pico_mb"]
                                 if tele.get("dispositivo") not in (None, "CPU") else None},
                "nvml": vram.active},
        "sala": {"camera_altura_m": room.camera_height, "pe_direito_m": room.ceiling_height,
                 "paredes": len(room.walls), "aberturas": [o.kind for o in room.openings],
                 "qualidade": room.quality},
        "layout": json.loads(layout.to_json()),
        "calibracao_luz": calib_view,
        "lente": {"k1": None, "melhora_retidao": None, "aplicada": False},
        "sol": sun.to_dict() if sun is not None else None,
        "download": download,
        "render": {
            "dispositivo": tele.get("dispositivo", render_meta.get("device")),
            "gpus": tele.get("gpus"),
            "optix": bool(tele.get("optix", render_meta.get("device") == "OPTIX")),
            "denoiser": tele.get("denoiser"),
            "denoiser_dispositivo": tele.get("denoiser_dispositivo"),
            "blender": tele.get("blender"),
            "threads": tele.get("threads"),
            "amostras_config": tele.get("samples_config", q.get("samples")),
            "amostras_camadas": tele.get("samples_camadas", q.get("samples_layers")),
            "desempenho": summ["desempenho"],
            "orcamento": {"previsto_s": None, "spp_dinamico": None, "spp_marcos": None, "escala_cg": None,
                          "motivo": "controle de orçamento ainda não ativo"},
            "camadas": summ["camadas"],
            "cobertura_recorte_media": summ["cobertura_recorte_media"],
            "aquecimento": tele.get("aquecimento"),
            "tempos_worker_s": tele.get("tempos_s"),
            "por_quadro": por_quadro,
            "quadros": quadros,
            "tempos": render_meta.get("timings"),
            "itens_removidos": render_meta.get("removed_items") or [],
            # saída por quadro: passes gravados, EXR recortado no border e codec (PT4)
            "passes": tele.get("passes", render_meta.get("passes")),
            "exr": render_meta.get("exr") or tele.get("exr"),
        },
        "composicao": comp["composicao"],
        "preservacao": comp["preservacao"],
        "encode": comp["encode"],
        "licencas_assets": sorted({i.asset_id for i in layout.items}),
        "previas": comp["previas"],
        "plataforma": platform.platform(),
    }


def stream_composer(work: Path, plate8: np.ndarray, spec: dict, out_path: Path, fps: int, seed: int,
                    em: ProgressEmitter, gpu: bool = False, delete_exr: bool = False):
    """Compositor em fluxo para o render que vai começar (``StreamComposer``): a placa é estimada
    e o FFmpeg aberto já agora, durante o início do Blender. NVENC quando a GPU codifica."""
    from .composite.stream import StreamComposer
    from .video.encoder import choose_codec

    rs = spec.get("render") or {}
    return StreamComposer(work, plate8, out_path, fps, seed, em, codec=choose_codec(gpu), delete_exr=delete_exr,
                          sequential=bool(rs.get("incremental", False)), n_frames=rs.get("n_frames"),
                          streaming=True)


def compose_video(work: Path, plate8: np.ndarray, render_meta: dict, out_path: Path, fps: int,
                  seed: int, em: ProgressEmitter, codec: str = "libx264", delete_exr: bool = False,
                  workers: int | None = None) -> dict:
    """Compõe os passes renderizados sobre a foto, grava o MP4 e verifica o resultado.

    Usa só o que está em ``work/render`` (EXRs + frames.json): pode ser repetido sem
    renderizar de novo (comando ``recompor``) enquanto os EXR existirem. Com ``delete_exr``, cada
    EXR é apagado logo depois do último quadro que o usa. É o mesmo caminho da composição em
    fluxo do ``run_job`` (``composite/stream.py``), com todos os quadros entregues de uma vez:
    composição só no recorte de influência (bit a bit igual à de quadro inteiro), quadros
    reaproveitados vindos do cache, threads em paralelo, verificação independente por quadro e
    verificação do MP4 por *seek*. Mede por quadro a leitura dos EXR, a composição, a verificação
    e o envio ao codificador (``composicao.por_quadro``).
    """
    from .composite.stream import StreamComposer

    em.stage("composicao", 0.0, "Compondo sobre a foto original")
    frames_dir = work / "render" / "frames"
    frames = render_meta["frames"]
    # último quadro que usa cada EXR (quadros reaproveitados apontam para o EXR anterior)
    last_use = {fr["source"]: i + 1 for i, fr in enumerate(frames) if fr.get("source")}
    missing = [s for s in last_use if not (frames_dir / s).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} EXR de quadro não existem mais em {frames_dir} (ex.: {missing[0]}). "
            f"Os EXR são apagados depois da composição; para recompor, gere de novo com "
            f"{ENV_MANTER_EXR}=1.")
    composer = StreamComposer(work, plate8, out_path, fps, seed, em, codec=codec, delete_exr=delete_exr,
                              workers=workers, sequential=any(fr.get("mode") == "incremental" for fr in frames),
                              n_frames=len(frames), last_use=last_use)
    try:
        for fr in frames:
            composer.push(fr)
        res = composer.finish()
    except BaseException:
        composer.abort()
        raise
    res["composicao"]["exr_recortado"] = bool((render_meta.get("exr") or {}).get("recorte", False))
    return res


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
    report.update({"preservacao": comp["preservacao"], "encode": comp["encode"], "previas": comp["previas"],
                   "composicao": comp["composicao"]})
    report.setdefault("tempos_s", {})["recomposicao"] = round(comp["seconds"] + comp["verify_seconds"], 1)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return report


def blender_worker_prefix() -> list[str]:
    """Comando do worker do Blender sem argumentos (``MOBILIA_BLENDER_WORKER`` substitui)."""
    override = os.environ.get(ENV_BLENDER_WORKER)
    if override:
        prefix = json.loads(override)
        if not isinstance(prefix, list) or not prefix or not all(isinstance(p, str) for p in prefix):
            raise ValueError(f"{ENV_BLENDER_WORKER} deve ser uma lista JSON não vazia de strings.")
        return list(prefix)
    return [sys.executable, "-m", "mobilia.render.blender_worker"]


def blender_worker_env() -> dict:
    """Ambiente do worker do Blender: o pacote ``mobilia`` no ``PYTHONPATH`` e caches de kernels
    da NVIDIA grandes (``runtime.GPU_CACHE_ENV``)."""
    from .runtime import apply_gpu_cache_env

    env = dict(os.environ)
    pkg_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return apply_gpu_cache_env(env)


def _blender_worker_command(spec_path: Path) -> list[str]:
    return blender_worker_prefix() + [str(spec_path)]


def _run_render_worker(spec_path: Path, em: ProgressEmitter, with_times: bool = False, on_frame=None):
    """Roda o worker do Blender e devolve o ``frames.json``.

    Com ``with_times`` devolve também os instantes (``perf_counter`` deste processo) do
    lançamento, do evento ``worker_ready`` e do fim do processo, para a telemetria.
    ``on_frame(entrada)`` recebe cada quadro anunciado (evento ``frame_ready``, na ordem do vídeo,
    com o EXR já completo no disco). Se ele levantar um erro (composição falhou), o Blender é
    encerrado na hora e o erro sobe.
    """
    env = blender_worker_env()
    cmd = _blender_worker_command(spec_path)
    times: dict[str, float | None] = {"start": time.perf_counter(), "ready": None, "exit": None}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                            encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    err_tail: list[str] = []
    import threading

    def _drain_err():
        for line in proc.stderr:
            err_tail.append(line)
            if len(err_tail) > 200:
                del err_tail[:100]

    th = threading.Thread(target=_drain_err, daemon=True)
    th.start()
    try:
        meta_path = _read_worker_events(proc, em, times, on_frame)
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        raise
    proc.wait()
    times["exit"] = time.perf_counter()
    th.join(timeout=5)
    if proc.returncode != 0 or not meta_path:
        raise RuntimeError("O render 3D falhou.\n" + "".join(err_tail[-40:]))
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return (meta, times) if with_times else meta


def _read_worker_events(proc, em: ProgressEmitter, times: dict, on_frame) -> str | None:
    """Lê os eventos JSON do worker até ele fechar a saída; devolve o caminho do ``frames.json``."""
    meta_path = None
    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("event")
        if kind == "frame_ready":
            if on_frame is not None and isinstance(ev.get("entry"), dict):
                on_frame(ev["entry"])
        elif kind == "worker_ready":
            times["ready"] = time.perf_counter()
            den = ev.get("denoiser_dispositivo")
            em.log(f"Blender {ev.get('blender', '?')}: render em {ev.get('dispositivo')}"
                   + (f" ({', '.join(ev['gpus'])})" if ev.get("gpus") else "")
                   + (f", denoiser {ev.get('denoiser')} em {den}" if den else ""))
            if ev.get("denoiser_gpu_pedida") and den == "CPU" and ev.get("dispositivo") != "CPU":
                em.log("O denoiser não encontrou GPU compatível e roda na CPU (mais lento).", "warning")
        elif kind == "render_progress":
            tot = max(1, ev.get("estimated_total", 1))
            em.stage("render", ev["done"] / tot, f"Renderizando quadro {ev['done']}/{tot} ({ev['seconds']:.1f} s/quadro)")
        elif kind == "calibration":
            em.log(f"Luz calibrada pela foto (erro relativo {ev['rel_rms']:.0%})")
        elif kind == "log":
            em.log(ev.get("message", ""), ev.get("level", "info"))
        elif kind == "render_done":
            meta_path = ev["meta"]
    return meta_path
