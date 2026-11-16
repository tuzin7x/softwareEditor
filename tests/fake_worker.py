"""DUBLÊ DE TESTE (tests/test_ui.py e tests/test_pipeline_e2e.py). Não faz parte do produto.

Modos:
- ``hardware``: responde como ``mobilia.cli hardware --json``;
- ``run ...``: imita o protocolo JSON do motor (``mobilia.cli run --json``), grava um MP4
  minúsculo de verdade com o FFmpeg e, com ``--work-dir``, os arquivos de um trabalho
  (report.json, frames.json, registros e uma placa derivada da foto);
- ``blender <scene.json>``: imita o worker do Blender (``mobilia.render.blender_worker``): emite
  ``worker_ready``/``render_progress``/``render_done`` e grava EXRs (Combined + Shadow Catcher)
  de um quadrado que entra em cena e para, com a telemetria por quadro do worker real.
"""
import json
import subprocess
import sys
import time
from pathlib import Path


def _emit(**ev):
    print(json.dumps(ev, ensure_ascii=False), flush=True)


def _write_exr(path: Path, rgba, shadow) -> None:
    import OpenEXR

    ch = {f"ViewLayer.Combined.{c}": rgba[..., i].copy() for i, c in enumerate("RGBA")}
    ch.update({f"ViewLayer.Shadow Catcher.{c}": shadow[..., i].copy() for i, c in enumerate("RGB")})
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, ch) as f:
        f.write(str(path))


def blender(spec_path: str) -> int:
    import numpy as np

    t0 = time.perf_counter()
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    W, H = int(spec["width"]), int(spec["height"])
    n = min(int(spec["render"]["n_frames"]), 18)
    out = Path(spec["render"]["out_dir"])
    (out / "frames").mkdir(parents=True, exist_ok=True)
    time.sleep(0.05)
    tempos = {"importacao_bpy": 0.05, "configuracao": round(time.perf_counter() - t0, 3)}
    _emit(event="worker_ready", dispositivo="CPU", optix=False, gpus=[], threads=4, blender="dublê",
          denoiser="OPENIMAGEDENOISE", denoiser_dispositivo="CPU", denoiser_gpu_pedida=False)
    for key, s in (("kernels", 0.05), ("cena_montagem", 0.1), ("calibracao", 0.1)):
        time.sleep(s)
        tempos[key] = s
    _emit(event="calibration", rel_rms=0.1)
    frames, por_quadro = [], []
    layer_s = {"final": 0.0, "marco": 0.0, "dinamico": 0.0}
    side = max(8, W // 6)
    stop_at = 12
    last = None
    for f in range(1, n + 1):
        if f <= 4:
            frames.append({"frame": f, "source": None, "border": None})
            continue
        x = min(f, stop_at) * (W - side) // (2 * stop_at)
        key = (x,)
        border = [max(0, x - 6), H // 3 - 6, min(W, x + side + 6), min(H, H // 3 + side + 6)]
        if key == last:
            frames.append({"frame": f, "source": frames[-1]["source"], "border": frames[-1]["border"],
                           "reused": True})
            continue
        ti = time.perf_counter()
        rgba = np.zeros((H, W, 4), np.float32)
        rgba[H // 3:H // 3 + side, x:x + side] = (0.3, 0.2, 0.1, 1.0)
        shadow = np.ones((H, W, 3), np.float32)
        shadow[H // 3 + side:H // 3 + side + 3, x:x + side] = 0.7
        name = f"f_{f:05d}.exr"
        _write_exr(out / "frames" / name, rgba, shadow)
        time.sleep(0.01)
        wall = time.perf_counter() - ti
        camada = "dinamico" if f < stop_at else "final"
        layer_s[camada] += wall
        area = (border[2] - border[0]) * (border[3] - border[1])
        frames.append({"frame": f, "source": name, "border": border, "mode": "full"})
        por_quadro.append({"quadro": f, "camada": camada, "itens": 1, "pecas_em_movimento": int(f < stop_at),
                           "area_px": area, "cobertura": round(area / (W * H), 4), "spp": 8, "amostras": 8,
                           "amostragem_s": round(wall * 0.6, 4), "denoise_s": round(wall * 0.2, 4),
                           "gravacao_s": round(wall * 0.1, 4), "sincronizacao_s": 0.0, "kernels_s": 0.0,
                           "preparo_s": 0.0, "render_s": round(wall, 4), "parede_s": round(wall, 4),
                           "mem_mb": 12.0})
        last = key
        _emit(event="render_progress", done=len(por_quadro), frame=f, n_frames=n, camada=camada,
              estimated_total=stop_at - 4, seconds=round(wall, 3))
    tempos.update(render_final=layer_s["final"], render_marcos=0.0, render_dinamico=layer_s["dinamico"],
                  laco_outros=0.0)
    meta = {"frames": frames, "calibration": {"window_power_per_m2": 1.0, "ambient_strength": 0.3,
                                              "sun_strength": 0.0, "color_ratio": [1.0, 1.0, 1.0],
                                              "rel_rms": 0.1, "pixels": 100, "pixels_total": 400},
            "device": "CPU", "width": W, "height": H, "removed_items": [],
            "timings": {"total_s": round(time.perf_counter() - t0, 1), "rendered_frames": len(por_quadro)},
            "telemetria": {"dispositivo": "CPU", "optix": False, "gpus": [], "threads": 4, "blender": "dublê",
                           "denoiser": "OPENIMAGEDENOISE", "denoiser_dispositivo": "CPU",
                           "tempos_s": tempos, "aquecimento": {"parede_s": 0.05, "kernels_s": 0.01},
                           "calibracao_renders": [], "por_quadro": por_quadro,
                           "quadros": {"total": len(frames), "renderizados": len(por_quadro),
                                       "reutilizados": sum(1 for fr in frames if fr.get("reused")),
                                       "so_foto": sum(1 for fr in frames if fr.get("source") is None)},
                           "samples_config": int(spec["render"]["samples"])}}
    (out / "frames.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    time.sleep(0.02)
    _emit(event="render_done", rendered=len(por_quadro), meta=str(out / "frames.json"))
    return 0


def _write_job_files(work: Path, photo: str) -> Path:
    """Arquivos de um trabalho concluído, como o motor real deixa na pasta."""
    from PIL import Image

    (work / "render").mkdir(parents=True, exist_ok=True)
    (work / "request.json").write_text(json.dumps({"photo_path": photo}), encoding="utf-8")
    Image.open(photo).convert("RGB").resize((64, 48)).save(work / "plate.png")
    (work / "quadros").mkdir(exist_ok=True)
    Image.open(photo).convert("RGB").resize((64, 48)).save(work / "quadros" / "quadro_00001.jpg")
    (work / "render" / "frames.json").write_text(json.dumps({"frames": [], "telemetria": {}}), encoding="utf-8")
    (work / "encode.log").write_text("ok\n", encoding="utf-8")
    rep = {"versao": "teste", "primeira_execucao": True,
           "tempos_s": {"inicio_motor": 1.0, "carga_foto": 0.2, "analise": {"geometria_carga": 2.0,
                                                                             "geometria_inferencia": 3.0},
                        "layout": 1.5, "ativos": 0.3, "download": 0.0, "blender_inicio": 2.0,
                        "blender_kernels": 0.5, "cena_montagem": 1.0, "calibracao": 1.0, "render_final": 4.0,
                        "render_marcos": 0.0, "render_dinamico": 20.0, "blender_fim": 0.2, "composicao": 3.0,
                        "codificacao": 1.0, "verificacao": 0.5, "relatorio": 0.0, "total": 42.0,
                        "total_percebido": 42.5}}
    rp = work / "report.json"
    rp.write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    return rp


def main(argv):
    if argv and argv[0] == "hardware":
        print(json.dumps({"gpus": [{"name": "NVIDIA GeForce RTX 4070", "vram_total_mb": 12282}],
                          "profile": {"name": "ultra", "description": "teste"}}), flush=True)
        return 0
    if argv and argv[0] == "blender":
        return blender(argv[1])
    a, i = {}, 1
    while i < len(argv):                               # --chave valor | --opcao
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            a[argv[i]] = argv[i + 1]
            i += 2
        else:
            a[argv[i]] = ""
            i += 1
    if "--fail" in argv or a.get("--room") == "banheiro":
        if a.get("--work-dir"):
            Path(a["--work-dir"]).mkdir(parents=True, exist_ok=True)
        print(json.dumps({"event": "error", "message": "Falha simulada de teste", "detail": "Traceback..."}), flush=True)
        return 1
    for stage, f in (("preparo", 0.02), ("analise", 0.2), ("layout", 0.25), ("render", 0.6), ("video", 1.0)):
        print(json.dumps({"event": "progress", "stage": stage, "fraction": f, "message": f"etapa {stage}"}), flush=True)
        time.sleep(0.05)
    out = a["--out"]
    import imageio_ffmpeg
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=gray:s=160x120:d=0.5", "-pix_fmt", "yuv420p", out], check=True)
    report, work = out + ".json", None
    if a.get("--work-dir"):
        work = Path(a["--work-dir"])
        report = str(_write_job_files(work, a["--photo"]))
    print(json.dumps({"event": "log", "level": "info", "message": "registro de teste"}), flush=True)
    print(json.dumps({"event": "done", "output": out, "report": report,
                      **({"work_dir": str(work)} if work else {})}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
