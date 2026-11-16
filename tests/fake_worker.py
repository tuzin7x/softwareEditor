"""DUBLÊ DE TESTE (só para tests/test_ui.py): imita o protocolo JSON do worker real e grava um
MP4 minúsculo de verdade com o FFmpeg. Não faz parte do produto."""
import json
import subprocess
import sys
import time


def main(argv):
    if argv and argv[0] == "hardware":
        print(json.dumps({"gpus": [{"name": "NVIDIA GeForce RTX 4070", "vram_total_mb": 12282}],
                          "profile": {"name": "ultra", "description": "teste"}}), flush=True)
        return 0
    a = dict(zip(argv[1::2], argv[2::2]))
    if "--fail" in argv or a.get("--room") == "banheiro":
        print(json.dumps({"event": "error", "message": "Falha simulada de teste", "detail": "Traceback..."}), flush=True)
        return 1
    for stage, f in (("preparo", 0.02), ("analise", 0.2), ("layout", 0.25), ("render", 0.6), ("video", 1.0)):
        print(json.dumps({"event": "progress", "stage": stage, "fraction": f, "message": f"etapa {stage}"}), flush=True)
        time.sleep(0.05)
    out = a["--out"]
    import imageio_ffmpeg
    from pathlib import Path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=gray:s=160x120:d=0.5", "-pix_fmt", "yuv420p", out], check=True)
    print(json.dumps({"event": "done", "output": out, "report": out + ".json"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
