"""Linha de comando / processo de trabalho.

Uso:
  python -m mobilia.cli run --photo sala.jpg --room sala --out video.mp4 [--quality final] [--json]
  python -m mobilia.cli hardware [--json]
  python -m mobilia.cli analyze --photo sala.jpg --out pasta/

Com ``--json`` cada evento de progresso é impresso como uma linha JSON (protocolo em
mobilia/jobs.py) — é assim que a interface gráfica acompanha o trabalho.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback


def _cmd_run(a) -> int:
    from .jobs import JobRequest, ProgressEmitter
    from .pipeline import run_job

    def human(ev):
        if ev["event"] == "progress":
            print(f"[{ev['fraction'] * 100:5.1f}%] {ev['message']}", flush=True)
        elif ev["event"] in ("log", "error"):
            print(f"{ev['event'].upper()}: {ev['message']}", flush=True)
        elif ev["event"] == "done":
            print(f"Pronto: {ev['output']}", flush=True)

    em = ProgressEmitter(stream=sys.stdout) if a.json else ProgressEmitter(callback=human)
    req = JobRequest(photo_path=a.photo, room_type=a.room, output_path=a.out, quality=a.quality,
                     work_dir=a.work_dir, seed=a.seed)
    try:
        req.validate()
        res = run_job(req, em)
        em.done(res)
        return 0
    except KeyboardInterrupt:
        em.error("Cancelado pelo usuário.")
        return 130
    except Exception as e:  # erro reportado à UI com detalhe técnico
        em.error(str(e) or e.__class__.__name__, traceback.format_exc())
        return 1


def _cmd_batch(a) -> int:
    """Processa uma pasta inteira (lote). Um CSV opcional define o ambiente por foto:
    arquivo,ambiente  (ex.: sala01.jpg,sala)."""
    import csv
    from pathlib import Path

    from .jobs import ROOM_TYPES, JobRequest, ProgressEmitter
    from .pipeline import run_job

    src, dst = Path(a.dir), Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    rooms = {}
    if a.manifest:
        with open(a.manifest, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[1].strip() in ROOM_TYPES:
                    rooms[row[0].strip()] = row[1].strip()
    photos = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"))
    results = []
    for i, p in enumerate(photos, 1):
        room = rooms.get(p.name, a.room)
        out = dst / f"{p.stem}_{room}.mp4"
        print(f"[{i}/{len(photos)}] {p.name} ({room})", flush=True)
        em = ProgressEmitter(callback=lambda ev: ev["event"] == "error" and print("  ERRO:", ev["message"], flush=True))
        try:
            res = run_job(JobRequest(str(p), room, str(out), quality=a.quality, seed=a.seed), em)
            results.append({"foto": p.name, "ambiente": room, "video": res.output_path, "ok": True})
        except Exception as e:
            results.append({"foto": p.name, "ambiente": room, "erro": str(e), "ok": False})
    (dst / "lote.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(r["ok"] for r in results)
    print(f"Lote concluído: {ok}/{len(results)} vídeos gerados.")
    return 0 if ok == len(results) else 1


def _cmd_prefetch(a) -> int:
    """Baixa pesos dos modelos de IA e assets 3D para uso 100% offline (primeiro uso/instalador)."""
    from .assets.polyhaven import prefetch_catalog
    from .config import (ASSETS_DIR, GEOMETRY_MODEL_ID, GEOMETRY_MODEL_ID_SMALL,
                         SEGMENTATION_MODEL_ID)

    def emit(ev):
        if a.json:
            print(json.dumps(ev, ensure_ascii=False), flush=True)
        elif ev.get("event") == "progress":
            print(f"[{ev['fraction'] * 100:5.1f}%] {ev['message']}", flush=True)

    from huggingface_hub import snapshot_download

    repos = [GEOMETRY_MODEL_ID, GEOMETRY_MODEL_ID_SMALL, SEGMENTATION_MODEL_ID]
    for i, repo in enumerate(repos):
        emit({"event": "progress", "stage": "modelos", "fraction": round(0.6 * i / len(repos), 3),
              "message": f"Baixando modelo de IA {repo}"})
        snapshot_download(repo, allow_patterns=["*.json", "*.pt", "*.bin", "*.safetensors", "*.txt"])
    failed = prefetch_catalog(ASSETS_DIR, ("1k", "2k"), progress=lambda f, pid: emit(
        {"event": "progress", "stage": "assets", "fraction": round(0.6 + 0.4 * f, 3),
         "message": f"Baixando modelo 3D {pid}"}))
    emit({"event": "done", "failed": failed})
    return 0 if not failed else 1


def _cmd_hardware(a) -> int:
    from .hardware import choose_profile, detect_hardware

    rep = detect_hardware()
    prof = choose_profile(rep)
    if a.json:
        out = json.loads(rep.to_json())
        out["profile"] = prof.__dict__
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(rep.to_json())
        print(f"Perfil: {prof.name} — {prof.description}")
    return 0


def _cmd_analyze(a) -> int:
    from .pipeline import analyze_photo

    analyze_photo(a.photo, a.out)
    print(f"Análise salva em {a.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mobilia", description="Mobília Studio — decoração virtual em vídeo")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="gera o vídeo de decoração a partir de uma foto")
    r.add_argument("--photo", required=True)
    r.add_argument("--room", required=True, help="sala | quarto | cozinha | banheiro | escritorio | sala_jantar")
    r.add_argument("--out", required=True)
    r.add_argument("--quality", default="auto", choices=["auto", "previa", "final"])
    r.add_argument("--work-dir", default=None)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--json", action="store_true", help="eventos de progresso em JSON (uso pela UI)")
    h = sub.add_parser("hardware", help="mostra GPU/VRAM detectadas e o perfil escolhido")
    h.add_argument("--json", action="store_true")
    an = sub.add_parser("analyze", help="só a análise da foto (debug)")
    an.add_argument("--photo", required=True)
    an.add_argument("--out", required=True)
    b = sub.add_parser("batch", help="processa uma pasta de fotos (lote)")
    b.add_argument("--dir", required=True, help="pasta com as fotos")
    b.add_argument("--out", required=True, help="pasta de saída dos vídeos")
    b.add_argument("--room", default="sala", help="ambiente padrão")
    b.add_argument("--manifest", default=None, help="CSV opcional: arquivo,ambiente")
    b.add_argument("--quality", default="auto", choices=["auto", "previa", "final"])
    b.add_argument("--seed", type=int, default=7)
    pf = sub.add_parser("prefetch", help="baixa modelos de IA e assets 3D (uso offline)")
    pf.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    return {"run": _cmd_run, "hardware": _cmd_hardware, "analyze": _cmd_analyze, "batch": _cmd_batch,
            "prefetch": _cmd_prefetch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
