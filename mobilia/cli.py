"""Linha de comando / processo de trabalho.

Uso:
  python -m mobilia.cli run --photo sala.jpg --room sala --out video.mp4 [--quality final] [--json]
  python -m mobilia.cli hardware [--json]
  python -m mobilia.cli analyze --photo sala.jpg --out pasta/
  python -m mobilia.cli prefetch [--modelos] [--ativos] [--res 1k,2k] [--json]
  python -m mobilia.cli aquecer [--sem-blender] [--sem-modelos] [--json]

Com ``--json`` cada evento de progresso é impresso como uma linha JSON (protocolo em
mobilia/jobs.py) — é assim que a interface gráfica acompanha o trabalho.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

# Instante em que o processo do motor começou a rodar código do Mobília (telemetria: o
# lançamento real vem de MOBILIA_T_INICIO quando a interface informa).
T_IMPORT = time.time()


def _cmd_recompose(a) -> int:
    from .jobs import ProgressEmitter
    from .pipeline import recompose

    em = ProgressEmitter(callback=lambda ev: ev["event"] in ("log", "error") and print(ev["message"], flush=True))
    rep = recompose(a.work_dir, a.out, em)
    pres, enc = rep["preservacao"], rep["encode"]
    print(f"Vídeo: {enc['arquivo']} | quadros {pres['quadros']} | bit-exato fora dos móveis: "
          f"{pres['todos_bit_exatos_fora_dos_objetos']} | PSNR: "
          + ", ".join(f"{c['psnr_total_db']} dB" for c in enc["verificacao"]))
    return 0 if pres["todos_bit_exatos_fora_dos_objetos"] else 1


def _cmd_run(a) -> int:
    from .jobs import JobRequest, ProgressEmitter
    from .pipeline import run_job
    from .telemetry import ENV_T_INICIO, env_time

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
        res = run_job(req, em, t_ref=env_time(ENV_T_INICIO) or T_IMPORT)
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


def _json_or_human(a):
    def emit(ev):
        if a.json:
            print(json.dumps(ev, ensure_ascii=False), flush=True)
        elif ev.get("event") == "progress":
            print(f"[{ev['fraction'] * 100:5.1f}%] {ev['message']}", flush=True)
        elif ev.get("event") == "log":
            print(f"{ev.get('level', 'info').upper()}: {ev.get('message', '')}", flush=True)
    return emit


def _cmd_prefetch(a) -> int:
    """Baixa pesos dos modelos de IA e assets 3D para uso 100% offline (instalador).

    ``--modelos``/``--ativos`` limitam a uma das partes (o instalador roda cada uma como uma
    etapa própria, retomável); sem nenhuma das duas, baixa tudo. Idempotente: o que já está no
    cache só é conferido."""
    from .assets.polyhaven import prefetch_catalog
    from .config import ASSETS_DIR

    emit = _json_or_human(a)
    both = not (a.modelos or a.ativos)
    failed: list[str] = []
    if a.modelos or both:
        from .analysis.weights import prefetch_models

        span = 0.6 if both else 1.0
        prefetch_models(lambda f, repo: emit({"event": "progress", "stage": "modelos",
                                              "fraction": round(span * f, 3),
                                              "message": f"Baixando modelo de IA {repo}"}))
    if a.ativos or both:
        res = tuple(r.strip() for r in a.res.split(",") if r.strip()) or ("2k",)
        off = 0.6 if both else 0.0
        failed = prefetch_catalog(ASSETS_DIR, res, progress=lambda f, pid: emit(
            {"event": "progress", "stage": "assets", "fraction": round(off + (1 - off) * f, 3),
             "message": f"Baixando modelo 3D {pid} ({'/'.join(res)})"}))
    emit({"event": "done", "failed": failed})
    return 0 if not failed else 1


def _cmd_warmup(a) -> int:
    """Aquecimento do instalador: kernels do Cycles (OptiX/CUDA) e carga dos modelos de IA.

    Sempre termina com ``{"event": "done", "aquecimento": {...}}``; falhas ficam no resultado
    (o instalador registra e segue)."""
    from .warmup import run

    emit = _json_or_human(a)
    res = run(emit, blender=not a.sem_blender, models=not a.sem_modelos)
    emit({"event": "done", "aquecimento": res})
    if not a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


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
    pf.add_argument("--modelos", action="store_true", help="só os pesos dos modelos de IA")
    pf.add_argument("--ativos", action="store_true", help="só os móveis 3D do catálogo")
    pf.add_argument("--res", default="1k,2k", help="resoluções das texturas dos móveis (ex.: 1k,2k)")
    aq = sub.add_parser("aquecer", help="aquece kernels da GPU (Cycles) e modelos de IA (instalador)")
    aq.add_argument("--json", action="store_true")
    aq.add_argument("--sem-blender", action="store_true")
    aq.add_argument("--sem-modelos", action="store_true")
    rc = sub.add_parser("recompor", help="refaz composição e MP4 de um trabalho já renderizado")
    rc.add_argument("--work-dir", required=True)
    rc.add_argument("--out", default=None, help="MP4 de saída (padrão: o do pedido original)")
    a = p.parse_args(argv)
    # caches de kernels da NVIDIA grandes também quando a CLI é chamada direto (desenvolvimento);
    # precisa vir antes do primeiro import do PyTorch e vale para o Blender (processo filho)
    from .runtime import apply_gpu_cache_env

    apply_gpu_cache_env(os.environ)
    return {"run": _cmd_run, "hardware": _cmd_hardware, "analyze": _cmd_analyze, "batch": _cmd_batch,
            "prefetch": _cmd_prefetch, "aquecer": _cmd_warmup, "recompor": _cmd_recompose}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
