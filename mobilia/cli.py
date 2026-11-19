"""Linha de comando / processo de trabalho.

Uso:
  python -m mobilia.cli run --photo sala.jpg --room sala --out video.mp4 [--quality final] [--json]
  python -m mobilia.cli hardware [--json]
  python -m mobilia.cli analyze --photo sala.jpg --out pasta/
  python -m mobilia.cli prefetch [--modelos] [--ativos] [--res 1k,2k] [--json]
  python -m mobilia.cli aquecer [--sem-blender] [--sem-modelos] [--json]
  python -m mobilia.cli design --photo sala.jpg --funcao mudar_estilo --estilo escandinavo --ambiente sala
                               [--n 4] [--seed 7] [--mask area.png] [--prompt "texto"] [--evento natal]
                               [--out-dir pasta/] [--quality rapido|final] [--json]
  python -m mobilia.cli design --photo sala.jpg --funcao melhorar [--mask remover.png] [--sem-ceu] [--ampliar-4k]
  python -m mobilia.cli design-lote --fotos a.jpg b.jpg --imovel "Apto 302" --funcao melhorar [--exportar 2048]
  python -m mobilia.cli exportar --images v1.jpg v2.jpg --out-dir anuncio/ [--lado 2048|4k] [--rotulo]
  python -m mobilia.cli ampliar|upscale --image versao.jpg --out versao_4k.jpg [--lado 3840] [--json]
  python -m mobilia.cli showcase --before antes.jpg --after v1.jpg [v2.jpg ...] --out video.mp4
                                 [--format 16x9|9x16] [--duracao 15] [--rotulos "Escandinavo" ...]
                                 [--sem-rotulos] [--sem-marca] [--json]

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
from pathlib import Path

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
        elif ev["event"] == "mode":
            print(f"Modo {ev['mode']}: {ev.get('reason', '')}", flush=True)
        elif ev["event"] == "done":
            if ev.get("warning"):
                print(f"AVISO: {ev['warning']}", flush=True)
            print(f"Pronto: {ev['output']}" if ev.get("output") else "Nenhum vídeo entregue (veja o aviso).",
                  flush=True)

    em = ProgressEmitter(stream=sys.stdout) if a.json else ProgressEmitter(callback=human)
    req = JobRequest(photo_path=a.photo, room_type=a.room, output_path=a.out, quality=a.quality,
                     work_dir=a.work_dir, seed=a.seed, mode=a.mode, accept_weak_plate=a.aceitar_placa_fraca,
                     plate=a.placa, empty_image=a.sala_vazia, fabric=not a.sem_tecido)
    try:
        req.validate()
        res = run_job(req, em, t_ref=env_time(ENV_T_INICIO) or T_IMPORT)
        if res.output_path and a.formato != "original":
            from .video.reframe import reframe_mp4

            em.log(f"Recortando o vídeo para {a.formato.replace('x', ':')}")
            res.output_path = reframe_mp4(res.output_path, a.formato, res.report_path)
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
            results.append({"foto": p.name, "ambiente": room, "video": res.output_path or None, "modo": res.mode,
                            "aviso": res.warning, "ok": bool(res.output_path)})
        except Exception as e:
            results.append({"foto": p.name, "ambiente": room, "erro": str(e), "ok": False})
    (dst / "lote.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(r["ok"] for r in results)
    print(f"Lote concluído: {ok}/{len(results)} vídeos gerados.")
    return 0 if ok == len(results) else 1


def _json_or_human(a):
    import threading

    lock = threading.Lock()   # o download dos pesos informa o progresso também de uma thread auxiliar

    def emit(ev):
        with lock:
            _emit(ev)

    def _emit(ev):
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
        from .analysis.weights import WeightsError, prefetch_models

        span = 0.6 if both else 1.0
        try:
            prefetch_models(lambda f, repo: emit({"event": "progress", "stage": "modelos",
                                                  "fraction": round(span * f, 3),
                                                  "message": f"Baixando modelo de IA {repo}"}))
        except WeightsError as e:  # peso bloqueado, ausente ou corrompido: mensagem já em português
            emit({"event": "log", "level": "error", "message": str(e)})
            print(str(e), file=sys.stderr, flush=True)
            return 2
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


def _cmd_design(a) -> int:
    """Redesenho de ambiente por IA (fluxo principal): N versões novas da mesma foto.

    Eventos (``--json``): ``progress`` (stage, fraction, message), ``preview`` (path, index) a cada versão pronta,
    ``log`` e, no fim, ``done`` com ``outputs`` (lista de JPEGs), ``output`` (o primeiro), ``before`` (a foto no
    tamanho das versões), ``report`` (design.json), ``funcao``, ``estilo``, ``ambiente``, ``seeds`` e ``size`` —
    mais ``aviso`` e ``funcao_pedida`` quando o motor trocou de função (foto de ambiente vazio -> "Mobiliar"); em
    falha, ``error`` (message, detail)."""
    from .design import api, presets

    def human(ev):
        if ev["event"] == "progress":
            print(f"[{ev['fraction'] * 100:5.1f}%] {ev['message']}", flush=True)
        elif ev["event"] == "preview":
            print(f"Versão {ev['index'] + 1}: {ev['path']}", flush=True)
        elif ev["event"] in ("log", "error"):
            print(f"{ev['event'].upper()}: {ev['message']}", flush=True)
        elif ev["event"] == "done":
            if ev.get("aviso"):
                print(ev["aviso"], flush=True)
            print("Pronto:\n  " + "\n  ".join(ev["outputs"]), flush=True)

    def emit(ev):
        if a.json:
            print(json.dumps(ev, ensure_ascii=False), flush=True)
        else:
            human(ev)

    try:
        f = presets.function(a.funcao)
        if not a.estilo and f.key not in ("evento", "editar_area", "melhorar", "esvaziar"):
            a.estilo = "moderno"
        common = dict(n=a.n, seed=a.seed, out_dir=a.out_dir, quality=a.quality, progress=emit, device=a.device)
        if f.key == "texto":
            res = api.text_to_design(a.prompt or "", a.estilo, a.ambiente, aspect=a.formato, **common)
        else:
            if not a.photo:
                raise ValueError("Escolha a foto do ambiente (--photo).")
            if not os.path.isfile(a.photo):
                raise FileNotFoundError(f"Foto não encontrada: {a.photo}")
            if f.key == "editar_area":
                if not a.mask:
                    raise ValueError("Pinte a área a editar (--mask).")
                res = api.edit_area(a.photo, a.mask, a.prompt or "", a.estilo, a.ambiente, **common)
            elif f.key == "esvaziar":
                res = api.empty_room(a.photo, prompt=a.prompt or "", keep_curtains=not a.tirar_cortinas,
                                     prefill=a.partida, **common)
            elif f.key == "melhorar":
                from .design.enhance import enhance

                if a.mask and not os.path.isfile(a.mask):
                    raise FileNotFoundError(f"Máscara não encontrada: {a.mask}")
                res = enhance(a.photo, a.out_dir, _enhance_opts(a), mask=a.mask, progress=emit, device=a.device)
            else:
                res = api.redesign(a.photo, f.key, a.estilo, a.ambiente, prompt=a.prompt or "", evento=a.evento,
                                   repair=False if a.sem_reparo else None, auto_empty=not a.manter_funcao,
                                   refine={"auto": None, "sim": True, "nao": False}[a.refino], **common)
        emit(res.to_event())
        return 0
    except KeyboardInterrupt:
        emit({"event": "error", "message": "Cancelado pelo usuário.", "detail": ""})
        return 130
    except Exception as e:  # erro reportado à UI com detalhe técnico
        emit({"event": "error", "message": str(e) or e.__class__.__name__, "detail": traceback.format_exc()})
        return 1


def _enhance_opts(a):
    from .design.enhance import EnhanceOptions

    return EnhanceOptions(exposicao=not a.sem_exposicao, branco=not a.sem_branco, verticais=not a.sem_verticais,
                          ceu=not a.sem_ceu, nitidez=not a.sem_nitidez, ampliar_4k=bool(a.ampliar_4k),
                          ceu_preset=int(a.ceu or 0))


def _add_enhance_args(p) -> None:
    g = p.add_argument_group("Melhorar foto (todas as etapas ligadas por padrão)")
    g.add_argument("--sem-exposicao", action="store_true", help="sem exposição automática / HDR natural")
    g.add_argument("--sem-branco", action="store_true", help="sem correção do balanço de branco")
    g.add_argument("--sem-verticais", action="store_true", help="sem endireitar as verticais")
    g.add_argument("--sem-ceu", action="store_true", help="sem trocar o céu")
    g.add_argument("--sem-nitidez", action="store_true", help="sem nitidez e redução de ruído")
    g.add_argument("--ampliar-4k", action="store_true", help="saída com 3840 px no lado maior")
    g.add_argument("--ceu", type=int, default=0, help="modelo de céu (0-3)")


def _cmd_design_batch(a) -> int:
    """Lote: várias fotos do mesmo imóvel com a mesma função. Eventos ``progress``, ``item`` (index, total, photo,
    status processando|pronto|erro, outputs, message) e ``done`` (outputs = entregas, output = a 1ª, folder)."""
    from .design import listing, presets

    emit = _json_or_human(a)
    try:
        f = presets.function(a.funcao)
        fotos = list(a.fotos or [])
        if a.pasta:
            exts = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff")
            fotos += sorted(str(p) for p in Path(a.pasta).iterdir() if p.suffix.lower() in exts)
        miss = [p for p in fotos if not os.path.isfile(p)]
        if miss:
            raise FileNotFoundError(f"Foto não encontrada: {miss[0]}")
        if a.out_dir:
            out = Path(a.out_dir)
        else:
            from .config import JOBS_DIR

            out = JOBS_DIR / listing.property_folder(a.imovel or time.strftime("lote_%Y%m%d_%H%M%S"))
        export = None
        if a.exportar:
            export = {"lado": listing.norm_long_side(a.exportar),
                      "rotulo": True if a.rotulo else False if a.sem_rotulo else None}
        rep = listing.run_batch(fotos, out, f.key, progress=emit, export=export, opts=_enhance_opts(a),
                                estilo=a.estilo, ambiente=a.ambiente, evento=a.evento, prompt=a.prompt or "",
                                n=a.n, seed=a.seed, quality=a.quality, device=a.device)
        ok = [i for i in rep["itens"] if i["status"] == "pronto"]
        if not ok:
            raise RuntimeError("Nenhuma foto do lote foi processada: " +
                               "; ".join(i.get("message", "") for i in rep["itens"])[:400])
        emit({"event": "done", "outputs": rep["entregas"], "output": rep["entregas"][0], "folder": str(out),
              "anuncio": rep["anuncio"], "itens": rep["itens"], "funcao": f.key})
        if not a.json:
            print(f"Pronto: {len(ok)} de {len(fotos)} fotos em {out}", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        ev = {"event": "error", "message": str(e) or e.__class__.__name__, "detail": traceback.format_exc()}
        print(json.dumps(ev, ensure_ascii=False) if a.json else f"ERRO: {ev['message']}", flush=True)
        return 1


def _cmd_export(a) -> int:
    """"Exportar para anúncio": JPEGs de 2048 px (ou 4K) com o selo "Imagem ilustrativa" opcional. Eventos
    ``progress`` e ``done`` (outputs, output = o 1º, folder)."""
    from .design import listing

    emit = _json_or_human(a)
    try:
        for p in a.images:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Imagem não encontrada: {p}")
        lado = listing.norm_long_side(a.lado)
        label = True if a.rotulo else False if a.sem_rotulo else listing.default_label(a.funcao)
        emit({"event": "progress", "stage": "exportar", "fraction": 0.0, "message": "Preparando as imagens"})
        outs = listing.export_for_listing(
            a.images, a.out_dir, lado, label,
            progress=lambda fr: emit({"event": "progress", "stage": "exportar", "fraction": round(fr, 3),
                                      "message": "Exportando para o anúncio"}),
            log=lambda m: emit({"event": "log", "level": "warning", "message": m}))
        emit({"event": "done", "outputs": outs, "output": outs[0], "folder": str(a.out_dir), "rotulo": label,
              "lado": lado})
        if not a.json:
            print("Pronto:\n  " + "\n  ".join(outs), flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        ev = {"event": "error", "message": str(e) or e.__class__.__name__, "detail": traceback.format_exc()}
        print(json.dumps(ev, ensure_ascii=False) if a.json else f"ERRO: {ev['message']}", flush=True)
        return 1


def _cmd_upscale(a) -> int:
    """"Ampliar 4K" de uma versão escolhida. Eventos: ``progress`` e ``done`` (output, size, s)."""
    from .design.upscale import upscale

    emit = _json_or_human(a)
    try:
        if not os.path.isfile(a.image):
            raise FileNotFoundError(f"Imagem não encontrada: {a.image}")
        emit({"event": "progress", "stage": "ampliar", "fraction": 0.0, "message": "Carregando o modelo de ampliação"})
        def dl(done, total):
            mb = f"{done / 2 ** 20:.0f} de {total / 2 ** 20:.0f} MB" if total else f"{done / 2 ** 20:.0f} MB"
            emit({"event": "progress", "stage": "baixar", "fraction": round(min(1.0, done / total), 3) if total else 0.0,
                  "message": f"Baixando o modelo de ampliação (só na primeira vez): {mb}"})

        _, info = upscale(a.image, a.out, long_side=a.lado, device=a.device, download_progress=dl,
                          progress=lambda f: emit({"event": "progress", "stage": "ampliar", "fraction": round(f, 3),
                                                   "message": "Ampliando para 4K"}))
        emit({"event": "done", "output": a.out, "size": info["para"], "s": info["s"]})
        if not a.json:
            print(f"Pronto: {a.out} ({info['para'][0]}x{info['para'][1]})", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        ev = {"event": "error", "message": str(e) or e.__class__.__name__, "detail": traceback.format_exc()}
        print(json.dumps(ev, ensure_ascii=False) if a.json else f"ERRO: {ev['message']}", flush=True)
        return 1


def _cmd_showcase(a) -> int:
    """"Vídeo antes e depois": a foto original revelada pelas versões redesenhadas (MP4 H.264 para redes sociais).

    Eventos (``--json``): ``progress`` (stage ``video``) e ``done`` (output, size, frames, fps, duration_s, labels,
    codec, s); em falha, ``error`` (message, detail)."""
    from .video.showcase import render_showcase

    emit = _json_or_human(a)
    try:
        if not os.path.isfile(a.before):
            raise FileNotFoundError(f"Foto original não encontrada: {a.before}")
        for f in a.after:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"Versão não encontrada: {f}")
        emit({"event": "progress", "stage": "video", "fraction": 0.0, "message": "Preparando o vídeo antes e depois"})
        res = render_showcase(a.before, a.after, a.out, fmt=a.format, fps=a.fps, duration=a.duracao,
                              labels=a.rotulos, show_labels=not a.sem_rotulos, brand=not a.sem_marca,
                              codec=a.codec, progress=lambda f: emit(
                                  {"event": "progress", "stage": "video", "fraction": round(f, 3),
                                   "message": "Gerando o vídeo antes e depois"}))
        emit(res.to_event())
        if not a.json:
            print(f"Pronto: {res.output} ({res.size[0]}x{res.size[1]}, {res.duration_s:.1f} s, {res.codec})",
                  flush=True)
        return 0
    except KeyboardInterrupt:
        emit({"event": "error", "message": "Cancelado pelo usuário.", "detail": ""})
        return 130
    except Exception as e:  # noqa: BLE001 - erro reportado à UI com detalhe técnico
        ev = {"event": "error", "message": str(e) or e.__class__.__name__, "detail": traceback.format_exc()}
        print(json.dumps(ev, ensure_ascii=False) if a.json else f"ERRO: {ev['message']}", flush=True)
        return 1


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
    p = argparse.ArgumentParser(prog="mobilia", description="7xAi — decoração virtual em vídeo")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="gera o vídeo de decoração a partir de uma foto")
    r.add_argument("--photo", required=True)
    r.add_argument("--room", required=True, help="sala | quarto | cozinha | banheiro | escritorio | sala_jantar")
    r.add_argument("--out", required=True)
    r.add_argument("--quality", default="auto", choices=["auto", "previa", "final"])
    r.add_argument("--work-dir", default=None)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--json", action="store_true", help="eventos de progresso em JSON (uso pela UI)")
    r.add_argument("--mode", default="auto", choices=["auto", "A", "B"],
                   help="auto (padrão): foto vazia -> A, foto mobiliada -> B; A/B força o modo")
    r.add_argument("--aceitar-placa-fraca", action="store_true",
                   help="modo B: gera o vídeo mesmo com a sala vazia reprovada pelo portão de qualidade")
    r.add_argument("--placa", default="auto", choices=["auto", "ia", "geometria"],
                   help="modo B: sala vazia do início gerada por IA (auto: quando o gerador está instalado) ou "
                        "refeita pela geometria")
    r.add_argument("--sala-vazia", default=None,
                   help="modo B: imagem do cômodo vazio já escolhida (resultado do 'Esvaziar ambiente')")
    r.add_argument("--sem-tecido", action="store_true",
                   help="modo B: sem o tecido (roupa de cama, almofadas) assentando quando a peça pousa")
    r.add_argument("--formato", default="original", choices=["original", "16x9", "9x16"],
                   help="modo B: recorta o MP4 final para 16:9 (1920x1080) ou 9:16 (1080x1920)")
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
    d = sub.add_parser("design", help="redesenho do ambiente por IA (N versões novas da mesma foto)")
    d.add_argument("--photo", default=None, help="foto do ambiente (não usada em --funcao texto)")
    d.add_argument("--funcao", default="mudar_estilo",
                   help="mudar_estilo | reforma | redesign_simples | mobiliar | editar_area | fachada | evento | texto | "
                        "melhorar | esvaziar")
    d.add_argument("--estilo", default=None, help="moderno, escandinavo, luxo... (veja mobilia/design/presets.py)")
    d.add_argument("--ambiente", default="sala",
                   help="sala | quarto | cozinha | banheiro | escritorio | sala_jantar | varanda | fachada")
    d.add_argument("--evento", default=None, help="função evento: aniversario | casamento | natal | ...")
    d.add_argument("--n", type=int, default=4, help="quantas versões (padrão 4)")
    d.add_argument("--seed", type=int, default=None, help="semente (padrão: aleatória)")
    d.add_argument("--mask", default=None,
                   help="PNG da área a editar (branco = editar), função editar_area; no melhorar, o que remover")
    d.add_argument("--prompt", default="", help="texto livre (o que o cliente quer)")
    d.add_argument("--formato", default="paisagem", help="função texto: paisagem | retrato | quadrado | 16:9 | 9:16")
    d.add_argument("--sem-reparo", action="store_true",
                   help="não refazer como parede as janelas/portas que o gerador inventar")
    d.add_argument("--manter-funcao", action="store_true",
                   help="não trocar para 'Mobiliar' quando a foto for de um ambiente vazio")
    d.add_argument("--refino", default="auto", choices=["auto", "sim", "nao"],
                   help="passada de detalhe a 1,5x (auto: só na GPU, qualidade final)")
    d.add_argument("--out-dir", default=None, help="pasta de saída (padrão: <dados>/jobs/design_<data>)")
    d.add_argument("--quality", default="final", choices=["rapido", "final"])
    d.add_argument("--device", default="auto", help="auto | cuda | cpu")
    d.add_argument("--json", action="store_true", help="eventos de progresso em JSON (uso pela UI)")
    d.add_argument("--tirar-cortinas", action="store_true", help="esvaziar: tira também cortinas e persianas")
    d.add_argument("--partida", default="auto", choices=["auto", "geometria", "suave"],
                   help="esvaziar: ponto de partida da geração (auto alterna geometria e suave entre as versões)")
    _add_enhance_args(d)
    bl = sub.add_parser("design-lote", help="lote: várias fotos de um imóvel com a mesma função (ex.: melhorar)")
    bl.add_argument("--fotos", nargs="*", default=None, help="fotos do imóvel")
    bl.add_argument("--pasta", default=None, help="ou uma pasta com as fotos")
    bl.add_argument("--imovel", default="", help="nome do imóvel (vira o nome da pasta de saída)")
    bl.add_argument("--funcao", default="melhorar")
    bl.add_argument("--estilo", default=None)
    bl.add_argument("--ambiente", default="sala")
    bl.add_argument("--evento", default=None)
    bl.add_argument("--prompt", default="")
    bl.add_argument("--n", type=int, default=1, help="versões por foto (funções de redesenho)")
    bl.add_argument("--seed", type=int, default=None)
    bl.add_argument("--quality", default="final", choices=["rapido", "final"])
    bl.add_argument("--out-dir", default=None, help="pasta de saída (padrão: <dados>/jobs/<imóvel>)")
    bl.add_argument("--exportar", default=None, help="também exporta para anúncio: 2048 | 4k")
    bl.add_argument("--rotulo", action="store_true", help="selo 'Imagem ilustrativa' na exportação")
    bl.add_argument("--sem-rotulo", action="store_true")
    bl.add_argument("--device", default="auto")
    bl.add_argument("--json", action="store_true")
    _add_enhance_args(bl)
    ex = sub.add_parser("exportar", help="exportar para anúncio (JPEG 2048 px ou 4K, selo opcional)")
    ex.add_argument("--images", nargs="+", required=True)
    ex.add_argument("--out-dir", required=True)
    ex.add_argument("--lado", default="2048", help="2048 | 4k")
    ex.add_argument("--funcao", default=None, help="função que gerou as imagens (define o selo padrão)")
    ex.add_argument("--rotulo", action="store_true", help="com o selo 'Imagem ilustrativa'")
    ex.add_argument("--sem-rotulo", action="store_true", help="sem o selo")
    ex.add_argument("--json", action="store_true")
    up = sub.add_parser("ampliar", aliases=["upscale"], help="amplia uma versão para 4K (Real-ESRGAN x4)")
    up.add_argument("--image", required=True)
    up.add_argument("--out", required=True)
    up.add_argument("--lado", "--long-side", type=int, default=3840, help="lado maior final (0 = 4x sem reduzir)")
    up.add_argument("--device", default="auto")
    up.add_argument("--json", action="store_true")
    sc = sub.add_parser("showcase", help="vídeo antes e depois (foto original -> versões redesenhadas)")
    sc.add_argument("--before", required=True, help="foto original (ou o antes.jpg do redesenho)")
    sc.add_argument("--after", required=True, nargs="+", help="versões redesenhadas, na ordem do vídeo")
    sc.add_argument("--out", required=True, help="MP4 de saída")
    sc.add_argument("--format", default="16x9", help="16x9 (1920x1080) | 9x16 (1080x1920, Reels/Stories)")
    sc.add_argument("--duracao", "--duration", type=float, default=None,
                    help="segundos (padrão: 10-20 s conforme o número de versões)")
    sc.add_argument("--fps", type=int, default=30)
    sc.add_argument("--rotulos", "--labels", nargs="*", default=None,
                    help="nome de cada versão no rótulo 'Depois' (padrão: estilo lido do design.json)")
    sc.add_argument("--sem-rotulos", action="store_true", help="sem os rótulos Antes/Depois")
    sc.add_argument("--sem-marca", action="store_true", help="sem a marca 7xAi no canto")
    sc.add_argument("--codec", default=None, help="libx264 | h264_nvenc (padrão: automático)")
    sc.add_argument("--json", action="store_true")
    rc = sub.add_parser("recompor", help="refaz composição e MP4 de um trabalho já renderizado "
                                       "(precisa dos EXR: gere o trabalho com MOBILIA_MANTER_EXR=1)")
    rc.add_argument("--work-dir", required=True)
    rc.add_argument("--out", default=None, help="MP4 de saída (padrão: o do pedido original)")
    a = p.parse_args(argv)
    # caches de kernels da NVIDIA grandes também quando a CLI é chamada direto (desenvolvimento);
    # precisa vir antes do primeiro import do PyTorch e vale para o Blender (processo filho)
    from .runtime import apply_gpu_cache_env, use_installed_models

    apply_gpu_cache_env(os.environ)
    use_installed_models(os.environ)
    return {"run": _cmd_run, "hardware": _cmd_hardware, "analyze": _cmd_analyze, "batch": _cmd_batch,
            "prefetch": _cmd_prefetch, "aquecer": _cmd_warmup, "recompor": _cmd_recompose,
            "design": _cmd_design, "design-lote": _cmd_design_batch, "exportar": _cmd_export,
            "ampliar": _cmd_upscale, "upscale": _cmd_upscale,
            "showcase": _cmd_showcase}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
