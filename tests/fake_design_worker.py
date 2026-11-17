"""DUBLÊ DE TESTE da interface de redesenho (tests/test_ui.py). Não faz parte do produto.

Imita o protocolo JSON de ``mobilia.cli``:
- ``design ...``: progress/preview por versão e ``done`` (outputs, before, report, size, seeds). Grava ``args.json`` na
  pasta de saída com os argumentos recebidos (os testes conferem que nenhum argumento de sinal Qt vazou).
  ``--prompt`` com "falhar" -> evento ``error``; com "devagar" -> etapas lentas (para testar o Cancelar).
  ``MOBILIA_FAKE_DESIGN_SRC=<pasta>``: copia as imagens dessa pasta (capturas de tela com imagens reais).
- ``upscale --image --out``: grava a imagem ampliada (2x, rápido) e ``done`` com ``size``.
- ``showcase --before --after a b ... --out --format``: grava um MP4 minúsculo e ``done``.
- qualquer outro comando: ``tests/fake_worker.py`` (modo legado e ``hardware``).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _emit(**ev):
    print(json.dumps(ev, ensure_ascii=False), flush=True)


def _parse(argv):
    a, multi, i = {}, {}, 0
    while i < len(argv):
        k = argv[i]
        if not k.startswith("--"):
            i += 1
            continue
        vals = []
        j = i + 1
        while j < len(argv) and not argv[j].startswith("--"):
            vals.append(argv[j]); j += 1
        a[k] = vals[0] if vals else ""
        multi[k] = vals
        i = j
    return a, multi


def design(argv) -> int:
    from PIL import Image, ImageDraw, ImageEnhance

    a, _ = _parse(argv)
    out = Path(a["--out-dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(["design"] + argv, ensure_ascii=False), encoding="utf-8")
    prompt = a.get("--prompt", "")
    if "falhar" in prompt:
        _emit(event="error", message="Falha simulada do redesenho", detail="Traceback (teste)")
        return 1
    slow = "devagar" in prompt
    n = int(a.get("--n", "4"))
    funcao = a.get("--funcao", "mudar_estilo")
    estilo = a.get("--estilo") or ""
    src_dir = os.environ.get("MOBILIA_FAKE_DESIGN_SRC")
    srcs = sorted(p for p in Path(src_dir).glob("*.jpg") if p.name != "antes.jpg" and "_4k" not in p.name) \
        if src_dir else []
    _emit(event="progress", stage="preparo", fraction=0.02, message="Preparando a foto")
    time.sleep(1.0 if slow else 0.03)
    if funcao != "texto":
        _emit(event="progress", stage="analise", fraction=0.12, message="Analisando o ambiente")
        time.sleep(1.0 if slow else 0.03)
    if funcao == "texto":
        base = Image.new("RGB", (192, 128), (200, 190, 175))
    else:
        base = Image.open(a["--photo"]).convert("RGB")
        base.thumbnail((256, 256))
    if src_dir and (Path(src_dir) / "antes.jpg").is_file():
        before = out / "antes.jpg"
        shutil.copyfile(Path(src_dir) / "antes.jpg", before)
    elif funcao != "texto":
        before = out / "antes.jpg"
        base.save(before, quality=90)
    else:
        before = None
    if funcao == "editar_area":
        assert Path(a["--mask"]).is_file(), "máscara ausente"
    outs, tag = [], "_".join(x for x in (funcao, a.get("--evento") or estilo) if x)
    for i in range(n):
        _emit(event="progress", stage="geracao", fraction=round(0.15 + 0.8 * i / n, 3),
              message=f"Gerando a versão {i + 1} de {n}")
        time.sleep(8.0 if slow else 0.05)
        p = out / f"{tag}_{i + 1}.jpg"
        if srcs:
            shutil.copyfile(srcs[i % len(srcs)], p)
        else:
            img = ImageEnhance.Color(base).enhance(0.4 + 0.5 * i)
            ImageDraw.Draw(img).rectangle([8, 8, 40, 24], fill=(60 * i % 255, 120, 200))
            img.save(p, quality=90)
        outs.append(str(p))
        _emit(event="preview", path=str(p), index=i)
    w, h = Image.open(outs[0]).size
    rep = out / "design.json"
    rep.write_text(json.dumps({"funcao": funcao, "estilo": estilo, "args": argv}, ensure_ascii=False), encoding="utf-8")
    _emit(event="progress", stage="final", fraction=1.0, message="Pronto")
    _emit(event="done", output=outs[0], outputs=outs, before=str(before) if before else None, report=str(rep),
          funcao=funcao, estilo=estilo, ambiente=a.get("--ambiente"), seeds=list(range(n)), size=[w, h])
    return 0


def upscale(argv) -> int:
    from PIL import Image

    a, _ = _parse(argv)
    img = Image.open(a["--image"]).convert("RGB")
    for f in (0.25, 0.6, 0.95):
        _emit(event="progress", stage="ampliar", fraction=f, message="Ampliando para 4K")
        time.sleep(0.03)
    big = img.resize((img.width * 2, img.height * 2))
    Path(a["--out"]).parent.mkdir(parents=True, exist_ok=True)
    big.save(a["--out"], quality=90)
    (Path(a["--out"]).parent / "upscale_args.json").write_text(json.dumps(argv), encoding="utf-8")
    _emit(event="done", output=a["--out"], size=[big.width, big.height], s=0.1)
    return 0


def showcase(argv) -> int:
    a, multi = _parse(argv)
    assert a.get("--format") in ("16x9", "9x16"), a.get("--format")
    assert Path(a["--before"]).is_file() and multi["--after"] and all(Path(p).is_file() for p in multi["--after"])
    out = Path(a["--out"]); out.parent.mkdir(parents=True, exist_ok=True)
    for f in (0.2, 0.5, 0.9):
        _emit(event="progress", stage="video", fraction=f, message="Montando o vídeo")
        time.sleep(0.03)
    try:
        import imageio_ffmpeg
        size = "96x54" if a["--format"] == "16x9" else "54x96"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"color=c=gray:s={size}:d=0.3", "-pix_fmt", "yuv420p", str(out)], check=True)
    except Exception:  # noqa: BLE001 — sem FFmpeg no ambiente de teste: arquivo simbólico
        out.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    (out.parent / "showcase_args.json").write_text(json.dumps(argv), encoding="utf-8")
    _emit(event="done", output=str(out), format=a["--format"], s=0.1)
    return 0


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "design":
        return design(argv[1:])
    if cmd == "upscale":
        return upscale(argv[1:])
    if cmd == "showcase":
        return showcase(argv[1:])
    import fake_worker
    return fake_worker.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
