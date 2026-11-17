"""Avalia a detecção automática do modo (A = sala vazia, B = foto mobiliada) num conjunto de fotos.

Uso (precisa do OneFormer no cache; CPU ~10–20 s por foto)::

    python scripts/avaliar_deteccao_modo.py <pasta_com_manifest.json> [--cache <pasta>] [--sam]

O ``manifest.json`` é uma lista de objetos com ``file`` e ``furnished`` (verdadeiro = mobiliada).
Imprime uma linha por foto e a matriz de confusão contra ``furnished``. Com ``--cache`` a segmentação de
cada foto é guardada (``<nome>.seg.npz`` / ``<nome>.pan.npz``) e reaproveitada, para ajustar limites sem
rodar o modelo de novo. Com ``--sam`` roda também as instâncias completas (SAM 2.1) e conta as peças.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--cache")
    ap.add_argument("--sam", action="store_true")
    ap.add_argument("--json", help="grava os resultados por foto neste arquivo")
    a = ap.parse_args()

    from mobilia.analysis.segmentation import SegmentationResult
    from mobilia.photo import load_photo
    from mobilia.removal import instances as ri

    folder = Path(a.folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    cache = Path(a.cache) if a.cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    proc = model = None
    rows = []
    conf = {("B", True): 0, ("B", False): 0, ("A", True): 0, ("A", False): 0}
    for e in manifest:
        f = folder / e["file"]
        if not f.exists() or "furnished" not in e:
            continue
        rgb = load_photo(f).rgb8
        stem = f.stem
        t = time.perf_counter()
        if cache and (cache / f"{stem}.seg.npz").exists():
            seg = SegmentationResult.load(cache / f"{stem}.seg.npz")
            pan = ri.PanopticResult.load(cache / f"{stem}.pan.npz")
        else:
            if model is None:
                proc, model = ri.load_oneformer("cpu")
            seg, pan = ri.oneformer_infer(proc, model, rgb)
            if cache:
                seg.save(cache / f"{stem}.seg.npz")
                pan.save(cache / f"{stem}.pan.npz")
        t_seg = time.perf_counter() - t
        if seg.labels.shape != rgb.shape[:2]:
            raise SystemExit(f"{f.name}: cache com resolução diferente da foto")
        d = ri.detect_mode(seg, pan)
        row = {"foto": e["file"], "mobiliada": bool(e["furnished"]), "modo": d.mode, "incerto": d.uncertain,
               "motivo": d.reason, **d.features, "seg_s": round(t_seg, 1)}
        if a.sam:
            t = time.perf_counter()
            s = ri.detect_instances(rgb, seg, pan, sam="auto")
            rep = s.report()
            row.update(n_soltos=rep["n_soltos"], n_planejados=rep["n_planejados"],
                       area_remocao_pct=rep["area_remocao_pct"], sam_s=round(time.perf_counter() - t, 1),
                       planejados=[p["rotulo"] for p in rep["pecas"] if p["tipo"] == "planejado"])
        conf[(d.mode, bool(e["furnished"]))] += 1
        ok = (d.mode == "B") == bool(e["furnished"])
        print(f"{'ok ' if ok else 'ERR'} {e['file']:<36} {d.mode}{'?' if d.uncertain else ' '} "
              f"area={100 * d.features['area_removivel']:5.1f}% sig={d.features['pecas_significativas']:2d} "
              f"maior={d.features['rotulo_maior_peca']}", flush=True)
        rows.append(row)
    n = sum(conf.values())
    acc = (conf[("B", True)] + conf[("A", False)]) / max(1, n)
    print(f"\nconfusão (linhas = manifest, colunas = detectado), {n} fotos, acerto {100 * acc:.1f}%")
    print(f"  mobiliada: B={conf[('B', True)]:3d}  A={conf[('A', True)]:3d}")
    print(f"  vazia:     B={conf[('B', False)]:3d}  A={conf[('A', False)]:3d}")
    if a.json:
        Path(a.json).write_text(json.dumps({"fotos": rows, "confusao": {f"{k[0]}|{k[1]}": v for k, v in conf.items()}},
                                           indent=1, ensure_ascii=False, default=lambda o: o.item()
                                           if isinstance(o, np.generic) else str(o)), encoding="utf-8")


if __name__ == "__main__":
    main()
