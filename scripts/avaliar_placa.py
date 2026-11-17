"""Avalia a sala vazia do modo B (PT-V3-3) num conjunto de fotos mobiliadas: nota, motivos e imagem lado a lado.

Uso (precisa de MoGe-2, OneFormer, SAM 2.1 e big-lama no cache; CPU ~1–2 min por foto)::

    python scripts/avaliar_placa.py <pasta_com_manifest.json> <pasta_saida> [--so nome1,nome2] [--max 1024]

Para cada foto mobiliada do ``manifest.json`` grava ``<nome>_placa.jpg`` (foto | placa) e, no fim,
``placas.json`` com o relatório de cada uma (planos, efeitos, nota, motivos do portão, avisos). Serve para
ajustar os limites de ``mobilia.removal.plate.LIMITS`` nas fotos reais dos clientes (não versionadas).
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
    ap.add_argument("out")
    ap.add_argument("--so", help="só estas fotos (nomes sem extensão, separados por vírgula)")
    ap.add_argument("--max", type=int, default=1024, help="lado maior de trabalho")
    a = ap.parse_args()

    from PIL import Image

    from mobilia.analysis.geometry import estimate_geometry
    from mobilia.analysis.room import build_room_model
    from mobilia.photo import load_photo
    from mobilia.removal import instances as ri
    from mobilia.removal import plate as rp

    folder, out = Path(a.folder), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    only = set(a.so.split(",")) if a.so else None
    oneformer = ri.load_oneformer("cpu")
    lama = rp.load_lama()
    results = {}
    for item in manifest:
        name = Path(item["file"]).stem
        if not item.get("furnished") or (only and name not in only):
            continue
        t0 = time.perf_counter()
        rgb = load_photo(folder / item["file"], max_long_edge=a.max).rgb8
        geo = estimate_geometry(rgb)
        seg, pan = ri.oneformer_infer(*oneformer, rgb)
        room, _ = build_room_model(geo, seg, rgb)
        st = 4
        geod = {"points": geo.points[::st, ::st].astype(np.float16), "mask": geo.mask[::st, ::st], "stride": st,
                "intrinsics": geo.intrinsics.to_dict()}
        inst = ri.detect_instances(rgb, seg, pan, sam="auto", geometry=geod, room=room)
        r = rp.build_plate_gated(rgb, inst, seg, room, geo.depth, lama, lambda p: ri.oneformer_infer(*oneformer, p))
        rep = r.report()["placa"]
        rep["tempo_total_cpu_s"] = round(time.perf_counter() - t0, 1)
        results[name] = rep
        Image.fromarray(np.concatenate([rgb, r.plate], 1)).save(out / f"{name}_placa.jpg", quality=88)
        q = rep["qualidade"]
        print(f"{name:32s} nota {q['nota']:5.1f} {'OK ' if q['aprovada'] else 'RUIM'} "
              f"planejados_mantidos={rep['planejados_mantidos']} {'; '.join(q['motivos'])}", flush=True)
    (out / "placas.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(r["qualidade"]["aprovada"] for r in results.values())
    print(f"aprovadas: {ok}/{len(results)}")


if __name__ == "__main__":
    main()
