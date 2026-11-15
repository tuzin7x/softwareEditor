"""Download e cache de modelos 3D da Poly Haven (licença CC0 — uso comercial livre).

Os arquivos glTF (+ texturas) são baixados uma única vez para
``<DATA_DIR>/assets/polyhaven/<id>/<res>/`` e verificados por MD5 (fornecido pela API).
Depois disso o software funciona offline.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

API = "https://api.polyhaven.com"
USER_AGENT = "MobiliaStudio/0.1 (local renderer)"


def _get_json(url: str, retries: int = 4):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # rede instável: tenta de novo com espera crescente
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Falha ao acessar {url}: {last}")


def _download(url: str, dest: Path, md5: str | None, retries: int = 4) -> None:
    if dest.exists() and (md5 is None or _md5(dest) == md5):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if md5 and _md5(tmp) != md5:
                raise IOError(f"MD5 não confere para {url}")
            os.replace(tmp, dest)
            return
        except Exception as e:
            last = e
            time.sleep(2.0 * (i + 1))
    raise RuntimeError(f"Falha ao baixar {url}: {last}")


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prefetch_catalog(cache_dir: Path, resolutions: tuple[str, ...] = ("1k", "2k"), progress=None) -> list[str]:
    """Baixa todos os modelos Poly Haven do catálogo (usado no primeiro uso / instalador),
    para que o software funcione 100% offline depois. Retorna os ids com falha."""
    from ..layout.catalog import CATALOG

    ids = sorted({a.asset_id[3:] for a in CATALOG if a.asset_id.startswith("ph:")})
    failed = []
    total = len(ids) * len(resolutions)
    k = 0
    for pid in ids:
        for res in resolutions:
            k += 1
            try:
                ensure_model(pid, cache_dir, res)
            except Exception:
                failed.append(f"{pid}@{res}")
            if progress:
                progress(k / total, pid)
    return failed


def ensure_model(asset_id: str, cache_dir: Path, res: str = "1k") -> Path:
    """Garante o glTF do asset no cache e devolve o caminho do arquivo .gltf."""
    base = Path(cache_dir) / "polyhaven" / asset_id / res
    manifest = base / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        gltf = base / m["gltf"]
        if gltf.exists() and all((base / rel).exists() for rel in m["files"]):
            return gltf
    files = _get_json(f"{API}/files/{asset_id}")
    gl = files.get("gltf", {})
    if res not in gl:
        avail = sorted(gl.keys())
        if not avail:
            raise RuntimeError(f"Asset {asset_id} não tem glTF disponível")
        res = "1k" if "1k" in avail else avail[0]
        base = Path(cache_dir) / "polyhaven" / asset_id / res
    entry = gl[res]["gltf"]
    gltf_name = Path(entry["url"]).name
    _download(entry["url"], base / gltf_name, entry.get("md5"))
    rels = []
    for rel, info in entry.get("include", {}).items():
        _download(info["url"], base / rel, info.get("md5"))
        rels.append(rel)
    (base / "manifest.json").write_text(json.dumps({"gltf": gltf_name, "files": rels, "id": asset_id,
                                                    "res": res, "license": "CC0"}), encoding="utf-8")
    return base / gltf_name
