"""Download e cache de modelos 3D e texturas da Poly Haven (licença CC0 — uso comercial livre).

Os arquivos glTF (+ texturas) são baixados uma única vez para
``<DATA_DIR>/assets/polyhaven/<id>/<res>/`` e as texturas de tecido (tapete) para
``<DATA_DIR>/assets/polyhaven/textures/<id>/<res>/``, verificados por MD5 (fornecido pela API).
O ``manifest.json`` de cada pasta registra a origem (URL da Poly Haven) e a licença. Depois disso
o software funciona offline.
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

# Tempo gasto na REDE (consultas à API + downloads), para a telemetria separar "download" de
# "ativos": com o cache completo fica em 0. Zerado pelo pipeline no início de cada trabalho.
NET_STATS = {"s": 0.0, "bytes": 0, "arquivos": 0, "consultas": 0}


def reset_net_stats() -> dict:
    NET_STATS.update(s=0.0, bytes=0, arquivos=0, consultas=0)
    return NET_STATS


def _get_json(url: str, retries: int = 4):
    last = None
    t0 = time.perf_counter()
    try:
        for i in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=60) as r:
                    NET_STATS["consultas"] += 1
                    return json.load(r)
            except Exception as e:  # rede instável: tenta de novo com espera crescente
                last = e
                time.sleep(1.5 * (i + 1))
    finally:
        NET_STATS["s"] += time.perf_counter() - t0
    raise RuntimeError(f"Falha ao acessar {url}: {last}")


def _download(url: str, dest: Path, md5: str | None, retries: int = 4) -> None:
    if dest.exists() and (md5 is None or _md5(dest) == md5):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    t0 = time.perf_counter()
    try:
        for i in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        NET_STATS["bytes"] += len(chunk)
                if md5 and _md5(tmp) != md5:
                    raise IOError(f"MD5 não confere para {url}")
                os.replace(tmp, dest)
                NET_STATS["arquivos"] += 1
                return
            except Exception as e:
                last = e
                time.sleep(2.0 * (i + 1))
    finally:
        NET_STATS["s"] += time.perf_counter() - t0
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

    from ..layout.catalog import RUG_TEXTURES

    ids = sorted({a.asset_id[3:] for a in CATALOG if a.asset_id.startswith("ph:")})
    tex_ids = sorted({t.tex_id for t in RUG_TEXTURES.values()})
    failed = []
    total = (len(ids) + len(tex_ids)) * len(resolutions)
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
    for tid in tex_ids:
        for res in resolutions:
            k += 1
            try:
                ensure_texture(tid, cache_dir, res)
            except Exception:
                failed.append(f"textura:{tid}@{res}")
            if progress:
                progress(k / total, tid)
    return failed


# Mapas de textura usados (nome na API -> chave local) e formato
TEXTURE_MAPS = {"Diffuse": "diff", "nor_gl": "nor", "Rough": "rough"}


def cached_texture(tex_id: str, cache_dir: Path, res: str = "1k") -> dict[str, Path] | None:
    """Mapas da textura já completos no cache em ``res`` (sem rede); ``None`` se faltar algo."""
    base = Path(cache_dir) / "polyhaven" / "textures" / tex_id / res
    manifest = base / "manifest.json"
    if not manifest.exists():
        return None
    try:
        m = json.loads(manifest.read_text(encoding="utf-8"))
        paths = {k: base / rel for k, rel in m["maps"].items()}
    except (ValueError, KeyError, AttributeError):
        return None
    return paths if paths and all(p.exists() for p in paths.values()) else None


def ensure_texture(tex_id: str, cache_dir: Path, res: str = "1k", retries: int = 4) -> dict[str, Path]:
    """Garante no cache os mapas JPG (difuso, normal OpenGL, rugosidade) de uma textura da Poly
    Haven e devolve {"diff": caminho, "nor": caminho, "rough": caminho}. ``retries`` = tentativas
    de rede (o render usa 1: a textura é opcional e não pode custar ~15 s de espera offline)."""
    base = Path(cache_dir) / "polyhaven" / "textures" / tex_id / res
    manifest = base / "manifest.json"
    paths = cached_texture(tex_id, cache_dir, res)
    if paths is not None:
        return paths
    files = _get_json(f"{API}/files/{tex_id}", retries=retries)
    maps, urls = {}, {}
    for api_name, key in TEXTURE_MAPS.items():
        entry = files.get(api_name, {})
        r = res if res in entry else ("1k" if "1k" in entry else (sorted(entry)[0] if entry else None))
        if r is None or "jpg" not in entry[r]:
            raise RuntimeError(f"Textura {tex_id} sem mapa {api_name} em JPG")
        info = entry[r]["jpg"]
        name = Path(info["url"]).name
        _download(info["url"], base / name, info.get("md5"), retries=retries)
        maps[key] = name
        urls[key] = info["url"]
    manifest.write_text(json.dumps({"id": tex_id, "res": res, "maps": maps, "origem": urls,
                                    "fonte": f"https://polyhaven.com/a/{tex_id}", "license": "CC0"}),
                        encoding="utf-8")
    return {k: base / v for k, v in maps.items()}


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
