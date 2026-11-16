"""Pesos dos modelos de IA no cache local do Hugging Face (instalador e telemetria).

O instalador baixa tudo de uma vez (``prefetch_models``). Durante um trabalho, a análise chama
``ensure_local`` antes de carregar cada modelo: se os arquivos já estão no disco o custo é só a
consulta ao cache (milissegundos, sem rede); se faltar algo, o tempo de REDE fica separado da
carga do modelo e entra em ``tempos_s.download`` (custo de primeira execução, não do render).
"""
from __future__ import annotations

import time
from collections.abc import Callable

from ..config import GEOMETRY_MODEL_ID, GEOMETRY_MODEL_ID_SMALL, SEGMENTATION_MODEL_ID

PATTERNS = ["*.json", "*.pt", "*.bin", "*.safetensors", "*.txt"]
# Metadados das classes do ADE20K que o ``OneFormerProcessor`` baixa à parte, de um repositório
# de dados (sem ele a primeira segmentação offline falha).
ONEFORMER_METADATA = ("shi-labs/oneformer_demo", "ade20k_panoptic.json")

# Tempo de rede acumulado neste processo (zerado pelo pipeline no início de cada trabalho).
NET_STATS = {"s": 0.0, "repos": 0}


def reset_net_stats() -> dict:
    NET_STATS.update(s=0.0, repos=0)
    return NET_STATS


def model_repos() -> list[str]:
    return [GEOMETRY_MODEL_ID, GEOMETRY_MODEL_ID_SMALL, SEGMENTATION_MODEL_ID]


def _cached(repo_id: str, **kw) -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id, local_files_only=True, **kw)
        return True
    except Exception:  # LocalEntryNotFoundError e afins: não está (todo) no cache
        return False


def _metadata_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    repo, fname = ONEFORMER_METADATA
    try:
        return isinstance(try_to_load_from_cache(repo, fname, repo_type="dataset"), str)
    except Exception:
        return False


def ensure_local(repo_id: str) -> float:
    """Garante os pesos de ``repo_id`` no cache; devolve os segundos gastos na rede (0 se já estavam).

    Para o OneFormer também garante o arquivo de metadados das classes."""
    from huggingface_hub import hf_hub_download, snapshot_download

    t0 = time.perf_counter()
    net = False
    if not _cached(repo_id, allow_patterns=PATTERNS):
        snapshot_download(repo_id, allow_patterns=PATTERNS)
        net = True
    if repo_id == SEGMENTATION_MODEL_ID and not _metadata_cached():
        hf_hub_download(ONEFORMER_METADATA[0], ONEFORMER_METADATA[1], repo_type="dataset")
        net = True
    if not net:
        return 0.0
    dt = time.perf_counter() - t0
    NET_STATS["s"] += dt
    NET_STATS["repos"] += 1
    return dt


def prefetch_models(progress: Callable[[float, str], None] | None = None) -> None:
    """Baixa (ou confere no cache) todos os pesos usados pela análise. Idempotente."""
    from huggingface_hub import hf_hub_download, snapshot_download

    repos = model_repos()
    for i, repo in enumerate(repos):
        if progress:
            progress(i / (len(repos) + 1), repo)
        snapshot_download(repo, allow_patterns=PATTERNS)
    if progress:
        progress(len(repos) / (len(repos) + 1), ONEFORMER_METADATA[0])
    hf_hub_download(ONEFORMER_METADATA[0], ONEFORMER_METADATA[1], repo_type="dataset")
