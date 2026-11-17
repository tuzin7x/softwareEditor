"""Pesos dos modelos de IA no cache local do Hugging Face (instalador e telemetria).

O instalador baixa tudo de uma vez (``prefetch_models``). Durante um trabalho, a análise chama
``ensure_local`` antes de carregar cada modelo: se os arquivos já estão no disco o custo é só a
consulta ao cache (milissegundos, sem rede); se faltar algo, o tempo de REDE fica separado da
carga do modelo e entra em ``tempos_s.download`` (custo de primeira execução, não do render).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

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


def prefetch_models(progress: Callable[[float, str], None] | None = None, v3: bool = True) -> None:
    """Baixa (ou confere no cache) todos os pesos usados pela análise. Idempotente.

    ``v3``: inclui os pesos do modo B (SAM 2.1 e big-lama, com SHA-256 conferido)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    repos = model_repos()
    v3w = V3_WEIGHTS if v3 else ()
    n = len(repos) + 1 + len(v3w)
    for i, repo in enumerate(repos):
        if progress:
            progress(i / n, repo)
        snapshot_download(repo, allow_patterns=PATTERNS)
    if progress:
        progress(len(repos) / n, ONEFORMER_METADATA[0])
    hf_hub_download(ONEFORMER_METADATA[0], ONEFORMER_METADATA[1], repo_type="dataset")
    for j, w in enumerate(v3w):
        if progress:
            progress((len(repos) + 1 + j) / n, w.label)
        ensure_weight(w)


# ============================================================================ pesos do modo B (V3)
# SAM 2.1 base+ (máscara por peça) e big-lama (sala vazia). Cada arquivo tem SHA-256 fixado: um
# download corrompido ou trocado é apagado e baixado de novo UMA vez; se continuar errado, o
# trabalho para com uma mensagem clara (nunca carregamos um peso que não conferiu).
#
# Espelho configurável: ``MOBILIA_ESPELHO_PESOS`` (URL base) é tentado antes da origem oficial dos
# arquivos avulsos (``<espelho>/<nome do arquivo>``). Para os repositórios do Hugging Face vale o
# padrão da biblioteca, ``HF_ENDPOINT``. O SDXL (perfil "B com SDXL") ainda não entra aqui.

ENV_MIRROR = "MOBILIA_ESPELHO_PESOS"


class WeightsError(RuntimeError):
    """Peso ausente, bloqueado ou corrompido. A mensagem já está em português, para o usuário."""


@dataclass(frozen=True)
class HubWeight:
    """Repositório do Hugging Face fixado numa revisão; ``sha256`` = arquivos grandes conferidos."""

    key: str
    repo_id: str
    revision: str
    allow_patterns: tuple[str, ...]
    sha256: dict[str, str]
    label: str
    licenca: str


@dataclass(frozen=True)
class FileWeight:
    """Arquivo avulso (fora do Hugging Face) baixado para ``<dados>/models/<subdir>/<filename>``."""

    key: str
    filename: str
    subdir: str
    sha256: str
    size: int
    urls: tuple[str, ...]
    label: str
    licenca: str
    extra: dict = field(default_factory=dict)


SAM2_WEIGHT = HubWeight(
    key="sam2", repo_id="facebook/sam2.1-hiera-base-plus",
    revision="b7320756a13354e7530a63935656d35b2f91a290",
    # Só o formato do transformers (``Sam2Model``); o ``.pt`` original (mais 320 MB) não é usado.
    allow_patterns=("*.json", "model.safetensors"),
    sha256={"model.safetensors": "2012733a0de5d03efd1bba550a2847c4551be9ef2e0d497c83074df66189f780"},
    label="SAM 2.1 base+ (recorte das peças)", licenca="Apache-2.0")

LAMA_WEIGHT = FileWeight(
    key="big_lama", filename="big-lama.pt", subdir="lama",
    sha256="344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9", size=205669692,
    # TorchScript do big-lama (LaMa, Apache-2.0) publicado pelo projeto lama-cleaner.
    urls=("https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",),
    label="big-lama (sala vazia)", licenca="Apache-2.0")

V3_WEIGHTS: tuple = (SAM2_WEIGHT, LAMA_WEIGHT)


def v3_repos() -> list[str]:
    return [w.repo_id for w in V3_WEIGHTS if isinstance(w, HubWeight)]


def models_dir() -> Path:
    """``<dados>/models`` (mesma pasta do motor instalado; ``MOBILIA_HOME`` muda a raiz)."""
    from .. import config

    env = os.environ.get("MOBILIA_HOME")
    return (Path(env) if env else config.DATA_DIR) / "models"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stamp(path: Path) -> Path:
    return path.with_name(path.name + ".sha256ok")


def verified(path: Path, sha256: str) -> bool:
    """O arquivo existe e o SHA-256 confere. O resultado fica num carimbo (tamanho + data), para
    não reler 200–300 MB a cada trabalho; qualquer mudança no arquivo invalida o carimbo."""
    try:
        st = path.stat()
    except OSError:
        return False
    sig = {"sha256": sha256, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    stamp = _stamp(path)
    try:
        if json.loads(stamp.read_text(encoding="utf-8")) == sig:
            return True
    except (OSError, ValueError):
        pass
    if file_sha256(path) != sha256:
        return False
    try:
        stamp.write_text(json.dumps(sig), encoding="utf-8")
    except OSError:
        pass
    return True


def _forget(path: Path) -> None:
    for p in (path, _stamp(path)):
        try:
            p.unlink()
        except OSError:
            pass


def friendly_error(label: str, exc: BaseException) -> WeightsError:
    """Traduz falhas de rede/Hugging Face numa mensagem clara em português."""
    name = type(exc).__name__
    status = getattr(exc, "code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if name == "GatedRepoError" or status in (401, 403):
        why = ("o servidor recusou o acesso (modelo restrito ou bloqueado pela rede/antivírus). "
               "Libere o acesso a huggingface.co e github.com ou configure um espelho de pesos "
               f"({ENV_MIRROR}).")
    elif name in ("RepositoryNotFoundError", "RevisionNotFoundError", "EntryNotFoundError") or status == 404:
        why = "o arquivo não foi encontrado no servidor (endereço mudou). Atualize o 7xAi."
    elif name == "LocalEntryNotFoundError" or isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, OSError)):
        why = ("não foi possível baixar: sem internet ou conexão bloqueada (firewall/proxy). "
               "Conecte-se e tente de novo; depois do primeiro download tudo funciona offline.")
    else:
        why = f"erro inesperado ({name}: {str(exc)[:200]})."
    return WeightsError(f"Modelo de IA '{label}' indisponível: {why}")


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "7xAi-setup"})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(part, dest)


def _file_urls(w: FileWeight) -> list[str]:
    mirror = os.environ.get(ENV_MIRROR, "").strip().rstrip("/")
    return ([f"{mirror}/{w.filename}"] if mirror else []) + list(w.urls)


def file_weight_path(w: FileWeight, root: Path | None = None) -> Path:
    return (root or models_dir()) / w.subdir / w.filename


def ensure_file_weight(w: FileWeight, root: Path | None = None, fetch=None, offline: bool = False) -> Path:
    """Garante ``w`` no disco com o SHA-256 certo e devolve o caminho.

    Hash errado → apaga e baixa de novo (cada origem é tentada no máximo duas vezes). Sem
    nenhuma origem válida → ``WeightsError`` com a explicação em português."""
    fetch = fetch or _fetch
    dest = file_weight_path(w, root)
    if verified(dest, w.sha256):
        return dest
    _forget(dest)
    if offline:
        raise WeightsError(f"Modelo de IA '{w.label}' não está instalado. Conecte-se à internet e abra o "
                           "7xAi de novo para baixá-lo (só desta vez).")
    t0 = time.perf_counter()
    last: BaseException | None = None
    corrupt = False
    for url in _file_urls(w):
        for _ in range(2):
            try:
                fetch(url, dest)
            except Exception as e:  # rede, 403, 404...
                last = e
                _forget(dest)
                break
            if verified(dest, w.sha256):
                NET_STATS["s"] += time.perf_counter() - t0
                NET_STATS["repos"] += 1
                return dest
            corrupt = True
            _forget(dest)
    if corrupt and last is None:
        raise WeightsError(f"Modelo de IA '{w.label}': o arquivo baixado está corrompido (SHA-256 não confere) "
                           "mesmo depois de baixar de novo. Verifique a conexão ou o antivírus e tente outra vez.")
    raise friendly_error(w.label, last or RuntimeError("sem origem"))


def _hub_snapshot(w: HubWeight, **kw) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(w.repo_id, revision=w.revision, allow_patterns=list(w.allow_patterns), **kw))


def ensure_hub_weight(w: HubWeight, offline: bool = False) -> Path:
    """Garante o repositório ``w`` (revisão fixada) no cache do Hugging Face com os SHA-256 certos.

    Hash errado → baixa de novo com ``force_download``; continua errado → ``WeightsError``."""
    t0 = time.perf_counter()
    net = False
    try:
        folder = _hub_snapshot(w, local_files_only=True)
    except Exception as e:
        if offline:
            raise WeightsError(f"Modelo de IA '{w.label}' não está instalado. Conecte-se à internet e abra o "
                               "7xAi de novo para baixá-lo (só desta vez).") from e
        try:
            folder, net = _hub_snapshot(w), True
        except Exception as e2:
            raise friendly_error(w.label, e2) from e2
    for attempt in range(2):
        bad = [f for f, sha in w.sha256.items() if not verified(folder / f, sha)]
        if not bad:
            break
        if attempt == 1 or offline:
            raise WeightsError(f"Modelo de IA '{w.label}': arquivo corrompido ({', '.join(bad)}; SHA-256 não "
                               "confere) mesmo depois de baixar de novo. Verifique a conexão ou o antivírus.")
        for f in bad:
            _forget_hub_file(folder / f)
        try:
            folder, net = _hub_snapshot(w, force_download=True), True
        except Exception as e:
            raise friendly_error(w.label, e) from e
    if net:
        NET_STATS["s"] += time.perf_counter() - t0
        NET_STATS["repos"] += 1
    return folder


def _forget_hub_file(path: Path) -> None:
    """No cache do HF o arquivo é um link para o blob; apaga os dois (e o carimbo)."""
    try:
        blob = path.resolve()
        if blob != path and blob.exists():
            blob.unlink()
    except OSError:
        pass
    _forget(path)


def ensure_weight(w, offline: bool = False) -> Path:
    return ensure_hub_weight(w, offline) if isinstance(w, HubWeight) else ensure_file_weight(w, offline=offline)


def sam2_dir(offline: bool = False) -> Path:
    """Pasta local do SAM 2.1 (para ``Sam2Model.from_pretrained``)."""
    return ensure_hub_weight(SAM2_WEIGHT, offline)


def big_lama_path(offline: bool = False) -> Path:
    """Arquivo TorchScript do big-lama (para ``torch.jit.load``)."""
    return ensure_file_weight(LAMA_WEIGHT, offline=offline)


# SDXL (perfil "B com SDXL" e modo A novo): só a sinalização por enquanto (PT-V3-4/9 incluem os pesos).
SDXL_WEIGHTS: tuple = ()


def sdxl_installed() -> bool:
    """Os pesos do SDXL estão no disco e conferem. Hoje sempre falso (ainda não são baixados)."""
    if not SDXL_WEIGHTS:
        return False
    try:
        for w in SDXL_WEIGHTS:
            ensure_weight(w, offline=True)
        return True
    except WeightsError:
        return False
