"""PT-V3-1: pesos do modo B (SAM 2.1, big-lama) com SHA-256 e perfis de hardware (sem rede, sem GPU)."""
import hashlib
import urllib.error
from pathlib import Path

import pytest

from mobilia.analysis import weights
from mobilia.hardware import (PERFIL_A, PERFIL_B_COM_SDXL, PERFIL_B_SEM_SDXL, GpuInfo, HardwareReport,
                              choose_profile_v3, sdxl_capacity, sysmem_fallback_suspected)

GOOD = b"pesos certos " * 100
BAD = b"pesos corrompidos" * 50


def _spec(**kw) -> weights.FileWeight:
    base = dict(key="teste", filename="modelo.pt", subdir="teste", sha256=hashlib.sha256(GOOD).hexdigest(),
                size=len(GOOD), urls=("https://origem.invalid/modelo.pt",), label="modelo de teste",
                licenca="Apache-2.0")
    base.update(kw)
    return weights.FileWeight(**base)


class FakeFetch:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.urls: list[str] = []

    def __call__(self, url, dest: Path):
        self.urls.append(url)
        p = self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        if isinstance(p, BaseException):
            raise p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p)


def test_corrupt_download_is_rejected_and_downloaded_again(tmp_path, monkeypatch):
    monkeypatch.delenv(weights.ENV_MIRROR, raising=False)
    fetch = FakeFetch(BAD, GOOD)
    p = weights.ensure_file_weight(_spec(), root=tmp_path, fetch=fetch)
    assert p.read_bytes() == GOOD and len(fetch.urls) == 2
    # já conferido: nenhuma rede (o carimbo evita reler o arquivo)
    again = FakeFetch(RuntimeError("não deveria baixar"))
    assert weights.ensure_file_weight(_spec(), root=tmp_path, fetch=again) == p and again.urls == []


def test_corrupt_on_disk_is_replaced(tmp_path, monkeypatch):
    monkeypatch.delenv(weights.ENV_MIRROR, raising=False)
    dest = weights.file_weight_path(_spec(), tmp_path)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(BAD)                              # arquivo truncado/alterado de uma instalação antiga
    fetch = FakeFetch(GOOD)
    assert weights.ensure_file_weight(_spec(), root=tmp_path, fetch=fetch).read_bytes() == GOOD
    assert len(fetch.urls) == 1


def test_always_corrupt_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv(weights.ENV_MIRROR, raising=False)
    fetch = FakeFetch(BAD)
    with pytest.raises(weights.WeightsError, match="corrompido"):
        weights.ensure_file_weight(_spec(), root=tmp_path, fetch=fetch)
    assert len(fetch.urls) == 2                        # uma nova tentativa, não um laço infinito
    assert not weights.file_weight_path(_spec(), tmp_path).exists()   # nada corrompido fica no disco


@pytest.mark.parametrize("exc, text", [
    (urllib.error.HTTPError("u", 403, "Forbidden", {}, None), "recusou o acesso"),
    (urllib.error.HTTPError("u", 401, "Unauthorized", {}, None), "recusou o acesso"),
    (urllib.error.HTTPError("u", 404, "Not Found", {}, None), "não foi encontrado"),
    (urllib.error.URLError("sem rota"), "sem internet"),
])
def test_blocked_or_missing_gives_portuguese_message(tmp_path, monkeypatch, exc, text):
    monkeypatch.delenv(weights.ENV_MIRROR, raising=False)
    with pytest.raises(weights.WeightsError, match=text) as ei:
        weights.ensure_file_weight(_spec(), root=tmp_path, fetch=FakeFetch(exc))
    assert "modelo de teste" in str(ei.value)


def test_offline_missing_weight(tmp_path):
    with pytest.raises(weights.WeightsError, match="não está instalado"):
        weights.ensure_file_weight(_spec(), root=tmp_path, offline=True, fetch=FakeFetch(GOOD))


def test_mirror_is_tried_first_then_official(tmp_path, monkeypatch):
    monkeypatch.setenv(weights.ENV_MIRROR, "https://espelho.invalid/pesos/")
    fetch = FakeFetch(urllib.error.URLError("fora do ar"), GOOD)
    weights.ensure_file_weight(_spec(), root=tmp_path, fetch=fetch)
    assert fetch.urls == ["https://espelho.invalid/pesos/modelo.pt", "https://origem.invalid/modelo.pt"]


def test_hub_weight_corrupt_file_forces_new_download(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(BAD)
    calls = []

    def fake_snapshot(w, **kw):
        calls.append(kw)
        if kw.get("force_download"):
            (snap / "model.safetensors").write_bytes(GOOD)
        return snap

    monkeypatch.setattr(weights, "_hub_snapshot", fake_snapshot)
    w = weights.HubWeight("x", "org/repo", "abc", ("*.json",), {"model.safetensors": hashlib.sha256(GOOD).hexdigest()},
                          "modelo hub", "MIT")
    assert weights.ensure_hub_weight(w) == snap
    assert calls == [{"local_files_only": True}, {"force_download": True}]


def test_hub_gated_or_absent_gives_clear_message(monkeypatch):
    class GatedRepoError(Exception):
        pass

    def fake_snapshot(w, **kw):
        raise (FileNotFoundError("cache") if kw.get("local_files_only") else GatedRepoError("401 gated"))

    monkeypatch.setattr(weights, "_hub_snapshot", fake_snapshot)
    with pytest.raises(weights.WeightsError, match="recusou o acesso"):
        weights.ensure_hub_weight(weights.SAM2_WEIGHT)
    with pytest.raises(weights.WeightsError, match="não está instalado"):
        weights.ensure_hub_weight(weights.SAM2_WEIGHT, offline=True)


def test_v3_weights_are_pinned_and_allowed():
    from test_fixtures_v3 import ALLOWED_REPOS

    assert set(weights.v3_repos()) <= set(ALLOWED_REPOS)
    for w in weights.V3_WEIGHTS:
        assert w.licenca in ("MIT", "Apache-2.0")
        shas = w.sha256.values() if isinstance(w, weights.HubWeight) else [w.sha256]
        assert all(len(s) == 64 and int(s, 16) >= 0 for s in shas)
    assert len(weights.SAM2_WEIGHT.revision) == 40                  # revisão fixada, não "main"
    assert "*.pt" not in weights.SAM2_WEIGHT.allow_patterns       # sem o checkpoint duplicado (+320 MB)



def test_sdxl_weights_are_pinned_fp16_and_licensed():
    """PT-V3-4: os pesos do SDXL têm revisão fixada, SHA-256 dos arquivos grandes, só variantes fp16 (sem o
    checkpoint único de 6,9 GB nem o fp32) e licença comercial (OpenRAIL++ ou MIT)."""
    allowed = {"SG161222/RealVisXL_V5.0_Lightning", "madebyollin/sdxl-vae-fp16-fix",
               "diffusers/controlnet-depth-sdxl-1.0-small"}
    assert set(weights.sdxl_repos()) == allowed
    for w in weights.SDXL_WEIGHTS:
        assert w.licenca in ("OpenRAIL++", "MIT") and len(w.revision) == 40
        assert w.sha256 and all(len(s) == 64 and int(s, 16) >= 0 for s in w.sha256.values())
        assert set(w.sha256) <= set(w.allow_patterns)
        for pat in w.allow_patterns:
            assert "*" not in pat or pat.endswith(".json")
            if pat.endswith(".safetensors"):
                assert "fp16" in pat or w.repo_id == "madebyollin/sdxl-vae-fp16-fix", pat
    assert weights.SDXL_TEXT_WEIGHT not in weights.SDXL_REQUIRED      # só para prompts sem embedding em cache
    assert not weights.sdxl_installed(required=())


def test_sdxl_installed_is_false_when_weights_missing(monkeypatch):
    def missing(w, offline=False):
        raise weights.WeightsError("ausente")

    monkeypatch.setattr(weights, "ensure_weight", missing)
    assert not weights.sdxl_installed()


def test_prefetch_includes_v3_weights(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: "/cache")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: "/cache/x")
    got = []
    monkeypatch.setattr(weights, "ensure_weight", lambda w, offline=False: got.append(w.key))
    steps = []
    weights.prefetch_models(lambda f, name: steps.append(f))
    assert got == ["sam2", "big_lama"]
    assert steps == sorted(steps) and steps[-1] < 1.0


# ----------------------------------------------------------------------------- perfis

def _rep(vram_gb: float | None, free_gb: float | None = None, ram_gb: float = 32.0) -> HardwareReport:
    rep = HardwareReport("Windows 11", "x86", 16, ram_gb)
    if vram_gb is not None:
        rep.gpus = [GpuInfo(f"NVIDIA GeForce RTX ({vram_gb} GB)", int(vram_gb * 1024),
                            None if free_gb is None else int(free_gb * 1024), "999.0", "12.0")]
        rep.torch_cuda = True
    return rep


@pytest.mark.parametrize("vram, free, ram, cabe, offload", [
    (4, None, 32, False, False),
    (6, None, 32, False, False),     # o UNet fp16 inteiro (4,9 GiB) + ControlNet + ativações não cabe em 6 GB
    (8, None, 32, True, False),
    (8, 6.8, 32, True, True),        # 8 GB com o desktop ocupando ~1,2 GB: offload
    (8, 6.8, 8, False, False),       # offload sem RAM suficiente
    (8, 6.0, 32, False, False),      # abaixo de SDXL_VRAM_MIN_MB (6500): o UNet não sobe inteiro
    (8, 4.5, 32, False, False),
    (12, None, 16, True, False),
])
def test_sdxl_capacity_by_vram(vram, free, ram, cabe, offload):
    ok, off, motivo = sdxl_capacity(_rep(vram, free, ram))
    assert (ok, off) == (cabe, offload), motivo


@pytest.mark.parametrize("vram, esperado", [(4, PERFIL_B_SEM_SDXL), (6, PERFIL_B_SEM_SDXL),
                                            (8, PERFIL_B_COM_SDXL), (12, PERFIL_B_COM_SDXL)])
def test_profiles_4_6_8_12_gb(vram, esperado):
    rep = _rep(vram)
    # SDXL não instalado → sempre "B sem SDXL", na GPU
    p = choose_profile_v3(rep, sdxl_instalado=False)
    assert p.nome == PERFIL_B_SEM_SDXL and p.dispositivo == "cuda" and not p.sdxl and p.modo == "B"
    # com o SDXL instalado, a VRAM decide
    p = choose_profile_v3(rep, sdxl_instalado=True)
    assert p.nome == esperado and p.sdxl == (esperado == PERFIL_B_COM_SDXL)
    assert not p.sdxl_offload            # 8 e 12 GB livres: SDXL inteiro na GPU; 6 GB: sem SDXL
    # o usuário pode desligar o SDXL
    assert choose_profile_v3(rep, sdxl_instalado=True, sdxl_ligado=False).nome == PERFIL_B_SEM_SDXL
    # foto vazia: perfil A; motor v2 sem SDXL
    a = choose_profile_v3(rep, modo="A", sdxl_instalado=False)
    assert a.nome == PERFIL_A and a.motor_a == "v2" and not a.sdxl
    a = choose_profile_v3(rep, modo="A", sdxl_instalado=True)
    assert a.motor_a == ("sdxl" if esperado == PERFIL_B_COM_SDXL else "v2")


def test_profile_without_gpu_falls_back_to_cpu():
    p = choose_profile_v3(_rep(None), sdxl_instalado=True)
    assert p.nome == PERFIL_B_SEM_SDXL and p.dispositivo == "cpu" and not p.sdxl
    assert any("CPU" in a for a in p.avisos)
    with pytest.raises(ValueError):
        choose_profile_v3(_rep(8), modo="C")


def test_sysmem_fallback_probe():
    assert sysmem_fallback_suspected(5.0, 0.6)
    assert not sysmem_fallback_suspected(0.9, 0.6)
    assert not sysmem_fallback_suspected(1.0, 0.0)


def test_prefetch_cli_reports_weight_error(monkeypatch, capsys):
    import json

    from mobilia import cli

    def boom(progress=None):
        raise weights.WeightsError("Modelo de IA 'big-lama (sala vazia)' indisponível: sem internet.")

    monkeypatch.setattr(weights, "prefetch_models", boom)
    assert cli.main(["prefetch", "--json", "--modelos"]) == 2
    io = capsys.readouterr()
    ev = json.loads(io.out.strip().splitlines()[-1])
    assert ev["event"] == "log" and ev["level"] == "error" and "big-lama" in ev["message"]
    assert "indisponível" in io.err
