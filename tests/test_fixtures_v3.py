"""PT-V3-0: fixtures das salas mobiliadas (modo B), licenças e regra absoluta reescrita.

Rápidos: carregam as análises cacheadas (sem modelos) e conferem coerência, licença e tempo de carga.
Lentos (``slow`` + ``models``): refazem a segmentação com o OneFormer e comparam com o cache.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import FURNISHED_ROOMS, ROOMS_DIR, load_room_fixture

ROOT = Path(__file__).resolve().parents[1]
LICENCAS = ROOT / "docs" / "LICENCAS_V3.md"

# Modelos com licença comercial conferida (PLANO_V3 §5.1, docs/LICENCAS_V3.md).
ALLOWED_REPOS = (
    "Ruicheng/moge-2-vitl-normal", "Ruicheng/moge-2-vitb-normal", "shi-labs/oneformer_ade20k_swin_large",
    "facebook/sam2.1-hiera-base-plus", "smartywu/big-lama", "ZhengPeng7/BiRefNet",
    "IDEA-Research/grounding-dino-tiny",
)
# Repositórios não comerciais que nunca podem aparecer no código do produto.
FORBIDDEN_REPOS = (
    "black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-Fill-dev", "black-forest-labs/FLUX.1-Kontext-dev",
    "stabilityai/sdxl-turbo", "tianweiy/DMD2", "briaai/RMBG-2.0", "pq-yang/MatAnyone", "jixinya/ObjectClear",
    "stabilityai/sv3d", "stabilityai/stable-zero123", "depth-anything/Depth-Anything-V2-Large",
)
FREE_LICENSES = ("CC0 1.0", "CC BY 2.0", "CC BY 4.0", "CC BY-SA 3.0", "CC BY-SA 4.0", "domínio público")


def _furniture_frac(fx) -> float:
    return fx.seg.fraction("furniture")


# ----------------------------------------------------------------------------- fixtures e licenças

@pytest.mark.parametrize("name", FURNISHED_ROOMS)
def test_source_md_has_url_and_commercial_license(name):
    d = ROOMS_DIR / name
    src = (d / "SOURCE.md").read_text(encoding="utf-8")
    assert re.search(r"https://\S+", src), "SOURCE.md sem URL de origem"
    assert any(lic in src for lic in FREE_LICENSES), "SOURCE.md sem licença comercial reconhecida"
    assert "creativecommons.org" in src
    im = Image.open(d / "photo.jpg")
    assert im.format == "JPEG" and max(im.size) <= 1280
    assert not dict(im.getexif()), "a foto de fixture não deve carregar EXIF"


def test_every_photo_fixture_folder_has_source():
    for d in ROOMS_DIR.iterdir():
        if (d / "photo.jpg").exists():
            assert (d / "SOURCE.md").exists(), d.name


def test_fixtures_load_fast():
    t = time.perf_counter()
    for name in FURNISHED_ROOMS:
        load_room_fixture(name)
    assert time.perf_counter() - t < 1.0


@pytest.mark.parametrize("name", FURNISHED_ROOMS)
def test_fixture_arrays_are_consistent(room_fixture, name):
    fx = room_fixture(name)
    h, w = fx.photo.shape[:2]
    assert (fx.meta["largura"], fx.meta["altura"]) == (w, h)
    assert fx.seg.labels.shape == fx.seg.groups.shape == (h, w)
    assert fx.panoptic_ids.shape == (h, w) and fx.panoptic_ids.dtype == np.uint8
    n = len(fx.panoptic_segments)
    assert n >= 3 and int(fx.panoptic_ids.max()) == n
    for k, _lid, score, area in fx.panoptic_segments:
        assert 0.0 < score <= 1.0
        assert int(area) == int(fx.panoptic_mask(int(k)).sum())
    s = fx.geometry["stride"]
    assert fx.geometry["points"].shape == (-(-h // s), -(-w // s), 3)
    assert fx.geometry["intrinsics"]["width"] == w and fx.geometry["intrinsics"]["height"] == h
    assert fx.room.intrinsics["width"] == w and fx.room.intrinsics["height"] == h
    # Pontos válidos à frente da câmera (referencial OpenCV, z > 0), a poucos metros.
    z = fx.geometry["points"][..., 2].astype(np.float32)[fx.geometry["mask"]]
    assert fx.geometry["mask"].mean() > 0.5 and np.median(z) > 0.5 and np.median(z) < 20


@pytest.mark.parametrize("name", FURNISHED_ROOMS)
def test_room_model_is_plausible(room_fixture, name):
    fx = room_fixture(name)
    assert 0.6 < fx.room.camera_height < 2.6
    assert 2.0 < fx.room.ceiling_height < 4.5
    assert fx.seg.fraction("floor") > 0.03
    assert fx.seg.fraction("wall") > 0.05


def test_furnished_vs_empty_furniture_fraction(room_fixture):
    """Base da detecção automática do modo: fotos mobiliadas têm muito mais "furniture" que a vazia."""
    from mobilia.analysis.segmentation import SegmentationResult

    empty = SegmentationResult.load(ROOMS_DIR / "stafford" / "seg.npz").fraction("furniture")
    for name in FURNISHED_ROOMS:
        assert _furniture_frac(room_fixture(name)) > max(0.08, 4 * empty), name


def test_typical_room_has_sofa_and_table(room_fixture):
    labels = room_fixture("sala_sofas_brancos").panoptic_labels()
    assert "sofa" in labels
    assert any(lb in labels for lb in ("table", "coffee table"))


def test_nice_room_has_dining_table_and_chairs(room_fixture):
    labels = room_fixture("nice_terrazzo").panoptic_labels()
    assert "table" in labels
    assert labels.count("chair") + labels.count("armchair") >= 2


def test_licencas_v3_lists_every_model_with_link():
    txt = LICENCAS.read_text(encoding="utf-8")
    for repo in ALLOWED_REPOS:
        row = next((ln for ln in txt.splitlines() if f"`{repo}`" in ln), None)
        assert row is not None, repo
        assert re.search(r"\b(MIT|Apache-2\.0)\b", row), repo
        assert f"https://huggingface.co/{repo}" in row, repo
    for name in ("OpenRAIL++", "Attachment A", "MIT", "Apache"):
        assert name in txt
    allowed_part = txt.split("## 2. Proibidos")[0]
    for bad in ("RMBG-2.0", "MatAnyone", "ObjectClear", "SDXL-Turbo", "DMD2", "pix2gestalt", "FLUX.1-dev"):
        assert bad not in allowed_part, bad


def test_product_uses_only_allowed_models():
    from mobilia.analysis.weights import model_repos

    assert set(model_repos()) <= set(ALLOWED_REPOS)
    code = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "mobilia").rglob("*.py"))
    for repo in FORBIDDEN_REPOS:
        assert repo not in code, repo


def test_arquitetura_rule_rewritten_for_mode_b():
    txt = (ROOT / "docs" / "ARQUITETURA.md").read_text(encoding="utf-8")
    sec1 = txt.split("## 1.")[1].split("## 2.")[0]
    assert "Modo B" in sec1 and "Modo A" in sec1
    assert "byte a byte" in sec1 and "np.array_equal" in sec1
    assert "pixels gerados" in sec1.lower()
    assert "não** usa difusão/inpainting sobre a foto" not in sec1  # regra antiga, só do modo A


# ----------------------------------------------------------------------------- lentos (modelos)

@pytest.mark.slow
@pytest.mark.models
def test_cached_segmentation_matches_model(room_fixture):
    """Refaz a segmentação semântica (OneFormer, CPU) e compara com o cache: o cache não pode estar velho."""
    from mobilia.analysis.segmentation import segment

    fx = room_fixture("sala_sofas_brancos")
    seg = segment(fx.photo, model_id=fx.meta["modelo_segmentacao"], device="cpu")
    assert (seg.groups == fx.seg.groups).mean() > 0.97
