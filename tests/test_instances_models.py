"""PT-V3-2 (lentos): instâncias com o OneFormer e o SAM 2.1 de verdade, na CPU."""
from pathlib import Path

import numpy as np
import pytest

from mobilia.analysis import weights
from mobilia.removal import instances as ri
from test_instances import DINING_TABLE, NICE_PROTOTYPE, _box_iou, _contains

pytestmark = [pytest.mark.slow, pytest.mark.models]
FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def oneformer():
    try:
        return ri.load_oneformer("cpu")
    except Exception as e:  # pesos ausentes / sem rede
        pytest.skip(f"OneFormer indisponível: {e}")


def test_nice_with_sam(oneformer, room_fixture):
    try:
        weights.sam2_dir(offline=True)
    except weights.WeightsError as e:
        pytest.skip(str(e))
    f = room_fixture("nice_terrazzo")
    seg, pan = ri.oneformer_infer(*oneformer, f.photo)
    s = ri.detect_instances(f.photo, seg, pan, sam="auto", geometry=f.geometry, room=f.room)
    assert not s.warnings
    boxes = [(i.label, i.bbox) for i in s.instances]
    found = sum(any(l == lab and (_box_iou(b, bb) >= 0.5 or _contains(bb, b)) for l, bb in boxes)
                for lab, b in NICE_PROTOTYPE)
    tables = [i for i in s.instances if i.label == "table" and _box_iou(i.bbox, DINING_TABLE) >= 0.8]
    assert len(tables) == 1
    found += len(tables[0].merged_from)
    assert found >= 16, boxes
    assert sum(i.label == "rug" for i in s.instances) == 2
    assert sum(i.sam_used for i in s.instances) >= 14
    assert s.mode.mode == "B"


def test_empty_stafford_photo_is_mode_a(oneformer):
    from mobilia.photo import load_photo

    rgb = load_photo(FIX / "photos" / "sala_vazia_stafford.jpg", max_long_edge=1024).rgb8
    seg, pan = ri.oneformer_infer(*oneformer, rgb)
    d = ri.detect_mode(seg, pan)
    assert d.mode == "A", d.reason


def test_run_oneformer_matches_fixture_panoptic(room_fixture):
    f = room_fixture("sala_sofas_brancos")
    try:
        seg, pan = ri.run_oneformer(f.photo)
    except Exception as e:
        pytest.skip(f"OneFormer indisponível: {e}")
    assert seg.labels.shape == pan.ids.shape == f.photo.shape[:2]
    # mesma segmentação semântica da fixture (mesmo modelo, mesma foto) em quase todos os pixels
    assert (seg.labels == f.seg.labels).mean() > 0.98
    assert ri.detect_mode(seg, pan).mode == "B"
    assert np.isin(pan.segments[:, 1], [23, 64]).sum() >= 2   # sofá e mesa de centro no panoptic
