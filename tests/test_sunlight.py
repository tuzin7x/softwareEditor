"""Sol direto: a direção estimada deve explicar as manchas de sol reais da foto de teste."""
from pathlib import Path

import numpy as np

from mobilia.analysis.room import RoomModel
from mobilia.analysis.segmentation import SegmentationResult, labels_to_groups
from mobilia.analysis.sunlight import detect_sun
from mobilia.photo import load_photo

FX = Path(__file__).parent / "fixtures"


def _fixture():
    room = RoomModel.from_json((FX / "rooms/stafford/room.json").read_text())
    seg = SegmentationResult.load(FX / "rooms/stafford/seg.npz")
    seg.groups = labels_to_groups(seg.labels, seg.id2label)
    photo = load_photo(FX / "photos/sala_vazia_stafford.jpg").rgb8
    return room, seg, photo


def test_sun_detected_through_left_windows_and_bay():
    room, seg, photo = _fixture()
    est = detect_sun(room, seg, photo)
    assert est is not None
    d = np.asarray(est.direction)
    assert abs(np.linalg.norm(d) - 1) < 1e-6
    # o sol entra pelas janelas da parede esquerda (normal +x) => vem de -x
    assert d[0] < -0.2 and 35 < est.elevation_deg < 70
    assert est.precision > 0.8
    assert {0, 1} <= set(est.windows)


def test_no_sun_without_bright_patches():
    room, seg, photo = _fixture()
    flat = photo.copy()
    floor = seg.mask("floor")
    flat[floor] = np.median(photo[floor], axis=0).astype(np.uint8)    # piso uniforme: sem manchas
    assert detect_sun(room, seg, flat) is None
