"""Fixtures compartilhadas: salas de teste já analisadas (``tests/fixtures/rooms/<nome>/``).

As salas mobiliadas do modo B (``sala_sofas_brancos``, ``nice_terrazzo``) trazem a foto, a segmentação
semântica e panóptica, a geometria reduzida e o modelo da sala, gerados uma vez por
``scripts/gerar_fixtures_salas.py``. Assim os testes rápidos rodam sem nenhum modelo de IA.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOMS_DIR = Path(__file__).parent / "fixtures" / "rooms"
FURNISHED_ROOMS = ("sala_sofas_brancos", "nice_terrazzo")


@dataclass
class RoomFixture:
    name: str
    folder: Path
    photo: np.ndarray            # (H, W, 3) uint8, exatamente os bytes decodificados de photo.jpg
    seg: object                  # mobilia.analysis.segmentation.SegmentationResult
    panoptic_ids: np.ndarray     # (H, W) uint8, 0 = sem segmento, k = linha k-1 de panoptic_segments
    panoptic_segments: np.ndarray  # (N, 4): [k, label_id ADE20K, score, área em px]
    geometry: dict               # points/normals (float16, passo ``stride``), mask, intrinsics, stride
    room: object                 # mobilia.analysis.room.RoomModel
    grid: np.ndarray
    meta: dict

    def panoptic_mask(self, k: int) -> np.ndarray:
        return self.panoptic_ids == k

    def panoptic_labels(self) -> list[str]:
        names = self.seg.id2label
        return [names.get(int(r[1]), "?") for r in self.panoptic_segments]


def load_room_fixture(name: str) -> RoomFixture:
    from PIL import Image

    from mobilia.analysis.room import RoomModel
    from mobilia.analysis.segmentation import SegmentationResult

    d = ROOMS_DIR / name
    photo = np.asarray(Image.open(d / "photo.jpg").convert("RGB"))
    pan = np.load(d / "panoptic.npz", allow_pickle=False)
    g = np.load(d / "geometry_small.npz", allow_pickle=False)
    w, h, fx, fy, cx, cy = g["intrinsics"].tolist()
    geometry = {"points": g["points"], "normals": g["normals"], "mask": g["mask"], "stride": int(g["stride"]),
                "intrinsics": {"width": int(w), "height": int(h), "fx": fx, "fy": fy, "cx": cx, "cy": cy},
                "model_id": str(g["model_id"])}
    return RoomFixture(
        name=name, folder=d, photo=photo, seg=SegmentationResult.load(d / "seg.npz"),
        panoptic_ids=pan["ids"], panoptic_segments=pan["segments"], geometry=geometry,
        room=RoomModel.from_json((d / "room.json").read_text(encoding="utf-8")),
        grid=np.load(d / "grid.npy"), meta=json.loads((d / "meta.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="session")
def room_fixture():
    """Fábrica com cache: ``room_fixture("nice_terrazzo")`` devolve um :class:`RoomFixture`."""
    cache: dict[str, RoomFixture] = {}

    def get(name: str) -> RoomFixture:
        if name not in cache:
            cache[name] = load_room_fixture(name)
        return cache[name]

    return get


@pytest.fixture(autouse=True)
def _placa_geometria_padrao(monkeypatch):
    """Nos testes, o modo B usa a sala vazia pela geometria (o padrão do produto é a sala vazia por IA quando o
    gerador está instalado: numa máquina de desenvolvimento com os pesos, os testes carregariam o gerador). Os
    testes da sala vazia "ia" pedem ``JobRequest(plate="ia")`` com um gerador falso."""
    monkeypatch.setenv("MOBILIA_PLACA", "geometria")
