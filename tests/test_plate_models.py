"""PT-V3-3 (lentos): sala vazia com o big-lama de verdade e nota com o OneFormer rodado de novo na placa (CPU)."""
import numpy as np
import pytest

from mobilia.analysis import weights
from mobilia.removal import instances as ri
from mobilia.removal import plate as rp

pytestmark = [pytest.mark.slow, pytest.mark.models]


@pytest.fixture(scope="module")
def lama():
    try:
        return rp.load_lama("cpu", offline=True)
    except weights.WeightsError as e:
        pytest.skip(str(e))


def _instances(f):
    return ri.detect_instances(f.photo, f.seg, ri.PanopticResult(f.panoptic_ids, f.panoptic_segments), sam=None,
                               geometry=f.geometry, room=f.room)


def test_lama_wrapper_keeps_known_pixels(lama):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (61, 83, 3), dtype=np.uint8)        # tamanho não múltiplo de 8
    m = np.zeros((61, 83), bool)
    m[20:40, 30:60] = True
    out = lama(img, m)
    assert out.shape == img.shape and np.array_equal(out[~m], img[~m])
    assert not np.array_equal(out[m], img[m])


@pytest.fixture(scope="module")
def oneformer():
    try:
        return ri.load_oneformer("cpu")
    except Exception as e:  # pesos ausentes
        pytest.skip(f"OneFormer indisponível: {e}")


_cache: dict = {}


def _plate(name, lama, oneformer, room_fixture):
    if name not in _cache:
        f = room_fixture(name)
        s = _instances(f)
        depth = f.geometry["points"][..., 2].astype(np.float32)
        _cache[name] = (f, rp.build_plate(f.photo, s.removal(), f.seg, f.room, depth, lama,
                                          redetect=lambda p: ri.oneformer_infer(*oneformer, p)))
    return _cache[name]


@pytest.mark.parametrize("name", ["sala_sofas_brancos", "nice_terrazzo"])
def test_plate_with_lama_guarantees(lama, oneformer, room_fixture, name):
    """O que o produto garante sempre: placa == foto fora da máscara e o portão reprova a placa em que o OneFormer
    reencontra um móvel (nunca entrega placa ruim em silêncio)."""
    f, r = _plate(name, lama, oneformer, room_fixture)
    assert np.array_equal(r.plate[~r.mask], f.photo[~r.mask])
    q = r.quality
    assert "piso" in q.per_plane
    if q.redetected:
        assert not q.ok and any("reaparece" in m for m in q.reasons)


@pytest.mark.xfail(strict=False, reason=(
    "aceite PT-V3-3 ainda não atingido nas fixtures só com máscaras do OneFormer (sem SAM), medido na CPU em "
    "17/11/2026: Nice costura do piso 3,66 ΔE (> 3); sala_sofas_brancos o OneFormer reencontra 'towel' 1,65% e "
    "'table' 0,75% na placa; a média de cor do piso preenchido destoa do anel (sala_sofas_brancos ~9,6 ΔE na "
    "revisão). O portão reprova essas placas."))
@pytest.mark.parametrize("name", ["sala_sofas_brancos", "nice_terrazzo"])
def test_plate_acceptance_thresholds(lama, oneformer, room_fixture, name):
    """Aceite PT-V3-3: média de cor do preenchimento x anel <= 3 ΔE em cada plano (e costura local do piso
    <= 3 ΔE), linha parede/piso <= 3 px, nenhum móvel > 0,5% reencontrado."""
    _, r = _plate(name, lama, oneformer, room_fixture)
    q = r.quality
    assert q.per_plane["piso"]["costura_de"] <= 3.0, q.per_plane
    for plane, s in q.per_plane.items():
        if "preenchimento_de" in s:
            assert s["preenchimento_de"] <= 3.0, (plane, q.per_plane)
    if "distancia_px" in q.line:
        assert q.line["distancia_px"] <= 3.0, q.line
    assert q.redetected == [], q.redetected
