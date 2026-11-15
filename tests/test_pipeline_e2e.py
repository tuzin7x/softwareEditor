"""Teste ponta a ponta REAL (modelos de IA + Blender + FFmpeg) em resolução mínima.

Lento (minutos em CPU) e precisa dos pesos dos modelos: marcado como slow/models.
Executar com:  python -m pytest -m "slow and models" tests/test_pipeline_e2e.py
"""
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.models]

FX = Path(__file__).parent / "fixtures"


def test_full_pipeline_preserves_photo(tmp_path, monkeypatch):
    from mobilia import pipeline
    from mobilia.jobs import JobRequest, ProgressEmitter

    # resolução mínima para o teste caber em CPU
    orig_quality = pipeline._quality

    def tiny_quality(req, prof):
        q = orig_quality(req, prof)
        q.update(long_edge=320, samples=8, fps=12)
        return q

    monkeypatch.setattr(pipeline, "_quality", tiny_quality)
    events = []
    em = ProgressEmitter(callback=events.append)
    out = tmp_path / "video.mp4"
    req = JobRequest(str(FX / "photos/sala_vazia_stafford.jpg"), "sala", str(out), quality="previa",
                     work_dir=str(tmp_path / "job"))
    res = pipeline.run_job(req, em)
    assert out.exists() and out.stat().st_size > 10_000
    rep = json.loads(Path(res.report_path).read_text(encoding="utf-8"))
    assert rep["preservacao"]["todos_bit_exatos_fora_dos_objetos"] is True
    assert rep["preservacao"]["quadros_identicos_a_foto"] >= 1       # o início do vídeo é a foto pura
    assert len(rep["layout"]["items"]) >= 4
    fracs = [e["fraction"] for e in events if e["event"] == "progress"]
    assert fracs == sorted(fracs) or max(np.diff(fracs)) > -0.05        # progresso essencialmente monotônico
    assert rep["encode"]["verificacao"][0]["psnr_fundo_db"] > 35
