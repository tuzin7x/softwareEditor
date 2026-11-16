"""Interface (offscreen): estados da janela e protocolo com o worker (usando um dublê de teste)."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

FX = Path(__file__).parent / "fixtures/photos/sala_vazia_stafford.jpg"
SHOTS = Path(os.environ.get("MOBILIA_UI_SHOTS", "/tmp/claude-0/ui"))


@pytest.fixture(scope="module")
def app():
    from mobilia.ui import theme
    a = QApplication.instance() or QApplication([])
    a.setStyle("Fusion"); a.setStyleSheet(theme.QSS)
    return a


@pytest.fixture()
def win(app, tmp_path, monkeypatch):
    monkeypatch.setenv("MOBILIA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MOBILIA_WORKER", json.dumps([sys.executable, str(Path(__file__).parent / "fake_worker.py")]))
    from mobilia.ui.app import MainWindow
    w = MainWindow(); w.resize(1360, 820); w.show(); app.processEvents()
    yield w
    w.close()


def _wait(app, cond, timeout=20):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents(); time.sleep(0.02)
    return cond()


def _shot(w, name):
    SHOTS.mkdir(parents=True, exist_ok=True)
    w.grab().save(str(SHOTS / f"{name}.png"))


def test_flow_success(app, win):
    _shot(win, "1_vazio")
    assert not win.btn_generate.isEnabled()
    win.load_photo(str(FX)); win.select_room("sala"); app.processEvents()
    assert win.btn_generate.isEnabled()
    _shot(win, "2_foto_e_ambiente")
    win.generate(); app.processEvents()
    assert win.right.currentIndex() == 1
    _shot(win, "3_gerando")
    assert _wait(app, lambda: win.right.currentIndex() == 2)
    assert win.last_output and Path(win.last_output).exists()
    _shot(win, "4_pronto")


def test_flow_error_returns_to_setup(app, win, monkeypatch):
    shown = {}
    monkeypatch.setattr("mobilia.ui.app.MainWindow.show_error", lambda self, m, d="": shown.update(m=m, d=d))
    win.load_photo(str(FX)); win.select_room("banheiro")
    win.generate()
    assert _wait(app, lambda: "m" in shown)
    assert shown["m"] == "Falha simulada de teste" and win.right.currentIndex() == 0


def test_hardware_probe_updates_status(app, win):
    win.probe.start()
    assert _wait(app, lambda: "RTX 4070" in win.statusBar().currentMessage())


def test_selftest_entrypoint():
    from mobilia.ui.app import main
    assert main(["--selftest"]) == 0
