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
    # o motor falhou antes de criar a pasta do trabalho: o diagnóstico (botão da tela de erro)
    # ainda sai, com a saída do motor
    import zipfile

    z = win.copy_diagnostic()
    assert z is not None and z.exists()
    zf = zipfile.ZipFile(z)
    assert "interface.log" in zf.namelist()
    assert "Falha simulada de teste" in zf.read("interface.log").decode("utf-8")


def test_hardware_probe_updates_status(app, win):
    win.probe.start()
    assert _wait(app, lambda: "RTX 4070" in win.statusBar().currentMessage())


def test_selftest_entrypoint():
    from mobilia.ui.app import main
    assert main(["--selftest"]) == 0


def test_copy_diagnostic_zip_without_photo(app, win, tmp_path):
    import zipfile

    win.load_photo(str(FX)); win.select_room("sala"); app.processEvents()
    win.generate()
    assert _wait(app, lambda: win.right.currentIndex() == 2)
    # tempo por etapa na tela de conclusão (vem do report.json do trabalho)
    assert "Render dos quadros" in win.result_times.text() and "Total" in win.result_times.text()
    assert win.btn_diag.isVisible()
    z = win.copy_diagnostic()
    assert z is not None and z.exists()
    names = zipfile.ZipFile(z).namelist()
    assert {"report.json", "render/frames.json", "interface.log", "encode.log", "manifesto.json"} <= set(names)
    assert not [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png", ".mp4"))]   # sem a foto
    assert QApplication.clipboard().mimeData().urls()[0].toLocalFile() == str(z)
    win.chk_diag_photo.setChecked(True)
    names = zipfile.ZipFile(win.copy_diagnostic()).namelist()
    assert f"foto/{FX.name}" in names and "plate.png" in names
    _shot(win, "5_pronto_com_tempos")


def test_worker_receives_click_time(app, win):
    from mobilia.telemetry import ENV_T_CLIQUE, ENV_T_INICIO, env_time
    import mobilia.ui.runner as runner_mod

    seen = {}
    orig = runner_mod.subprocess.Popen

    def spy(cmd, *a, **kw):
        seen.update(kw.get("env") or {})
        return orig(cmd, *a, **kw)

    runner_mod.subprocess.Popen = spy
    try:
        win.load_photo(str(FX)); win.select_room("sala")
        win.generate()
        assert _wait(app, lambda: win.right.currentIndex() == 2)
    finally:
        runner_mod.subprocess.Popen = orig
    assert env_time(ENV_T_CLIQUE, seen) <= env_time(ENV_T_INICIO, seen)


def test_first_run_notice(app, win, tmp_path):
    """Aviso "primeira execução — mais lenta" na tela de progresso, só enquanto for a primeira."""
    from mobilia import bootstrap, runtime
    sf = runtime.runtime_paths().state_file
    win.load_photo(str(FX)); win.select_room("sala")
    win.generate(); app.processEvents()
    assert win.notice.isVisible() and "Primeira execução" in win.notice.text()
    assert "kernels" in win.notice.text()                     # sem aquecimento do instalador
    _shot(win, "5_primeira_execucao")
    assert _wait(app, lambda: win.right.currentIndex() == 2)
    # instalador aqueceu a GPU: aviso brando
    st = bootstrap.load_state(sf)
    st.update(done=list(bootstrap.STEPS))
    bootstrap.mark_warmup(st, {"ok": True}, {"dependencias": st.get("hashes", {}).get("dependencias")})
    bootstrap.save_state(sf, st)
    win.new_photo(); win.load_photo(str(FX)); win.select_room("sala")
    win.generate(); app.processEvents()
    assert win.notice.isVisible() and "um pouco mais lenta" in win.notice.text()
    assert _wait(app, lambda: win.right.currentIndex() == 2)
    # depois de uma geração concluída: sem aviso
    bootstrap.record_run(sf)
    win.new_photo(); win.load_photo(str(FX)); win.select_room("sala")
    win.generate(); app.processEvents()
    assert not win.notice.isVisible()
    assert _wait(app, lambda: win.right.currentIndex() == 2)


def test_setup_dialog_text_for_update(app):
    from mobilia import bootstrap
    from mobilia.ui.setup_dialog import intro_text
    title, _ = intro_text({"done": []})
    assert title.startswith("Primeiro uso")
    title, text = intro_text({"done": [s for s in bootstrap.STEPS if s != "aquecimento"]})
    assert title.startswith("Atualização") and "preparando a GPU" in text
    _, text = intro_text({"done": ["python", "pip"]})
    assert "instalando PyTorch" in text and "copiando o motor do 7xAi" in text
    assert "baixando os modelos de IA" in text
