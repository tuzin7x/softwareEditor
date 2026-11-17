"""Registro de módulos da janela principal (mobilia/ui/modules.py + pacote mobilia/modules).

Sem módulos registrados, a janela tem que ser exatamente a de antes; com um módulo de teste registrado em tempo
de execução, aparece o seletor no cabeçalho e a troca de páginas funciona (página criada uma só vez).
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QWidget  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FX = Path(__file__).parent / "fixtures/photos/sala_vazia_stafford.jpg"
WORKER = Path(__file__).parent / "fake_design_worker.py"


@pytest.fixture(scope="module")
def app():
    from mobilia.ui import theme
    a = QApplication.instance() or QApplication([])
    a.setStyle("Fusion"); a.setStyleSheet(theme.QSS)
    yield a
    a.clipboard().clear()
    a.processEvents()


@pytest.fixture(autouse=True)
def no_modal(monkeypatch):
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: 0))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MOBILIA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MOBILIA_WORKER", json.dumps([sys.executable, str(WORKER)]))
    return tmp_path


@pytest.fixture()
def reg():
    """Registro isolado: guarda o estado global e o restaura no fim (os outros testes veem o registro vazio)."""
    from mobilia.ui import modules as m
    saved = (dict(m._REGISTRY), list(m.failed), m._loaded)
    m._reset_for_tests()
    yield m
    m._REGISTRY.clear(); m._REGISTRY.update(saved[0])
    m.failed[:] = saved[1]
    m._loaded = saved[2]


class DummyPage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window_ref = window
        self.events: list[str] = []
        QLabel("Módulo de teste", self)

    def on_show(self):
        self.events.append("show")

    def on_hide(self):
        self.events.append("hide")


def _make_window(app):
    from mobilia.ui.app import MainWindow
    w = MainWindow(); w.resize(1360, 840); w.show(); app.processEvents()
    return w


# ============================================================================================ registro

def test_register_validates_ids(reg):
    spec = reg.ModuleSpec("teste", "Teste", lambda w: QWidget())
    assert reg.register(spec) is spec
    assert reg.get("teste") is spec
    with pytest.raises(ValueError):
        reg.register(spec)                                            # repetido
    for bad in ("design", "Maiusculo", "1abc", "com-hifen", ""):
        with pytest.raises(ValueError):
            reg.register(reg.ModuleSpec(bad, "X", lambda w: QWidget()))
    with pytest.raises(ValueError):
        reg.register(reg.ModuleSpec("sem_titulo", "  ", lambda w: QWidget()))
    with pytest.raises(TypeError):
        reg.register(reg.ModuleSpec("nao_chamavel", "X", None))        # type: ignore[arg-type]
    with pytest.raises(TypeError):
        reg.register("teste")                                         # type: ignore[arg-type]
    reg.unregister("teste")
    assert reg.get("teste") is None and reg.registered() == []


def test_registered_is_sorted_by_order_then_title(reg):
    for mid, title, order in [("b", "Beta", 20), ("a", "alfa", 20), ("z", "Zeta", 5)]:
        reg.register(reg.ModuleSpec(mid, title, lambda w: QWidget(), order=order))
    assert [s.id for s in reg.registered()] == ["z", "a", "b"]


def test_repository_ships_no_module_yet():
    """A base para o desenvolvimento paralelo não traz nenhum módulo: o pacote só tem o __init__."""
    pkg = ROOT / "mobilia/modules"
    extra = [p.name for p in pkg.iterdir() if p.name not in {"__init__.py", "__pycache__"}]
    assert extra == []


def test_discover_imports_submodules_and_skips_broken(reg, tmp_path, monkeypatch):
    import mobilia.modules as pkg
    (tmp_path / "bom.py").write_text(
        "from mobilia.ui.modules import ModuleSpec, register\n"
        "from PySide6.QtWidgets import QWidget\n"
        "register(ModuleSpec('bom', 'Bom', lambda w: QWidget()))\n", encoding="utf-8")
    (tmp_path / "quebrado.py").write_text("raise RuntimeError('falhou de propósito')\n", encoding="utf-8")
    (tmp_path / "_privado.py").write_text("raise RuntimeError('não deveria ser importado')\n", encoding="utf-8")
    monkeypatch.setattr(pkg, "__path__", [str(tmp_path)])
    for n in ("bom", "quebrado", "_privado"):
        monkeypatch.delitem(sys.modules, f"mobilia.modules.{n}", raising=False)
    try:
        ok = pkg.discover()
    finally:
        for n in ("bom", "quebrado", "_privado"):
            sys.modules.pop(f"mobilia.modules.{n}", None)
    assert ok == ["mobilia.modules.bom"]
    assert [s.id for s in reg.registered()] == ["bom"]
    assert [f[0] for f in reg.failed] == ["mobilia.modules.quebrado"] and "falhou de propósito" in reg.failed[0][1]


def test_load_never_raises(reg, monkeypatch):
    import mobilia.modules as pkg

    def boom():
        raise RuntimeError("descoberta quebrada")
    monkeypatch.setattr(pkg, "discover", boom)
    reg._loaded = False
    assert reg.load() == []
    assert reg.failed and reg.failed[0][0] == "mobilia.modules"
    assert reg.load() == []                                           # só tenta uma vez
    assert len(reg.failed) == 1


# ============================================================================================ janela

def test_window_without_modules_is_unchanged(app, env, reg):
    w = _make_window(app)
    try:
        assert not w.modules.active
        assert w.modules.views is None and w.modules.nav is None
        tag = w.findChild(QLabel, "tagline")
        assert tag is not None and tag.isVisible()
        assert w.btn_legacy.isVisible()
        # mesma estrutura de antes: cabeçalho, stepper e corpo direto no layout principal
        assert w.outer.count() == 3 and w.outer.itemAt(1).widget() is w.stepper
        w.modules.show("qualquer")                                    # inócuo
        w.modules.set_compact(True)
    finally:
        w.close()


def test_window_with_dummy_module(app, env, reg):
    built = []

    def factory(window):
        page = DummyPage(window); built.append(page)
        return page
    reg.register(reg.ModuleSpec("dummy", "Dummy", factory, tooltip="Módulo de teste"))
    w = _make_window(app)
    try:
        host = w.modules
        assert host.active and host.views is not None and host.nav is not None
        assert host.nav.isVisible() and list(host.nav.buttons) == ["design", "dummy"]
        assert host.nav.buttons["dummy"].toolTip() == "Módulo de teste"
        assert not w.findChild(QLabel, "tagline").isVisible()
        assert host.views.currentIndex() == 0 and w.stepper.isVisible()
        assert built == []                                            # preguiçoso: nada criado ainda

        w.load_photo(str(FX)); app.processEvents()
        host.nav.buttons["dummy"].click(); app.processEvents()
        assert host.current == "dummy" and len(built) == 1
        page = built[0]
        assert page.window_ref is w and host.views.currentWidget() is page
        assert page.events == ["show"]
        assert not w.stepper.isVisible() and not w.btn_legacy.isVisible()

        host.nav.buttons["design"].click(); app.processEvents()
        assert host.current == "design" and host.views.currentIndex() == 0
        assert page.events == ["show", "hide"]
        assert w.stepper.isVisible() and w.btn_legacy.isVisible()
        assert w.photo_path == str(FX)                                # estado do redesenho preservado

        host.show("dummy"); app.processEvents()
        assert len(built) == 1 and host.nav.value() == "dummy"        # mesma página, sem recriar
        host.show("design"); app.processEvents()

        # tela baixa: com o seletor, o botão principal continua visível
        w.resize(1280, 720); app.processEvents()
        assert w._compact
        w.function_cards["mudar_estilo"].click(); app.processEvents()
        bottom = w.btn_generate.mapTo(w, w.btn_generate.rect().bottomLeft()).y()
        assert w.btn_generate.isVisible() and bottom < w.statusBar().y()
    finally:
        w.close()


def test_broken_factory_shows_notice(app, env, reg):
    def factory(window):
        raise RuntimeError("página quebrada")
    reg.register(reg.ModuleSpec("ruim", "Ruim", factory))
    w = _make_window(app)
    try:
        w.modules.show("ruim"); app.processEvents()
        page = w.modules.views.currentWidget()
        notice = page.findChild(QLabel, "notice")
        assert notice is not None and "Ruim" in notice.text()
        assert reg.failed and reg.failed[-1][0] == "ruim"
        w.modules.show("design"); app.processEvents()
        assert w.modules.views.currentIndex() == 0
    finally:
        w.close()


# ======================================================================================== importações

def test_module_code_does_not_import_engine_libraries():
    """Mesma regra de tests/test_ui.py, estendida ao registro e aos módulos: o .exe da interface não leva
    numpy/PIL/torch/cv2 nem QtWebEngine."""
    files = [ROOT / "mobilia/ui/modules.py", *sorted((ROOT / "mobilia/modules").rglob("*.py"))]
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            for n in names:
                assert n.split(".")[0] not in {"numpy", "PIL", "torch", "cv2", "diffusers"}, (f, n)
                assert not n.startswith("PySide6.QtWebEngine"), (f, n)
        text = f.read_text(encoding="utf-8")
        assert "mobilia.design" not in text and "mobilia.video" not in text, f


def test_spec_collects_module_package():
    spec = (ROOT / "packaging/7xai.spec").read_text(encoding="utf-8")
    assert 'collect_submodules("mobilia.modules")' in spec
    assert '"mobilia.ui.modules"' in spec
