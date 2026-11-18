"""Interface (offscreen): fluxo de redesenho por IA e o modo legado, com dublês do motor.

Os testes do fluxo principal clicam nos botões de verdade (``QAbstractButton.click``/``QTest``): o sinal
``clicked(bool)`` entrega ``checked=False`` e isso nunca pode virar parâmetro de pedido (o erro antigo do "modo False").
O motor é ``tests/fake_design_worker.py`` (``design``/``upscale``/``showcase``; o resto vai para ``fake_worker.py``).
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDropEvent, QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FX = Path(__file__).parent / "fixtures/photos/sala_vazia_stafford.jpg"
SHOTS = Path(os.environ.get("MOBILIA_UI_SHOTS", "/tmp/claude-0/ui"))
WORKER = Path(__file__).parent / "fake_design_worker.py"


@pytest.fixture(scope="module")
def app():
    from mobilia.ui import theme
    a = QApplication.instance() or QApplication([])
    a.setStyle("Fusion"); a.setStyleSheet(theme.QSS)
    yield a
    # "Copiar diagnóstico" deixa um QMimeData na área de transferência; com ele lá, o PySide6
    # (plataforma offscreen) derruba o interpretador no encerramento (código 139) mesmo com todos
    # os testes aprovados. Limpar a área de transferência ainda com o Python vivo evita isso.
    a.clipboard().clear()
    a.processEvents()


@pytest.fixture(autouse=True)
def no_modal(monkeypatch):
    """Nenhum diálogo modal trava o teste: avisos e erros são registrados em vez de exibidos."""
    seen = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: seen.append(("box", self.text(), self.informativeText())) or 0)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: seen.append(("warning",) + a[1:3]) or 0))
    return seen


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MOBILIA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MOBILIA_WORKER", json.dumps([sys.executable, str(WORKER)]))
    monkeypatch.delenv("MOBILIA_FAKE_DESIGN_SRC", raising=False)
    return tmp_path


@pytest.fixture()
def win(app, env):
    """Janela principal (redesenho por IA)."""
    from mobilia.ui.app import MainWindow
    w = MainWindow(); w.resize(1360, 840); w.show(); app.processEvents()
    opened = []
    w.open_url = lambda path: opened.append(str(path))
    w.opened = opened
    yield w
    w.close()
    if w.legacy_window is not None:
        w.legacy_window.close()


@pytest.fixture()
def lwin(app, env):
    """Janela do modo legado (vídeo de montagem, modos A/B)."""
    from mobilia.ui.legacy import LegacyWindow
    w = LegacyWindow(); w.resize(1360, 820); w.show(); app.processEvents()
    yield w
    w.close()


def _wait(app, cond, timeout=30):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents(); time.sleep(0.02)
    return cond()


def _shot(w, name):
    SHOTS.mkdir(parents=True, exist_ok=True)
    w.grab().save(str(SHOTS / f"{name}.png"))


def _args(out_dir) -> list[str]:
    return json.loads((Path(out_dir) / "args.json").read_text(encoding="utf-8"))


def _opt(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def _no_signal_leak(args):
    """Nenhum valor booleano de sinal Qt virou argumento do motor."""
    assert not {"False", "True", "None"} & set(args), args


def _to_results(app, win, funcao="mudar_estilo", estilo="escandinavo", n=None):
    """Foto -> função -> estilo -> gerar, tudo por cliques, até a tela de resultados."""
    win.load_photo(str(FX)); app.processEvents()
    win.function_cards[funcao].click(); app.processEvents()
    if estilo:
        win.style_cards[estilo].click()
    if n:
        win.n_seg.buttons[n].click()
    win.btn_generate.click(); app.processEvents()
    assert _wait(app, lambda: win.page() == win.P_RESULTS), win.statusBar().currentMessage()
    return win.result


# ================================================================================== fluxo principal

def test_catalog_matches_engine_presets():
    """Os rótulos da interface são os mesmos do motor (a interface não importa o pacote do motor)."""
    from mobilia.ui.app import CATALOG
    spec = importlib.util.spec_from_file_location("_presets", ROOT / "mobilia/design/presets.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_presets"] = mod                     # dataclasses consulta o módulo durante a execução
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("_presets", None)
    assert CATALOG == mod.catalog()
    from mobilia.ui.design_widgets import FUNCTION_ORDER, STYLE_SWATCHES
    assert set(FUNCTION_ORDER) == set(CATALOG["funcoes"]) and set(STYLE_SWATCHES) == set(CATALOG["estilos"])
    assert len(CATALOG["estilos"]) >= 12


def test_ui_does_not_import_engine_libraries():
    """O executável da interface não leva numpy/PIL/torch: os módulos da UI não podem importá-los."""
    import ast
    for f in (ROOT / "mobilia/ui").glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            for n in names:
                assert n.split(".")[0] not in {"numpy", "PIL", "torch", "cv2", "diffusers"}, (f.name, n)
        assert "design.api" not in f.read_text(encoding="utf-8")


def test_on_click_discards_signal_args(app):
    from PySide6.QtWidgets import QPushButton

    from mobilia.ui.widgets import on_click
    got = []
    b = QPushButton("x"); b.setCheckable(True)
    on_click(b, lambda *a, **k: got.append((a, k)), "mobiliar", n=2)
    b.click(); b.click()
    assert got == [(("mobiliar",), {"n": 2})] * 2


def test_request_validation_rejects_signal_values(tmp_path):
    from mobilia.ui.app import DesignRequest
    ok = DesignRequest("texto", "moderno", "sala", 4, str(tmp_path / "o"))
    assert "--funcao" in ok.args() and "--photo" not in ok.args()
    for bad in (dict(n=False), dict(n=True), dict(n=5), dict(funcao=False), dict(estilo=False), dict(ambiente=False),
                dict(formato=False), dict(quality=False), dict(seed=False)):
        with pytest.raises(ValueError):
            DesignRequest(**{**ok.__dict__, **bad}).validate()
    with pytest.raises(ValueError):                 # funções sobre a foto exigem a foto
        DesignRequest("mudar_estilo", "moderno", "sala", 2, str(tmp_path / "o")).validate()


def test_ui_arguments_parse_with_the_real_cli(tmp_path, monkeypatch):
    """Os argumentos que a interface monta passam pelo parser real de ``mobilia.cli`` (design, upscale, showcase)."""
    import mobilia.cli as cli
    from mobilia.ui.app import DesignRequest, showcase_args, upscale_args
    seen = {}
    for name in ("_cmd_design", "_cmd_upscale", "_cmd_showcase"):
        monkeypatch.setattr(cli, name, lambda a, _n=name: seen.__setitem__(_n, a) or 0)
    mask = tmp_path / "m.png"; mask.write_bytes(b"x")
    req = DesignRequest("editar_area", "boho", "quarto", 3, str(tmp_path / "o"), photo=str(FX), mask=str(mask),
                        prompt="poltrona caramelo", quality="rapido")
    assert cli.main(["design", *req.args()]) == 0
    a = seen["_cmd_design"]
    assert (a.funcao, a.estilo, a.ambiente, a.n, a.photo, a.mask, a.prompt, a.quality, a.json) == \
        ("editar_area", "boho", "quarto", 3, str(FX), str(mask), "poltrona caramelo", "rapido", True)
    req = DesignRequest("evento", None, "sala", 1, str(tmp_path / "e"), photo=str(FX), evento="natal")
    assert cli.main(["design", *req.args()]) == 0
    assert (seen["_cmd_design"].evento, seen["_cmd_design"].estilo) == ("natal", None)
    req = DesignRequest("texto", "japandi", "quarto", 2, str(tmp_path / "t"), formato="retrato")
    assert cli.main(["design", *req.args()]) == 0
    assert (seen["_cmd_design"].formato, seen["_cmd_design"].photo) == ("retrato", None)
    assert cli.main(["upscale", *upscale_args("a.jpg", "a_4k.jpg")]) == 0
    assert (seen["_cmd_upscale"].image, seen["_cmd_upscale"].out, seen["_cmd_upscale"].json) == ("a.jpg", "a_4k.jpg", True)
    assert cli.main(["showcase", *showcase_args("antes.jpg", ["v1.jpg", "v2.jpg"], "v.mp4", "9x16")]) == 0
    sc = seen["_cmd_showcase"]
    assert (sc.before, sc.after, sc.out, sc.format, sc.json) == ("antes.jpg", ["v1.jpg", "v2.jpg"], "v.mp4", "9x16", True)


def test_drag_and_drop_photo_advances_to_functions(app, win):
    _shot(win, "1_foto")
    assert win.page() == win.P_PHOTO and not win.btn_photo_next.isEnabled()
    for k in ("mudar_estilo", "mobiliar", "editar_area"):        # sem foto: só "Texto para ambiente"
        win.go(win.P_FUNC); assert not win.function_cards[k].isEnabled()
    assert win.function_cards["texto"].isEnabled()
    win.go(win.P_PHOTO)
    mime = QMimeData(); mime.setUrls([QUrl.fromLocalFile(str(FX))])
    ev = QDropEvent(QPointF(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    win.drop.dropEvent(ev); app.processEvents()
    assert win.photo_path == str(FX) and win.page() == win.P_FUNC
    assert all(c.isEnabled() for c in win.function_cards.values())
    _shot(win, "2_funcao")


def test_full_flow_by_clicks(app, win):
    """Foto -> Mudar estilo -> quarto, escandinavo, 2 versões -> gerar -> resultados, só com cliques."""
    win.load_photo(str(FX)); app.processEvents()
    win.function_cards["mudar_estilo"].click(); app.processEvents()
    assert win.page() == win.P_STYLE and win.funcao == "mudar_estilo"
    win.room_chips["quarto"].click(); win.style_cards["escandinavo"].click(); win.n_seg.buttons[2].click()
    win.chk_fast.click()
    win.prompt.setPlainText("sofá verde-musgo")
    assert win.btn_generate.text().strip() == "Gerar 2 versões"
    _shot(win, "3_estilo")
    win.btn_generate.click(); app.processEvents()
    assert win.page() == win.P_PROGRESS and win.left.currentIndex() == win.L_PROGRESS
    assert _wait(app, lambda: win.page() == win.P_RESULTS)
    a = _args(win.last_request.out_dir)
    _no_signal_leak(a)
    assert a[0] == "design" and _opt(a, "--funcao") == "mudar_estilo" and _opt(a, "--estilo") == "escandinavo"
    assert _opt(a, "--ambiente") == "quarto" and _opt(a, "--n") == "2" and _opt(a, "--quality") == "rapido"
    assert _opt(a, "--photo") == str(FX) and _opt(a, "--prompt") == "sofá verde-musgo" and "--json" in a
    r = win.result
    assert len(r.outputs) == 2 and all(Path(o).is_file() for o in r.outputs)
    assert r.before and Path(r.before).name == "antes.jpg" and win.compare.has_before()
    assert len(win.thumbs.buttons) == 2 and win.selected == 0
    # escolher a versão 2 pela miniatura e arrastar o divisor do antes/depois
    win.thumbs.buttons[1].click(); app.processEvents()
    assert win.selected == 1 and win.selected_output() == r.outputs[1] and "VERSÃO 2 DE 2" == win.selected_label.text()
    rect = win.compare.image_rect()
    y = int(rect.center().y())
    QTest.mousePress(win.compare, Qt.LeftButton, Qt.NoModifier, QPoint(int(rect.left() + rect.width() * 0.2), y))
    QTest.mouseMove(win.compare, QPoint(int(rect.left() + rect.width() * 0.8), y))
    QTest.mouseRelease(win.compare, Qt.LeftButton, Qt.NoModifier, QPoint(int(rect.left() + rect.width() * 0.8), y))
    assert 0.15 < win.compare.split < 0.85
    QTest.mouseClick(win.compare, Qt.LeftButton, Qt.NoModifier, QPoint(int(rect.left() + rect.width() * 0.25), y))
    assert abs(win.compare.split - 0.25) < 0.02
    _shot(win, "5_resultados")
    # salvar imagem / abrir pasta / novas versões
    win.ask_save_path = lambda suggested, flt: str(Path(win.result.folder).parent / "salva.jpg")
    win.btn_save.click()
    assert (Path(win.result.folder).parent / "salva.jpg").read_bytes() == Path(r.outputs[1]).read_bytes()
    win.btn_folder.click()
    assert win.opened == [r.folder]
    first = win.last_request.out_dir
    win.btn_again.click(); app.processEvents()
    assert _wait(app, lambda: win.page() == win.P_RESULTS and win.last_request.out_dir != first)
    a2 = _args(win.last_request.out_dir)
    assert _opt(a2, "--estilo") == "escandinavo" and _opt(a2, "--n") == "2" and "--seed" not in a2


def test_every_main_button_click_is_safe(app, win, no_modal):
    """Clica em todos os botões das telas: nada levanta exceção nem manda valores de sinal ao motor."""
    _to_results(app, win, "reforma", "luxo", n=1)
    a = _args(win.last_request.out_dir)
    _no_signal_leak(a)
    assert _opt(a, "--funcao") == "reforma" and _opt(a, "--n") == "1"
    win.go(win.P_STYLE)
    for k in win.style_cards:
        win.style_cards[k].click()
        assert win.estilo == k
    for k in win.room_chips:
        win.room_chips[k].click()
        assert win.ambiente == k
    for v in (1, 2, 3, 4):
        win.n_seg.buttons[v].click()
        assert win.n_seg.value() == v and type(win.n_seg.value()) is int
    req = win.build_request()
    assert req.validate() and type(req.n) is int and req.estilo in win.style_cards
    assert not [m for m in no_modal if m[0] == "box"]


def test_cancel_stops_generation(app, win):
    win.load_photo(str(FX)); win.function_cards["mobiliar"].click()
    win.prompt.setPlainText("devagar")
    win.btn_generate.click(); app.processEvents()
    assert _wait(app, lambda: win.stages._idx >= 1, 15)
    assert win.page() == win.P_PROGRESS and win.design.busy()
    # navegar não sai da tela de progresso enquanto gera
    win.go(win.P_PHOTO); assert win.page() == win.P_PROGRESS
    _shot(win, "4_gerando")
    t0 = time.time()
    win.btn_cancel.click()
    assert _wait(app, lambda: not win.design.busy(), 15)
    assert time.time() - t0 < 10
    assert win.page() == win.P_STYLE and win.result is None
    assert "cancelada" in win.statusBar().currentMessage()


def test_engine_error_returns_to_settings(app, win, no_modal):
    win.load_photo(str(FX)); win.function_cards["mudar_estilo"].click()
    win.prompt.setPlainText("falhar")
    win.btn_generate.click()
    assert _wait(app, lambda: not win.design.busy() and win.page() != win.P_PROGRESS)
    assert win.page() == win.P_STYLE
    assert ("box", "Não foi possível gerar as versões.", "Falha simulada do redesenho") in no_modal


def test_upscale_and_save_4k(app, win):
    r = _to_results(app, win, "mobiliar", "moderno", n=2)
    win.thumbs.buttons[1].click()
    win.btn_upscale.click(); app.processEvents()
    assert win.taskbox.isVisible() and not win.btn_upscale.isEnabled()
    assert _wait(app, lambda: not win.side.busy())
    up = json.loads((Path(r.folder) / "upscale_args.json").read_text())
    _no_signal_leak(up)
    assert _opt(up, "--image") == r.outputs[1] and _opt(up, "--out").endswith("_2_4k.jpg") and "--json" in up
    assert win.upscaled[r.outputs[1]] == _opt(up, "--out") and Path(_opt(up, "--out")).is_file()
    assert win.task_title.text() == "Imagem 4K pronta" and win.thumbs.buttons[1].badge == "4K"
    assert "4K" in win.btn_save.text()
    _shot(win, "5b_resultados_4k")
    dest = Path(r.folder) / "exportada_4k.jpg"
    win.ask_save_path = lambda suggested, flt: str(dest)
    win.btn_save.click()
    assert dest.read_bytes() == Path(win.upscaled[r.outputs[1]]).read_bytes()
    win.btn_task_open.click()
    assert win.opened[-1] == win.upscaled[r.outputs[1]]


def test_before_after_video(app, win):
    r = _to_results(app, win, "mudar_estilo", "industrial", n=3)
    win.btn_video.click(); app.processEvents()
    assert win.page() == win.P_VIDEO and len(win.video_checks) == 3
    win.format_chips["9x16"].click()
    win.video_checks[0].click()                     # tira a versão 1
    assert win.video_selection() == r.outputs[1:]
    _shot(win, "6_video")
    win.btn_video_generate.click(); app.processEvents()
    assert win.page() == win.P_RESULTS and win.taskbox.isVisible()
    assert _wait(app, lambda: not win.side.busy())
    sa = json.loads((Path(r.folder) / "showcase_args.json").read_text())
    _no_signal_leak(sa)
    assert _opt(sa, "--before") == r.before and _opt(sa, "--format") == "9x16"
    afters = sa[sa.index("--after") + 1: sa.index("--out")]
    assert afters == r.outputs[1:]
    assert win.task_title.text() == "Vídeo pronto" and Path(win.videos[-1]).is_file()
    _shot(win, "6b_video_pronto")
    dest = Path(r.folder) / "video_exportado.mp4"
    win.ask_save_path = lambda suggested, flt: str(dest)
    win.btn_task_export.click()
    assert dest.is_file()


def _paint(app, editor, frac_from=(0.3, 0.6), frac_to=(0.6, 0.7)):
    r = editor.image_rect()
    a = QPoint(int(r.left() + r.width() * frac_from[0]), int(r.top() + r.height() * frac_from[1]))
    b = QPoint(int(r.left() + r.width() * frac_to[0]), int(r.top() + r.height() * frac_to[1]))
    QTest.mousePress(editor, Qt.LeftButton, Qt.NoModifier, a)
    for k in range(1, 6):
        QTest.mouseMove(editor, a + (b - a) * (k / 5))
    QTest.mouseRelease(editor, Qt.LeftButton, Qt.NoModifier, b)
    app.processEvents()


def test_edit_area_on_a_result(app, win):
    r = _to_results(app, win, "mudar_estilo", "japandi", n=2)
    win.thumbs.buttons[1].click()
    win.btn_edit.click(); app.processEvents()
    assert win.page() == win.P_EDIT and win.left.currentIndex() == win.L_MASK and win.edit_base == r.outputs[1]
    assert not win.btn_edit_generate.isEnabled()
    _paint(app, win.mask)
    assert win.mask.has_mask() and win.btn_edit_generate.isEnabled() and win.btn_undo.isEnabled()
    # borracha + desfazer
    win.tool_seg.buttons["borracha"].click(); assert win.mask.erasing
    win.brush_slider.setValue(120); assert win.mask.brush_px == 120 and win.brush_value.text() == "120 px"
    cov = win.mask.coverage()
    _paint(app, win.mask)
    assert win.mask.coverage() < cov
    win.btn_undo.click(); assert abs(win.mask.coverage() - cov) < 1e-6
    win.tool_seg.buttons["pincel"].click(); assert not win.mask.erasing
    win.edit_prompt.setPlainText("poltrona de couro caramelo")
    win.edit_style.setCurrentIndex(list(win.style_cards).index("boho") + 1)
    win.edit_n.buttons[1].click()
    _shot(win, "7_editar_area")
    win.btn_edit_generate.click(); app.processEvents()
    assert _wait(app, lambda: win.page() == win.P_RESULTS and win.result.request.funcao == "editar_area")
    a = _args(win.last_request.out_dir)
    _no_signal_leak(a)
    assert _opt(a, "--funcao") == "editar_area" and _opt(a, "--photo") == r.outputs[1]
    assert _opt(a, "--estilo") == "boho" and _opt(a, "--n") == "1" and _opt(a, "--prompt") == "poltrona de couro caramelo"
    m = QImage(_opt(a, "--mask"))
    base = QImage(r.outputs[1])
    assert m.size() == base.size()
    g = m.convertToFormat(QImage.Format_Grayscale8)
    r_ = win.mask.image_rect()
    assert g.pixelColor(int(g.width() * 0.45), int(g.height() * 0.65)).red() == 255
    assert g.pixelColor(2, 2).red() == 0 and r_.width() > 0
    assert win.result_title.text().startswith("Área editada")


def test_edit_area_from_function_card(app, win):
    win.load_photo(str(FX))
    win.function_cards["editar_area"].click(); app.processEvents()
    assert win.page() == win.P_EDIT and win.edit_base == str(FX)
    win.btn_edit_back.click(); app.processEvents()
    assert win.page() == win.P_FUNC and win.funcao is None


def test_text_to_design_without_photo(app, win):
    win.btn_text_mode.click(); app.processEvents()
    assert win.page() == win.P_STYLE and win.funcao == "texto" and win.left.currentIndex() == win.L_TEXT
    assert win.aspect_box.isVisibleTo(win) and win.style_box.isVisibleTo(win)
    win.room_chips["quarto"].click(); win.style_cards["japandi"].click()
    win.aspect_seg.buttons["retrato"].click()
    win.prompt.setPlainText("quarto de casal com cabeceira de palhinha")
    _shot(win, "8_texto")
    win.btn_generate.click()
    assert _wait(app, lambda: win.page() == win.P_RESULTS)
    a = _args(win.last_request.out_dir)
    _no_signal_leak(a)
    assert _opt(a, "--funcao") == "texto" and _opt(a, "--formato") == "retrato" and "--photo" not in a
    assert win.result.before is None and not win.compare.has_before() and not win.btn_video.isEnabled()


def test_event_and_facade_requests(app, win):
    win.load_photo(str(FX))
    win.function_cards["evento"].click(); app.processEvents()
    assert win.event_box.isVisibleTo(win) and not win.style_box.isVisibleTo(win)
    win.event_chips["natal"].click(); win.n_seg.buttons[1].click()
    _shot(win, "9_evento")
    win.btn_generate.click()
    assert _wait(app, lambda: win.page() == win.P_RESULTS)
    a = _args(win.last_request.out_dir)
    assert _opt(a, "--funcao") == "evento" and _opt(a, "--evento") == "natal" and "--estilo" not in a
    assert win.result_title.text().startswith("Natal")
    win.go(win.P_FUNC)
    win.function_cards["fachada"].click(); app.processEvents()
    assert not win.room_box.isVisibleTo(win)
    win.style_cards["tropical"].click(); win.btn_generate.click()
    assert _wait(app, lambda: win.page() == win.P_RESULTS and win.result.request.funcao == "fachada")
    a = _args(win.last_request.out_dir)
    assert _opt(a, "--ambiente") == "fachada" and _opt(a, "--estilo") == "tropical"
    # voltar a outra função devolve o ambiente de interior
    win.go(win.P_FUNC); win.function_cards["mudar_estilo"].click()
    assert win.ambiente == "sala"


def test_stepper_and_new_photo(app, win):
    _to_results(app, win, "redesign_simples", "costeiro", n=1)
    assert win.stepper.current == 4 and {0, 1, 2, 4} <= win.stepper.reachable
    win.stepper.step_clicked.emit(2); app.processEvents()
    assert win.page() == win.P_STYLE
    win.stepper.step_clicked.emit(4); app.processEvents()
    assert win.page() == win.P_RESULTS
    win.btn_new.click(); app.processEvents()
    assert win.page() == win.P_PHOTO and win.photo_path is None and win.result is None


def test_legacy_button_opens_old_video_mode(app, win):
    win.load_photo(str(FX))
    win.btn_legacy.click(); app.processEvents()
    lw = win.legacy_window
    assert lw is not None and lw.isVisible() and lw.photo_path == str(FX)
    assert "legado" in lw.windowTitle()
    _shot(lw, "10_legado")


def test_compact_1280x720_keeps_main_buttons_visible(app, win):
    win.resize(1280, 720); app.processEvents()
    assert win._compact
    win.load_photo(str(FX)); win.function_cards["mudar_estilo"].click(); app.processEvents()
    _shot(win, "11_1280x720_estilo")
    for btn in (win.btn_generate,):
        bottom = btn.mapTo(win, btn.rect().bottomLeft()).y()
        assert btn.isVisible() and bottom < win.statusBar().y()
    win.resize(1360, 840); app.processEvents()
    assert not win._compact


# ================================================================================== modo legado

def test_flow_success(app, lwin):
    _shot(lwin, "legado_1_vazio")
    assert not lwin.btn_generate.isEnabled()
    lwin.load_photo(str(FX)); lwin.select_room("sala"); app.processEvents()
    assert lwin.btn_generate.isEnabled()
    _shot(lwin, "legado_2_foto_e_ambiente")
    lwin.generate(); app.processEvents()
    assert lwin.right.currentIndex() == 1
    _shot(lwin, "legado_3_gerando")
    assert _wait(app, lambda: lwin.right.currentIndex() == 2)
    assert lwin.last_output and Path(lwin.last_output).exists()
    # previsão do controle de orçamento (PT7) mostrada durante a geração
    assert lwin.forecast.text() == "Previsão: ≈ 1 min 40 s no total"
    assert lwin.last_job.forecast_s == 100.0
    _shot(lwin, "legado_4_pronto")


def test_click_generate_button_sends_valid_mode(app, lwin):
    """O clique real passa clicked(bool); o pedido tem de sair com modo "auto", não "False"."""
    lwin.load_photo(str(FX)); lwin.select_room("sala"); app.processEvents()
    lwin.btn_generate.click(); app.processEvents()
    assert lwin.last_request.mode == "auto"
    cmd = lwin.runner.command_for(lwin.last_request)
    assert "False" not in cmd and "--mode" not in cmd
    assert _wait(app, lambda: lwin.right.currentIndex() == 2)


def test_flow_error_returns_to_setup(app, lwin, monkeypatch):
    shown = {}
    monkeypatch.setattr("mobilia.ui.legacy.LegacyWindow.show_error", lambda self, m, d="": shown.update(m=m, d=d))
    lwin.load_photo(str(FX)); lwin.select_room("banheiro")
    lwin.generate()
    assert _wait(app, lambda: "m" in shown)
    assert shown["m"] == "Falha simulada de teste" and lwin.right.currentIndex() == 0
    # o motor falhou antes de criar a pasta do trabalho: o diagnóstico (botão da tela de erro)
    # ainda sai, com a saída do motor
    import zipfile

    z = lwin.copy_diagnostic()
    assert z is not None and z.exists()
    zf = zipfile.ZipFile(z)
    assert "interface.log" in zf.namelist()
    assert "Falha simulada de teste" in zf.read("interface.log").decode("utf-8")


def test_hardware_probe_updates_status(app, lwin):
    lwin.probe.start()
    assert _wait(app, lambda: "RTX 4070" in lwin.statusBar().currentMessage())


def test_selftest_entrypoint():
    from mobilia.ui.app import main
    assert main(["--selftest"]) == 0


def test_selftest_fails_when_a_module_did_not_load(monkeypatch):
    """No .exe, um módulo que faltou no pacote (ImportError na descoberta) reprova o autoteste do CI."""
    from mobilia.ui import modules
    from mobilia.ui.app import main
    monkeypatch.setattr(modules, "failed", [("mobilia.modules.quebrado", "ImportError: simulado")])
    assert main(["--selftest"]) == 3


def test_copy_diagnostic_zip_without_photo(app, lwin, tmp_path):
    import zipfile

    lwin.load_photo(str(FX)); lwin.select_room("sala"); app.processEvents()
    lwin.generate()
    assert _wait(app, lambda: lwin.right.currentIndex() == 2)
    # tempo por etapa na tela de conclusão (vem do report.json do trabalho)
    assert "Render dos quadros" in lwin.result_times.text() and "Total" in lwin.result_times.text()
    # narrativa de designer (PT12) do report.json na tela de conclusão
    assert "Escolhi um sofá de tecido claro de 1,57 m" in lwin.result_notes.text()
    assert lwin.notes_scroll.isVisible() and lwin.result_notes.text().startswith("•  Sala clara")
    assert lwin.btn_diag.isVisible()
    z = lwin.copy_diagnostic()
    assert z is not None and z.exists()
    names = zipfile.ZipFile(z).namelist()
    assert {"report.json", "render/frames.json", "interface.log", "encode.log", "manifesto.json"} <= set(names)
    assert not [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png", ".mp4"))]   # sem a foto
    assert QApplication.clipboard().mimeData().urls()[0].toLocalFile() == str(z)
    lwin.chk_diag_photo.setChecked(True)
    names = zipfile.ZipFile(lwin.copy_diagnostic()).namelist()
    assert f"foto/{FX.name}" in names and "plate.png" in names
    _shot(lwin, "legado_5_pronto_com_tempos")


def test_worker_receives_click_time(app, lwin):
    from mobilia.telemetry import ENV_T_CLIQUE, ENV_T_INICIO, env_time
    import mobilia.ui.runner as runner_mod

    seen = {}
    orig = runner_mod.subprocess.Popen

    def spy(cmd, *a, **kw):
        seen.update(kw.get("env") or {})
        return orig(cmd, *a, **kw)

    runner_mod.subprocess.Popen = spy
    try:
        lwin.load_photo(str(FX)); lwin.select_room("sala")
        lwin.generate()
        assert _wait(app, lambda: lwin.right.currentIndex() == 2)
    finally:
        runner_mod.subprocess.Popen = orig
    assert env_time(ENV_T_CLIQUE, seen) <= env_time(ENV_T_INICIO, seen)


def test_first_run_notice(app, lwin, tmp_path):
    """Aviso "primeira execução — mais lenta" na tela de progresso, só enquanto for a primeira."""
    from mobilia import bootstrap, runtime
    sf = runtime.runtime_paths().state_file
    lwin.load_photo(str(FX)); lwin.select_room("sala")
    lwin.generate(); app.processEvents()
    assert lwin.notice.isVisible() and "Primeira execução" in lwin.notice.text()
    assert "kernels" in lwin.notice.text()                     # sem aquecimento do instalador
    _shot(lwin, "legado_5_primeira_execucao")
    assert _wait(app, lambda: lwin.right.currentIndex() == 2)
    # instalador aqueceu a GPU: aviso brando
    st = bootstrap.load_state(sf)
    st.update(done=list(bootstrap.STEPS))
    bootstrap.mark_warmup(st, {"ok": True}, {"dependencias": st.get("hashes", {}).get("dependencias")})
    bootstrap.save_state(sf, st)
    lwin.new_photo(); lwin.load_photo(str(FX)); lwin.select_room("sala")
    lwin.generate(); app.processEvents()
    assert lwin.notice.isVisible() and "um pouco mais lenta" in lwin.notice.text()
    assert _wait(app, lambda: lwin.right.currentIndex() == 2)
    # depois de uma geração concluída: sem aviso
    bootstrap.record_run(sf)
    lwin.new_photo(); lwin.load_photo(str(FX)); lwin.select_room("sala")
    lwin.generate(); app.processEvents()
    assert not lwin.notice.isVisible()
    assert _wait(app, lambda: lwin.right.currentIndex() == 2)


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


def test_layout_1280x720_compact_fits(app, lwin):
    """Tela baixa (1280x720 / 1366x768): modo compacto, nada passa da janela, botão principal visível."""
    lwin.set_compact(True); app.processEvents()          # como faz _fit_to_screen ao abrir numa tela baixa
    lwin.resize(1280, 720); app.processEvents()
    assert lwin.size().height() == 720 and lwin._compact
    lwin.load_photo(str(FX)); lwin.select_room("sala"); app.processEvents()
    _shot(lwin, "legado_6_1280x720_foto_e_ambiente")
    bottom = lwin.btn_generate.mapTo(lwin, lwin.btn_generate.rect().bottomLeft()).y()
    assert lwin.btn_generate.isVisible() and bottom < lwin.statusBar().y()
    lwin.resize(1360, 820); app.processEvents()
    assert not lwin._compact


def _furnished_copy(tmp_path, name):
    import shutil
    p = tmp_path / name
    shutil.copyfile(FX, p)
    return str(p)


def test_mode_b_flow_shows_mode_b_stages_and_seal(app, lwin, tmp_path):
    """PT-V3-8: o motor anuncia o modo B (etapas próprias) e a tela final troca o selo e o título da narrativa;
    a tela inicial continua pedindo só a foto e o ambiente."""
    lwin.load_photo(_furnished_copy(tmp_path, "sala_mobiliada.jpg")); lwin.select_room("sala"); app.processEvents()
    lwin.generate()
    assert _wait(app, lambda: lwin.right.currentIndex() == 2)
    assert "sala_vazia" in lwin.stages.order and "render" not in lwin.stages.order
    assert lwin.last_job.mode == "B" and lwin.last_request.mode == "auto"
    assert "idêntico à sua foto" in lwin.seal_text.text() and "ilustrativa" in lwin.seal_text.text()
    assert lwin.notes_title.text() == "O QUE O 7xAi FEZ"
    assert not lwin.result_warning.isVisible()
    # um novo trabalho volta à lista de etapas do modo A até o motor anunciar o modo
    lwin.new_photo(); lwin.load_photo(str(FX)); lwin.generate()
    assert "render" in lwin.stages.order
    assert _wait(app, lambda: lwin.right.currentIndex() == 2)
    assert lwin.seal_text.text() == lwin.SEAL_A


def test_mode_b_gate_rejected_offers_choices(app, lwin, tmp_path, monkeypatch):
    """Sala vazia reprovada: sem MP4; o aviso oferece gerar mesmo assim (com aviso na tela final), o modo A ou
    outra foto."""
    asked = []

    def choose(self, job):
        asked.append(job)
        return "forcar" if len(asked) == 1 else None

    monkeypatch.setattr("mobilia.ui.legacy.LegacyWindow.ask_gate_choice", choose)
    lwin.load_photo(_furnished_copy(tmp_path, "sala_mobiliada_reprovada.jpg")); lwin.select_room("sala")
    lwin.generate()
    assert _wait(app, lambda: len(asked) == 1)
    assert asked[0].status == "reprovado" and asked[0].output is None and "qualidade" in asked[0].warning
    # "Gerar mesmo assim" refaz no modo B com o aceite explícito
    assert lwin.last_request.mode == "B" and lwin.last_request.accept_weak_plate
    assert "--aceitar-placa-fraca" in lwin.runner.command_for(lwin.last_request)
    assert _wait(app, lambda: lwin.right.currentIndex() == 2)
    assert lwin.result_warning.isVisible() and "a seu pedido" in lwin.result_warning.text()
    _shot(lwin, "legado_5_modo_b_com_aviso")


def test_gate_choice_mode_a(app, lwin, tmp_path, monkeypatch):
    monkeypatch.setattr("mobilia.ui.legacy.LegacyWindow.ask_gate_choice", lambda self, job: "modo_a")
    lwin.load_photo(_furnished_copy(tmp_path, "sala_mobiliada_reprovada.jpg")); lwin.select_room("sala")
    lwin.generate()
    assert _wait(app, lambda: lwin.last_request is not None and lwin.last_request.mode == "A")
    assert "--mode" in lwin.runner.command_for(lwin.last_request)
    assert _wait(app, lambda: lwin.right.currentIndex() == 2)
    assert lwin.last_job.mode == "A"


def test_engine_notice_shown_on_results(app, win):
    """O motor trocou a função (foto de ambiente vazio -> Mobiliar): a tela de resultados diz isso."""
    win.load_photo(str(FX)); app.processEvents()
    win.function_cards["mudar_estilo"].click(); app.processEvents()
    win.n_seg.buttons[1].click()
    win.prompt.setPlainText("vazio")
    win.btn_generate.click(); app.processEvents()
    assert _wait(app, lambda: win.page() == win.P_RESULTS)
    assert win.result.aviso and win.result.funcao == "mobiliar"
    assert win.result_notice.isVisible() and "vazio" in win.result_notice.text()
    assert "Mobiliar" in win.result_info.text() and " px" not in win.result_info.text()
    assert win.statusBar().currentMessage() == ""


def test_theme_tokens_come_from_7x_intelligence_oklch_and_are_readable(app):
    """Identidade 7x Intelligence: tokens oklch convertidos com precisão, contraste >= 4.5:1, fontes e marca."""
    from mobilia.ui import brand, theme
    # conversão oklch -> sRGB conferida com valores de referência (corte no gamut sRGB, como o Chrome)
    assert theme.oklch(0.12, 0.012, 290) == theme.BG == "#06050A"
    assert theme.oklch(0.68, 0.22, 298) == theme.ACCENT == "#AC70FF"
    assert theme.oklch(0.76, 0.18, 315) == theme.GLOW == "#DB8AFE"
    assert theme.oklch(0.28, 0.03, 292) == "#292637"            # --border do site
    assert theme.oklch(1, 0, 0) == "#FFFFFF" and theme.oklch(0, 0, 0) == "#000000"
    assert all(v >= 4.5 for v in theme.contrast_table().values()), theme.contrast_table()
    assert theme.assets_ok()
    from PySide6.QtGui import QFont, QFontInfo
    for fam, w in ((theme.UI_FAMILY, QFont.Normal), (theme.DISPLAY_FAMILY, QFont.DemiBold),
                   (theme.MONO_FAMILY, QFont.Medium)):
        assert QFontInfo(theme.font(13, w, fam)).family() == fam
    for f in theme.FONT_FILES:
        assert (ROOT / "mobilia/ui/fonts" / f).is_file(), f
    for lic in ("OFL-Exo2.txt", "OFL-IBMPlexSans.txt", "OFL-JetBrainsMono.txt"):
        assert "SIL Open Font License" in (ROOT / "mobilia/ui/fonts" / lic).read_text(encoding="utf-8")
    # emblema e cabeçalho: nada em branco
    for pm in (brand.app_icon_pixmap(16), brand.app_icon_pixmap(256), brand.lockup_pixmap(34)):
        img = pm.toImage()
        assert not pm.isNull() and any(img.pixelColor(x, img.height() // 2).alpha() > 200 for x in range(img.width()))
