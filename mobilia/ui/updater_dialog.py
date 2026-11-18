"""Interface da atualização automática (a lógica, sem Qt, fica em ``updater.py``).

* ``attach_updater(janela, barra_de_status)``: põe o botão "Atualizações" na barra de status (menu com "Procurar
  atualizações", a opção de consulta automática e o link das versões) e agenda a consulta automática, em segundo
  plano e no máximo uma vez por dia. Ao rodar pelo código-fonte não faz nada (``updater.updater_enabled``).
* ``UpdateDialog``: janela não modal "Nova versão X.Y.Z disponível" com as notas da versão e os botões
  "Atualizar agora" (portátil: "Baixar nova versão"), "Depois" e "Pular esta versão"; mostra o progresso do
  download, com "Cancelar".
* ``UpdateController``: o fluxo (consulta, oferta, download verificado, instalação silenciosa e saída do app). As
  funções de rede, de download, de instalação e de saída são injetáveis (testes offscreen sem internet).

Visual só com os tokens e nomes de objeto do tema (``theme``), sem cores próprias.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QTextDocument
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QHBoxLayout, QLabel, QMenu, QMessageBox,
                               QProgressBar, QPushButton, QTextBrowser, QVBoxLayout)

from .. import APP_NAME, __version__
from . import theme, updater
from .widgets import on_click


def _button(text: str, kind: str) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName(kind)
    b.setCursor(Qt.PointingHandCursor)
    return b


def _mb(done: int) -> str:
    return f"{done / (1 << 20):.1f} MB"


class UpdateDialog(QDialog):
    """Oferta de atualização (não modal). Os botões só emitem sinais; quem decide é o ``UpdateController``."""

    update_requested = Signal()
    later_requested = Signal()
    skip_requested = Signal()
    cancel_requested = Signal()
    open_requested = Signal()          # portátil: abrir o novo .exe
    folder_requested = Signal()
    auto_check_changed = Signal(bool)

    def __init__(self, release: updater.Release, kind: str = "installed", current: str = __version__,
                 auto_check: bool = True, parent=None):
        super().__init__(parent)
        self.release, self.kind = release, kind
        self.state = "offer"
        self.setWindowTitle(f"{APP_NAME} — atualização")
        self.setWindowIcon(theme.window_icon())
        self.setWindowModality(Qt.NonModal)
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self); lay.setContentsMargins(28, 24, 28, 22); lay.setSpacing(10)

        self.title = QLabel(f"Nova versão {release.version} disponível"); self.title.setObjectName("title")
        when = f", publicada em {release.published_at}" if release.published_at else ""
        self.subtitle = QLabel(f"Você está usando a versão {current}{when}. Seus trabalhos, o motor de IA, os "
                               "modelos e os dados do Lead Finder são mantidos.")
        self.subtitle.setObjectName("body"); self.subtitle.setWordWrap(True)
        lay.addWidget(self.title); lay.addWidget(self.subtitle); lay.addSpacing(4)

        self.notes = QTextBrowser()
        self.notes.setOpenLinks(False)                 # links só pelo navegador, e só do GitHub
        self.notes.anchorClicked.connect(self._open_link)
        self.notes.setMinimumHeight(220)
        # o QTextEdit do tema usa a fonte monoespaçada (registros); as notas são texto corrido
        self.notes.setStyleSheet(f"QTextBrowser {{ font-family: {theme.FONT_UI}; font-size: 13px; "
                                 f"color: {theme.TEXT_2}; padding: 10px 12px; }}")
        self.notes.document().setMarkdown(
            updater.notes_for_display(release.notes),
            QTextDocument.MarkdownFeature.MarkdownNoHTML)  # sem HTML embutido; nada das notas é executado
        lay.addWidget(self.notes, 1)

        self.progress = QProgressBar(); self.progress.setRange(0, 1000); self.progress.setTextVisible(False)
        self.progress.hide()
        self.status = QLabel(""); self.status.setObjectName("hint"); self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status.hide()
        lay.addWidget(self.progress); lay.addWidget(self.status)

        self.chk_auto = QCheckBox("Procurar atualizações automaticamente ao abrir o 7xAi")
        self.chk_auto.setChecked(auto_check)
        self.chk_auto.toggled.connect(self.auto_check_changed)
        lay.addWidget(self.chk_auto)

        row = QHBoxLayout(); row.setSpacing(10)
        self.btn_skip = _button("Pular esta versão", "ghost")
        self.btn_later = _button("Depois", "outline")
        self.btn_cancel = _button("Cancelar", "danger"); self.btn_cancel.hide()
        self.btn_folder = _button("Abrir a pasta", "outline"); self.btn_folder.hide()
        self.btn_update = _button("Baixar nova versão" if kind == "portable" else "Atualizar agora", "primary")
        self.btn_update.setMinimumWidth(180)
        row.addWidget(self.btn_skip); row.addStretch(1)
        for b in (self.btn_cancel, self.btn_folder, self.btn_later, self.btn_update):
            row.addWidget(b)
        lay.addLayout(row)
        on_click(self.btn_update, self._primary)
        on_click(self.btn_later, self.later_requested.emit)
        on_click(self.btn_skip, self.skip_requested.emit)
        on_click(self.btn_cancel, self.cancel_requested.emit)
        on_click(self.btn_folder, self.folder_requested.emit)
        self.btn_update.setDefault(True)

    # ------------------------------------------------------------------------------------------ estados
    def _primary(self) -> None:
        (self.open_requested if self.state == "ready" else self.update_requested).emit()

    def _open_link(self, url: QUrl) -> None:
        if updater.url_allowed(url.toString()):
            QDesktopServices.openUrl(url)

    def _show_status(self, text: str) -> None:
        self.status.setText(text); self.status.setVisible(bool(text))

    def set_offer(self, message: str = "") -> None:
        self.state = "offer"
        self.progress.hide(); self.btn_cancel.hide(); self.btn_folder.hide()
        for b in (self.btn_update, self.btn_later, self.btn_skip):
            b.show(); b.setEnabled(True)
        self._show_status(message)

    def set_downloading(self) -> None:
        self.state = "downloading"
        self.progress.setRange(0, 0); self.progress.show()        # indeterminado até saber o tamanho
        self.btn_update.setEnabled(False); self.btn_skip.hide(); self.btn_later.hide()
        self.btn_cancel.show(); self.btn_cancel.setEnabled(True)
        self._show_status(f"Baixando a versão {self.release.version}…")

    def set_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.progress.setRange(0, 1000); self.progress.setValue(int(1000 * min(done, total) / total))
            self._show_status(f"Baixando a versão {self.release.version}… {_mb(done)} de {_mb(total)}")
        else:
            self._show_status(f"Baixando a versão {self.release.version}… {_mb(done)}")

    def set_installing(self) -> None:
        self.state = "installing"
        self.progress.setRange(0, 0); self.btn_cancel.hide(); self.btn_update.setEnabled(False)
        self._show_status(f"Download conferido (SHA-256). Instalando: o {APP_NAME} vai fechar e abrir de novo "
                          "sozinho em alguns segundos.")

    def set_ready(self, path: Path) -> None:
        """Portátil baixado e conferido: o usuário abre o novo arquivo (um .exe aberto não pode ser trocado)."""
        self.state = "ready"
        self.progress.setRange(0, 1000); self.progress.setValue(1000)
        self.btn_cancel.hide(); self.btn_skip.hide()
        self.btn_later.setText("Fechar"); self.btn_later.show()
        self.btn_folder.show()
        self.btn_update.setText("Abrir a nova versão"); self.btn_update.setEnabled(True)
        self._show_status(f"Pronto: a versão {self.release.version} foi baixada e conferida (SHA-256) em\n{path}\n"
                          f"Feche este {APP_NAME} e abra o novo arquivo (ou clique em \"Abrir a nova versão\"). "
                          "Depois, pode apagar o arquivo antigo.")

    def set_error(self, message: str) -> None:
        self.set_offer(message)
        self.btn_update.setText("Tentar de novo")

    def closeEvent(self, e):  # noqa: N802
        if self.state == "downloading":
            self.cancel_requested.emit()
        super().closeEvent(e)


class _Bridge(QObject):
    checked = Signal(object, bool)        # CheckResult, consulta manual?
    progress = Signal(int, int)
    downloaded = Signal(str)
    failed = Signal(str, bool)            # mensagem, cancelado?


def _thread(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="7xai-updater", daemon=True).start()


class UpdateController(QObject):
    """Fluxo completo da atualização. Tudo que toca a rede roda em ``spawn`` (uma thread), nunca na interface."""

    def __init__(self, window=None, *, settings_path: Path | None = None, enabled: bool | None = None,
                 kind: str | None = None, fetch=None, downloader=None, launcher=None, open_file=None,
                 quit_app: Callable[[], None] | None = None, busy: Callable[[], bool] | None = None,
                 confirm: Callable[[str, str], bool] | None = None, spawn: Callable = _thread,
                 download_dir: Path | None = None, current: str = __version__):
        super().__init__(window)
        self.window = window
        self.enabled = updater.updater_enabled() if enabled is None else enabled
        self.kind = kind or updater.install_kind()
        self.settings_path = settings_path or updater.UpdaterSettings.path()
        self.settings = updater.UpdaterSettings.load(self.settings_path)
        self.fetch = fetch or (lambda: updater.fetch_latest(current=current))
        self.downloader = downloader or updater.download_verified
        self.launcher = launcher or updater.launch_installer
        self.open_file = open_file or (lambda p: updater.launch_detached([str(p)]))
        self.quit_app = quit_app or self._quit
        self.busy = busy or (lambda: False)
        self.confirm = confirm or self._confirm
        self.spawn = spawn
        self.current = current
        self.download_dir = download_dir
        self.dialog: UpdateDialog | None = None
        self.message: QMessageBox | None = None
        self.checking = False
        self.downloaded_path: Path | None = None
        self._cancel = threading.Event()
        self.bridge = _Bridge(self)
        self.bridge.checked.connect(self._on_checked)
        self.bridge.progress.connect(self._on_progress)
        self.bridge.downloaded.connect(self._on_downloaded)
        self.bridge.failed.connect(self._on_failed)

    # ------------------------------------------------------------------------------------------ consulta
    def start_auto_check(self, now: float | None = None) -> bool:
        """Consulta automática (no máximo 1 vez por dia, se estiver ligada). Devolve True se consultou."""
        if not self.enabled:
            return False
        self._cleanup()
        if self.checking or not self.settings.due(now):
            return False
        self._check(manual=False)
        return True

    def check_now(self) -> None:
        """"Procurar atualizações": consulta já, e mostra o resultado mesmo sem versão nova."""
        if self.enabled and not self.checking:
            self._check(manual=True)

    def _check(self, manual: bool) -> None:
        self.checking = True

        def work():
            try:
                res = self.fetch()
            except Exception as e:  # noqa: BLE001 — a consulta nunca derruba o app
                res = updater.CheckResult("error", message=str(e)[:200])
            self.bridge.checked.emit(res, manual)
        self.spawn(work)

    def _cleanup(self) -> None:
        """Apaga (em segundo plano) o instalador já usado numa atualização anterior."""
        def work():
            try:
                updater.cleanup_updates(self.download_dir, self.current)
            except OSError:
                pass
        self.spawn(work)

    def _on_checked(self, res: updater.CheckResult, manual: bool) -> None:
        self.checking = False
        self.settings.record_check(res)
        self.settings.save(self.settings_path)
        if self.settings.should_offer(res, manual):
            self.show_offer(res.release)
        elif manual:
            texts = {"current": f"Você já está usando a versão mais recente do {APP_NAME} ({self.current}).",
                     "offline": "Não foi possível falar com o GitHub (sem internet?). Tente de novo mais tarde.",
                     "rate_limited": "O GitHub limitou as consultas por agora. Tente de novo daqui a uma hora."}
            self._notify(texts.get(res.status, f"Não foi possível procurar atualizações. {res.message}".strip()))

    def _notify(self, text: str) -> None:
        box = QMessageBox(QMessageBox.Information, f"{APP_NAME} — atualizações", text, QMessageBox.Ok, self.window)
        box.setWindowModality(Qt.NonModal)
        box.setAttribute(Qt.WA_DeleteOnClose)
        box.show()
        self.message = box

    # ------------------------------------------------------------------------------------------ oferta
    def show_offer(self, release: updater.Release) -> UpdateDialog:
        if self.dialog is not None:
            self.dialog.close()
        d = UpdateDialog(release, self.kind, self.current, self.settings.auto_check, self.window)
        d.update_requested.connect(self.start_update)
        d.later_requested.connect(d.close)
        d.skip_requested.connect(self.skip)
        d.cancel_requested.connect(self.cancel)
        d.open_requested.connect(self.open_downloaded)
        d.folder_requested.connect(self.open_folder)
        d.auto_check_changed.connect(self.set_auto_check)
        self.dialog = d
        d.show(); d.raise_()
        return d

    def set_auto_check(self, on: bool) -> None:
        self.settings.auto_check = bool(on)
        self.settings.save(self.settings_path)
        act = getattr(self, "action_auto", None)
        if act is not None and act.isChecked() != bool(on):
            act.setChecked(bool(on))

    def skip(self) -> None:
        if self.dialog:
            self.settings.skipped_version = self.dialog.release.version
            self.settings.save(self.settings_path)
            self.dialog.close()

    # ------------------------------------------------------------------------------------------ download
    def _target_dir(self) -> Path:
        if self.download_dir:
            return Path(self.download_dir)
        return updater.portable_target_dir() if self.kind == "portable" else updater.updates_dir()

    def start_update(self) -> None:
        d = self.dialog
        if d is None or d.state == "downloading":
            return
        if self.kind != "portable" and self.busy() and not self.confirm(
                "Trabalho em andamento",
                f"Um trabalho está em andamento e será interrompido: o {APP_NAME} precisa fechar para instalar a "
                "atualização. Atualizar mesmo assim?"):
            return
        self._cancel.clear()
        d.set_downloading()
        release, kind, folder = d.release, updater.asset_kind(self.kind), self._target_dir()

        def work():
            try:
                path = self.downloader(release, kind, folder, progress=self.bridge.progress.emit,
                                       cancelled=self._cancel.is_set)
                self.bridge.downloaded.emit(str(path))
            except updater.DownloadCancelled:
                self.bridge.failed.emit("", True)
            except updater.UpdateError as e:
                self.bridge.failed.emit(str(e), False)
            except Exception as e:  # noqa: BLE001
                self.bridge.failed.emit(f"Não foi possível baixar a atualização: {e}", False)
        self.spawn(work)

    def cancel(self) -> None:
        self._cancel.set()

    def _on_progress(self, done: int, total: int) -> None:
        if self.dialog:
            self.dialog.set_progress(done, total)

    def _on_failed(self, message: str, cancelled: bool) -> None:
        if self.dialog:
            if cancelled:
                self.dialog.set_offer("Download cancelado.")
            else:
                self.dialog.set_error(message)

    def _on_downloaded(self, path: str) -> None:
        self.downloaded_path = Path(path)
        if self.dialog is None:
            return
        if self.kind == "portable":
            self.dialog.set_ready(self.downloaded_path)
            return
        self.dialog.set_installing()
        try:
            self.launcher(self.downloaded_path)
        except OSError as e:
            self.dialog.set_error(f"Não foi possível abrir o instalador: {e}")
            return
        QTimer.singleShot(400, self.quit_app)     # fecha para o instalador trocar os arquivos

    def open_downloaded(self) -> None:
        if self.downloaded_path is None:
            return
        try:
            self.open_file(self.downloaded_path)
        except OSError as e:
            if self.dialog:
                self.dialog._show_status(f"Não foi possível abrir {self.downloaded_path}: {e}")
            return
        QTimer.singleShot(400, self.quit_app)

    def open_folder(self) -> None:
        if self.downloaded_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.downloaded_path.parent)))

    # ------------------------------------------------------------------------------------------ apoio
    def _confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self.window, title, text) == QMessageBox.Yes

    def _quit(self) -> None:
        if self.window is not None:
            self.window.close()
        QApplication.quit()


def attach_updater(window, statusbar, busy: Callable[[], bool] | None = None, **kw) -> UpdateController:
    """Liga a atualização à janela principal: botão "Atualizações" na barra de status e consulta automática
    alguns segundos depois de abrir. Pelo código-fonte (sem ``MOBILIA_UPDATER_TEST=1``) não acrescenta nada."""
    ctl = UpdateController(window, busy=busy, **kw)
    if not ctl.enabled:
        return ctl
    btn = QPushButton("Atualizações"); btn.setObjectName("link"); btn.setCursor(Qt.PointingHandCursor)
    btn.setToolTip("Procurar uma versão nova do 7xAi no GitHub")
    menu = QMenu(btn)
    menu.setStyleSheet(f"QMenu {{ background: {theme.SURFACE}; color: {theme.TEXT}; border: 1px solid "
                       f"{theme.HAIRLINE_STRONG}; padding: 4px; }} QMenu::item {{ padding: 6px 18px; }} "
                       f"QMenu::item:selected {{ background: {theme.SELECTED_BG}; color: {theme.ACCENT_HI}; }}")
    act_check = QAction("Procurar atualizações", menu)
    act_check.triggered.connect(lambda *_: ctl.check_now())
    act_auto = QAction("Procurar automaticamente ao abrir", menu)
    act_auto.setCheckable(True); act_auto.setChecked(ctl.settings.auto_check)
    act_auto.toggled.connect(ctl.set_auto_check)
    act_web = QAction("Ver todas as versões no GitHub", menu)
    act_web.triggered.connect(lambda *_: QDesktopServices.openUrl(QUrl(updater.RELEASES_URL)))
    for a in (act_check, act_auto, act_web):
        menu.addAction(a)
    btn.setMenu(menu)
    statusbar.addPermanentWidget(btn)
    ctl.button, ctl.menu, ctl.action_check, ctl.action_auto = btn, menu, act_check, act_auto
    timer = QTimer(ctl); timer.setSingleShot(True)          # filho do controller: morre junto com a janela
    timer.timeout.connect(ctl.start_auto_check)
    timer.start(5000)
    ctl.auto_timer = timer
    return ctl
