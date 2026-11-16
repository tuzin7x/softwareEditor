"""Janela principal do Mobília Studio (PySide6).

Fluxo único e simples: 1) importar a foto  2) escolher o ambiente  3) gerar  4) assistir e exportar.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QMimeData, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__, bootstrap, diagnostics, runtime
from ..jobs import ROOM_TYPES, STAGES, JobRequest
from . import theme
from .runner import HardwareProbe, JobRunner, QueuedJob

PHOTO_FILTER = "Fotos (*.jpg *.jpeg *.png *.webp *.tif *.tiff *.heic)"


class ImageView(QLabel):
    """Mostra uma imagem inteira, preservando a proporção, em qualquer tamanho."""

    def __init__(self, text: str = ""):
        super().__init__(text)
        self._pm: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, path: str) -> bool:
        pm = QPixmap(path)
        if pm.isNull():
            return False
        self._pm = pm
        self._rescale()
        return True

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self._pm is not None:
            self.setPixmap(self._pm.scaled(self.size() - QSize(8, 8), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class DropZone(QFrame):
    """Área de importação: clique ou arraste uma foto."""

    def __init__(self, on_file):
        super().__init__()
        self.setObjectName("drop")
        self.on_file = on_file
        self.setAcceptDrops(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 18)
        self.view = ImageView()
        self.hint = QLabel("Arraste a foto do ambiente vazio aqui\nou clique em “Importar foto”")
        self.hint.setObjectName("subtitle")
        self.hint.setAlignment(Qt.AlignCenter)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.hint)
        self.stack.addWidget(self.view)
        lay.addWidget(self.stack)

    def show_photo(self, path: str) -> bool:
        ok = self.view.set_image(path)
        self.stack.setCurrentIndex(1 if ok else 0)
        return ok

    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setProperty("hover", True); self.style().polish(self)

    def dragLeaveEvent(self, e):  # noqa: N802
        self.setProperty("hover", False); self.style().polish(self)

    def dropEvent(self, e):  # noqa: N802
        self.setProperty("hover", False); self.style().polish(self)
        for u in e.mimeData().urls():
            if u.isLocalFile():
                self.on_file(u.toLocalFile())
                break

    def mousePressEvent(self, e):  # noqa: N802
        path, _ = QFileDialog.getOpenFileName(self, "Importar foto", "", PHOTO_FILTER)
        if path:
            self.on_file(path)


class VideoView(QWidget):
    """Player em loop do vídeo gerado (QtMultimedia)."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.player = None
        try:
            from PySide6.QtMultimedia import QMediaPlayer
            from PySide6.QtMultimediaWidgets import QVideoWidget

            self.widget = QVideoWidget()
            self.player = QMediaPlayer(self)
            self.player.setVideoOutput(self.widget)
            self.player.setLoops(QMediaPlayer.Infinite)
            lay.addWidget(self.widget)
        except Exception:  # sem backend de mídia: mostra a última prévia no lugar
            self.widget = ImageView("Pré-visualização indisponível neste sistema")
            lay.addWidget(self.widget)
        self.source: str | None = None

    def play(self, path: str) -> None:
        self.source = path
        if self.player is not None:
            self.player.setSource(QUrl.fromLocalFile(path))
            self.player.play()

    def stop(self) -> None:
        if self.player is not None:
            self.player.stop()


class StageList(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.rows = {}
        for key, label, _ in STAGES:
            lb = QLabel(f"○  {label}")
            lb.setObjectName("muted")
            lay.addWidget(lb)
            self.rows[key] = (lb, label)
        self.order = [k for k, _, _ in STAGES]

    def set_current(self, stage: str) -> None:
        if stage not in self.rows:
            return
        idx = self.order.index(stage)
        for i, k in enumerate(self.order):
            lb, label = self.rows[k]
            if i < idx:
                lb.setText(f"✓  {label}"); lb.setStyleSheet(f"color: {theme.OK};")
            elif i == idx:
                lb.setText(f"●  {label}"); lb.setStyleSheet(f"color: {theme.ACCENT}; font-weight: 600;")
            else:
                lb.setText(f"○  {label}"); lb.setStyleSheet(f"color: {theme.MUTED};")

    def reset(self) -> None:
        for k in self.order:
            lb, label = self.rows[k]
            lb.setText(f"○  {label}"); lb.setStyleSheet(f"color: {theme.MUTED};")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}")
        self.setWindowIcon(QIcon(theme.app_icon()))
        self.resize(1360, 820)
        self.photo_path: str | None = None
        self.room: str | None = None
        self.runner = JobRunner(self)
        self.runner.progress.connect(self._on_progress)
        self.runner.forecast.connect(self._on_forecast)
        self.runner.preview.connect(self._on_preview)
        self.runner.finished.connect(self._on_finished)
        self.started_at = 0.0
        self.last_output: str | None = None
        self.last_job: QueuedJob | None = None
        self.last_diagnostic: Path | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        root = QWidget(); root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(28, 22, 28, 18)
        outer.setSpacing(16)
        head = QHBoxLayout()
        logo = QLabel(); logo.setPixmap(theme.app_icon(44))
        titles = QVBoxLayout(); titles.setSpacing(0)
        t = QLabel(APP_NAME); t.setObjectName("title")
        st = QLabel("Decoração virtual em vídeo — a sua foto, intacta, com os móveis surgindo dentro dela")
        st.setObjectName("subtitle")
        titles.addWidget(t); titles.addWidget(st)
        head.addWidget(logo); head.addLayout(titles); head.addStretch(1)
        outer.addLayout(head)

        body = QHBoxLayout(); body.setSpacing(18)
        outer.addLayout(body, 1)
        # ---- esquerda: foto / vídeo
        self.left = QStackedWidget()
        self.drop = DropZone(self.load_photo)
        self.video = VideoView()
        vcard = QFrame(); vcard.setObjectName("card")
        vl = QVBoxLayout(vcard); vl.setContentsMargins(10, 10, 10, 10); vl.addWidget(self.video)
        self.live = ImageView()
        lcard = QFrame(); lcard.setObjectName("card")
        ll = QVBoxLayout(lcard); ll.setContentsMargins(10, 10, 10, 10); ll.addWidget(self.live)
        self.left.addWidget(self.drop); self.left.addWidget(lcard); self.left.addWidget(vcard)
        body.addWidget(self.left, 3)
        # ---- direita: passos / progresso / resultado
        self.right = QStackedWidget()
        self.right.setFixedWidth(400)
        self.right.addWidget(self._build_setup())
        self.right.addWidget(self._build_progress())
        self.right.addWidget(self._build_result())
        body.addWidget(self.right)

        self.statusBar().showMessage("Detectando GPU…")
        self.probe = HardwareProbe(self)
        self.probe.done.connect(self._on_hardware)
        self._update_ready()

    # ------------------------------------------------------------------ páginas
    def _build_setup(self) -> QWidget:
        w = QFrame(); w.setObjectName("card")
        lay = QVBoxLayout(w); lay.setContentsMargins(22, 22, 22, 22); lay.setSpacing(12)
        s1 = QLabel("1  ·  FOTO"); s1.setObjectName("section")
        self.btn_import = QPushButton("Importar foto…")
        self.btn_import.clicked.connect(lambda: self.drop.mousePressEvent(None))
        self.photo_label = QLabel("Nenhuma foto selecionada"); self.photo_label.setObjectName("muted")
        self.photo_label.setWordWrap(True)
        s2 = QLabel("2  ·  AMBIENTE"); s2.setObjectName("section")
        grid = QGridLayout(); grid.setSpacing(10)
        self.room_group = QButtonGroup(self); self.room_group.setExclusive(True)
        self.room_buttons = {}
        for i, (key, label) in enumerate(ROOM_TYPES.items()):
            b = QToolButton(); b.setObjectName("room"); b.setCheckable(True)
            b.setText(label); b.setIcon(theme.room_icon(key)); b.setIconSize(QSize(40, 40))
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon); b.setMinimumSize(108, 92)
            b.clicked.connect(lambda _=False, k=key: self.select_room(k))
            self.room_group.addButton(b); self.room_buttons[key] = b
            grid.addWidget(b, i // 3, i % 3)
        s3 = QLabel("3  ·  QUALIDADE"); s3.setObjectName("section")
        self.quality = QComboBox()
        self.quality.addItem("Final (recomendado com GPU)", "final")
        self.quality.addItem("Prévia rápida", "previa")
        self.quality.addItem("Automática", "auto")
        self.btn_generate = QPushButton("Gerar vídeo"); self.btn_generate.setObjectName("primary")
        self.btn_generate.clicked.connect(self.generate)
        lay.addWidget(s1); lay.addWidget(self.btn_import); lay.addWidget(self.photo_label)
        lay.addSpacing(6); lay.addWidget(s2); lay.addLayout(grid)
        lay.addSpacing(6); lay.addWidget(s3); lay.addWidget(self.quality)
        lay.addStretch(1); lay.addWidget(self.btn_generate)
        return w

    def _build_progress(self) -> QWidget:
        w = QFrame(); w.setObjectName("card")
        lay = QVBoxLayout(w); lay.setContentsMargins(22, 22, 22, 22); lay.setSpacing(12)
        s = QLabel("GERANDO O VÍDEO"); s.setObjectName("section")
        self.bar = QProgressBar(); self.bar.setRange(0, 1000); self.bar.setTextVisible(False)
        self.pct = QLabel("0%"); self.pct.setStyleSheet("font-size: 28px; font-weight: 600;")
        self.msg = QLabel("Preparando…"); self.msg.setWordWrap(True)
        self.elapsed = QLabel(""); self.elapsed.setObjectName("muted")
        # previsão do tempo total (controle de orçamento do render, PT7): aparece após o quadro final
        self.forecast = QLabel(""); self.forecast.setObjectName("muted"); self.forecast.hide()
        # aviso "primeira execução — mais lenta" (estado da instalação, ver bootstrap.first_run_notice)
        self.notice = QLabel(""); self.notice.setObjectName("notice"); self.notice.setWordWrap(True)
        self.notice.hide()
        self.stages = StageList()
        self.btn_cancel = QPushButton("Cancelar"); self.btn_cancel.setObjectName("danger")
        self.btn_cancel.clicked.connect(self.cancel)
        lay.addWidget(s); lay.addWidget(self.pct); lay.addWidget(self.bar); lay.addWidget(self.msg)
        lay.addWidget(self.elapsed); lay.addWidget(self.forecast); lay.addWidget(self.notice); lay.addSpacing(8); lay.addWidget(self.stages)
        lay.addStretch(1)
        lay.addWidget(self.btn_cancel)
        return w

    def _build_result(self) -> QWidget:
        w = QFrame(); w.setObjectName("card")
        lay = QVBoxLayout(w); lay.setContentsMargins(22, 22, 22, 22); lay.setSpacing(12)
        s = QLabel("PRONTO"); s.setObjectName("section")
        self.result_info = QLabel(""); self.result_info.setWordWrap(True)
        # narrativa de designer (PT12): por que cada peça foi escolhida e onde ficou
        self.notes_title = QLabel("ESCOLHAS DO DESIGNER"); self.notes_title.setObjectName("section")
        self.result_notes = QLabel(""); self.result_notes.setWordWrap(True)
        self.result_notes.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.result_notes.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        st = QLabel("TEMPO POR ETAPA"); st.setObjectName("section")
        self.result_times = QLabel(""); self.result_times.setObjectName("muted"); self.result_times.setWordWrap(True)
        self.result_times.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)   # nunca cortado pela narrativa
        self.btn_export = QPushButton("Exportar MP4…"); self.btn_export.setObjectName("primary")
        self.btn_export.clicked.connect(self.export)
        self.btn_folder = QPushButton("Abrir pasta"); self.btn_folder.clicked.connect(self.open_folder)
        self.btn_diag = QPushButton("Copiar diagnóstico")
        self.btn_diag.setToolTip("Cria um .zip com o relatório de tempos e os registros desta geração "
                                 "e copia o arquivo para a área de transferência.")
        self.btn_diag.clicked.connect(lambda: self.copy_diagnostic())
        self.chk_diag_photo = QCheckBox("Incluir a foto no diagnóstico")
        self.btn_new = QPushButton("Nova foto"); self.btn_new.clicked.connect(self.new_photo)
        lay.addWidget(s); lay.addWidget(self.result_info); lay.addSpacing(6)
        self.notes_scroll = QScrollArea(); self.notes_scroll.setWidgetResizable(True)
        self.notes_scroll.setFrameShape(QFrame.NoFrame); self.notes_scroll.setWidget(self.result_notes)
        self.notes_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.notes_scroll.setMinimumHeight(110)
        self.notes_scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        lay.addWidget(self.notes_title); lay.addWidget(self.notes_scroll, 1); lay.addSpacing(6)
        lay.addWidget(st); lay.addWidget(self.result_times)
        lay.addWidget(self.btn_export); lay.addWidget(self.btn_folder)
        lay.addWidget(self.btn_diag); lay.addWidget(self.chk_diag_photo); lay.addWidget(self.btn_new)
        return w

    # ------------------------------------------------------------------ ações
    def showEvent(self, e):  # noqa: N802
        super().showEvent(e)
        if not getattr(self, "_probed", False):
            self._probed = True
            self.probe.start()

    def load_photo(self, path: str) -> None:
        if not Path(path).is_file() or not self.drop.show_photo(path):
            QMessageBox.warning(self, APP_NAME, "Não foi possível abrir essa imagem.")
            return
        self.photo_path = path
        self.photo_label.setText(Path(path).name)
        self.left.setCurrentIndex(0)
        self._update_ready()

    def select_room(self, key: str) -> None:
        self.room = key
        self.room_buttons[key].setChecked(True)
        self._update_ready()

    def _update_ready(self) -> None:
        self.btn_generate.setEnabled(bool(self.photo_path and self.room) and not self.runner.busy())

    def generate(self) -> None:
        if not (self.photo_path and self.room):
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        work = runtime.runtime_paths().jobs_dir / f"{stamp}_{Path(self.photo_path).stem}"
        out = work / f"{Path(self.photo_path).stem}_{self.room}.mp4"
        req = JobRequest(self.photo_path, self.room, str(out), quality=self.quality.currentData(),
                         work_dir=str(work))
        self.stages.reset(); self.bar.setValue(0); self.pct.setText("0%"); self.msg.setText("Iniciando o motor…")
        self.forecast.setText(""); self.forecast.hide()
        self.live.set_image(self.photo_path)
        notice = self.first_run_notice()
        self.notice.setText(notice or ""); self.notice.setVisible(bool(notice))
        self.left.setCurrentIndex(1); self.right.setCurrentIndex(1)
        self.started_at = time.monotonic(); self.timer.start(1000)
        self.runner.submit(req)
        self._update_ready()

    @staticmethod
    def first_run_notice() -> str | None:
        """Texto do aviso de primeira execução (None nas execuções seguintes)."""
        try:
            return bootstrap.first_run_notice(bootstrap.load_state(runtime.runtime_paths().state_file))
        except Exception:  # aviso é cosmético: nunca impede a geração
            return None

    def cancel(self) -> None:
        self.btn_cancel.setEnabled(False)
        self.msg.setText("Cancelando…")
        self.runner.cancel()

    def export(self) -> None:
        if not self.last_output:
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Exportar vídeo", Path(self.last_output).name, "Vídeo MP4 (*.mp4)")
        if dest:
            shutil.copyfile(self.last_output, dest)
            self.statusBar().showMessage(f"Vídeo exportado: {dest}", 8000)

    def open_folder(self) -> None:
        if self.last_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_output).parent)))

    def copy_diagnostic(self, include_photo: bool | None = None, job: QueuedJob | None = None) -> Path | None:
        """Gera o zip de diagnóstico do último trabalho (sem a foto, salvo se marcado) e copia o
        arquivo para a área de transferência (colar no e-mail/WhatsApp anexa o zip)."""
        job = job or self.last_job
        work = job.work_dir if job else None
        if not work or not Path(work).is_dir():
            self.statusBar().showMessage("Nenhum trabalho para diagnosticar.", 6000)
            return None
        if include_photo is None:
            include_photo = self.chk_diag_photo.isChecked()
        dest = runtime.runtime_paths().home / "diagnosticos" / f"diagnostico_{Path(work).name}.zip"
        try:
            path = diagnostics.make_diagnostic_zip(work, dest, include_photo=include_photo,
                                                   app_version=__version__)
        except OSError as e:
            self.statusBar().showMessage(f"Não foi possível criar o diagnóstico: {e}", 10000)
            return None
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(path))])
        mime.setText(str(path))
        QApplication.clipboard().setMimeData(mime)
        self.last_diagnostic = path
        self.statusBar().showMessage(f"Diagnóstico copiado ({'com' if include_photo else 'sem'} a foto): {path}",
                                     15000)
        return path

    def new_photo(self) -> None:
        self.video.stop()
        self.photo_path = None
        self.photo_label.setText("Nenhuma foto selecionada")
        self.drop.stack.setCurrentIndex(0)
        self.left.setCurrentIndex(0); self.right.setCurrentIndex(0)
        self._update_ready()

    # ------------------------------------------------------------------ eventos do worker
    def _tick(self) -> None:
        s = int(time.monotonic() - self.started_at)
        self.elapsed.setText(f"Tempo decorrido: {s // 60:d} min {s % 60:02d} s")

    def _on_progress(self, stage: str, frac: float, message: str) -> None:
        self.bar.setValue(int(frac * 1000)); self.pct.setText(f"{frac * 100:.0f}%")
        self.msg.setText(message); self.stages.set_current(stage)

    def _on_forecast(self, total_s: float, message: str) -> None:
        self.forecast.setText(message or f"Previsão: ≈ {int(round(total_s))} s no total")
        self.forecast.show()

    def _on_preview(self, path: str) -> None:
        if path and Path(path).exists():
            self.live.set_image(path)

    def _on_finished(self, job: QueuedJob) -> None:
        self.timer.stop()
        self.btn_cancel.setEnabled(True)
        self.last_job = job
        self._update_ready()
        if job.status == "pronto" and job.output and Path(job.output).exists():
            self.last_output = job.output
            mb = Path(job.output).stat().st_size / 1e6
            self.result_info.setText(f"{Path(job.output).name}\n{mb:.1f} MB · {self.elapsed.text().lower()}\n\n"
                                     "Fora dos móveis e das sombras, a foto original foi preservada "
                                     "pixel a pixel (verificado em cada quadro).")
            self.result_times.setText(self._times_text(job))
            # altura mínima do texto (linhas + 1 de quebra): a narrativa rola, os tempos não são cortados
            n = self.result_times.text().count("\n") + 2
            self.result_times.setMinimumHeight(self.result_times.fontMetrics().lineSpacing() * n)
            notes = self._designer_notes(job)
            self.result_notes.setText("\n".join(f"•  {n}" for n in notes))
            self.notes_title.setVisible(bool(notes)); self.notes_scroll.setVisible(bool(notes))
            self.left.setCurrentIndex(2); self.right.setCurrentIndex(2)
            self.video.play(job.output)
            return
        self.left.setCurrentIndex(0); self.right.setCurrentIndex(0)
        if job.status == "cancelado":
            self.statusBar().showMessage("Geração cancelada.", 6000)
            return
        self.show_error(job.error or "Erro desconhecido.", job.detail)

    @staticmethod
    def _times_text(job: QueuedJob) -> str:
        if not job.report:
            return ""
        try:
            rep = json.loads(Path(job.report).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        return diagnostics.summary_text(rep)

    @staticmethod
    def _designer_notes(job: QueuedJob) -> list[str]:
        """Narrativa de designer do trabalho: report.json (``narrativa``) ou, sem ela, layout.json."""
        srcs = []
        if job.report:
            srcs.append((Path(job.report), lambda d: d.get("narrativa") or (d.get("layout") or {}).get("notes")))
        if job.work_dir:
            srcs.append((Path(job.work_dir) / "layout.json", lambda d: d.get("notes")))
        for path, pick in srcs:
            try:
                notes = pick(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, AttributeError):
                continue
            if notes:
                return [str(n) for n in notes]
        return []

    def show_error(self, message: str, detail: str = "") -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(APP_NAME)
        box.setText("Não foi possível gerar o vídeo.")
        box.setInformativeText(message)
        if detail:
            box.setDetailedText(detail)
        diag = None
        if self.last_job is not None and self.last_job.work_dir and Path(self.last_job.work_dir).is_dir():
            diag = box.addButton("Copiar diagnóstico", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if diag is not None and box.clickedButton() is diag:
            self.copy_diagnostic()

    def _on_hardware(self, info) -> None:
        if not info:
            self.statusBar().showMessage("GPU: não detectada (o motor ainda não está instalado ou não respondeu).")
            return
        prof = info.get("profile", {})
        gpus = info.get("gpus") or []
        gpu = max(gpus, key=lambda g: g.get("vram_total_mb", 0)) if gpus else None
        g = f"{gpu['name']} · {gpu['vram_total_mb'] / 1024:.0f} GB" if gpu else "sem GPU NVIDIA (modo CPU)"
        self.statusBar().showMessage(f"GPU: {g}   ·   Perfil: {prof.get('name', '?')} — {prof.get('description', '')}")
        if gpu is None:
            self.quality.setCurrentIndex(1)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("--worker", "worker"):
        from ..cli import main as cli_main

        return cli_main(argv[1:])
    selftest = "--selftest" in argv
    if selftest:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if "--version" in argv:
        print(f"{APP_NAME} {__version__}")
        return 0
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(theme.QSS)
    if not selftest and runtime.uses_embedded_runtime():
        from .setup_dialog import ensure_runtime

        if not ensure_runtime(None):
            return 1
    win = MainWindow()
    if selftest:
        win.show()
        app.processEvents()
        ok = win.windowTitle() == APP_NAME and not theme.app_icon().isNull()
        print("selftest ok" if ok else "selftest FALHOU")
        return 0 if ok else 2
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
