"""Assistente de primeiro uso: instala o motor (Python + PyTorch + Blender + modelos) com progresso real."""
from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
                               QTextEdit, QVBoxLayout)

from .. import APP_NAME, bootstrap, runtime
from . import brand, theme


def intro_text(state: dict) -> tuple[str, str]:
    """Título e texto do assistente: instalação nova ou atualização (só as etapas pendentes)."""
    if not state.get("done"):
        return ("Primeiro uso: vamos instalar o motor de IA e renderização",
                "Tudo roda no seu computador, na sua GPU NVIDIA. O download (~7 GB) acontece uma única "
                f"vez; depois o {APP_NAME} funciona sem internet. No fim, a GPU é preparada (kernels de "
                "render e modelos de IA) para que o primeiro vídeo já saia rápido. Se a conexão cair, é só "
                "abrir de novo: a instalação continua de onde parou.")
    # só a inicial em minúscula: siglas e nomes próprios (GPU, IA, PyTorch, 7xAi) ficam
    labels = [bootstrap.STEP_LABELS[s].split(" (")[0] for s in bootstrap.pending_steps(state)]
    pend = [lb[:1].lower() + lb[1:] for lb in labels]
    return ("Atualização: preparando o motor para esta versão",
            "Só falta: " + "; ".join(pend) + ". O que já está instalado não é baixado de novo.")


class _Bridge(QObject):
    progress = Signal(str, float, str)
    done = Signal(bool, str)


class SetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        theme.load_fonts()
        self.setWindowTitle(f"{APP_NAME} — instalação do motor")
        self.setWindowIcon(theme.window_icon())
        self.setMinimumWidth(640)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 28, 32, 28)
        lay.setSpacing(0)
        state = bootstrap.sync_with_payload(bootstrap.load_state(runtime.runtime_paths().state_file),
                                            bootstrap.payload_fingerprints(runtime.payload_root()))
        title, text = intro_text(state)
        head = QHBoxLayout(); head.setSpacing(0)
        mark = QLabel(); mark.setPixmap(brand.wordmark_pixmap(20)); mark.setAccessibleName(APP_NAME)
        rule = QFrame(); rule.setObjectName("vrule"); rule.setFixedSize(1, 18)
        cap = QLabel("INSTALAÇÃO DO MOTOR"); cap.setObjectName("section"); theme.tracked(cap)
        head.addWidget(mark, 0, Qt.AlignVCenter); head.addSpacing(18); head.addWidget(rule, 0, Qt.AlignVCenter)
        head.addSpacing(18); head.addWidget(cap, 0, Qt.AlignVCenter); head.addStretch(1)
        t = QLabel(title)
        t.setObjectName("title"); t.setWordWrap(True)
        info = QLabel(text)
        info.setWordWrap(True); info.setObjectName("subtitle")
        card = QFrame(); card.setObjectName("card")
        cl = QVBoxLayout(card); cl.setContentsMargins(20, 18, 20, 20); cl.setSpacing(0)
        top = QHBoxLayout()
        self.step = QLabel("Pronto para começar."); self.step.setStyleSheet("font-weight: 600;")
        self.step.setWordWrap(True)
        self.pct = QLabel(""); self.pct.setObjectName("hint")
        top.addWidget(self.step, 1); top.addWidget(self.pct, 0, Qt.AlignTop)
        self.bar = QProgressBar(); self.bar.setRange(0, 1000); self.bar.setTextVisible(False)
        self.detail = QTextEdit(); self.detail.setReadOnly(True); self.detail.setFixedHeight(140)
        self.detail.setPlaceholderText("Os detalhes da instalação aparecem aqui.")
        cl.addLayout(top); cl.addSpacing(12); cl.addWidget(self.bar); cl.addSpacing(16); cl.addWidget(self.detail)
        self.btn = QPushButton("Instalar agora"); self.btn.setObjectName("primary")
        self.btn.setMinimumWidth(190); self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.clicked.connect(self.start)
        cancel = QPushButton("Sair"); cancel.setObjectName("ghost"); cancel.setMinimumWidth(96)
        cancel.clicked.connect(self.reject)
        row = QHBoxLayout(); row.setSpacing(10); row.addStretch(1); row.addWidget(cancel); row.addWidget(self.btn)
        lay.addLayout(head); lay.addSpacing(30)
        lay.addWidget(t); lay.addSpacing(10); lay.addWidget(info); lay.addSpacing(24)
        lay.addWidget(card); lay.addSpacing(24)
        lay.addLayout(row)
        self.bridge = _Bridge()
        self.bridge.progress.connect(self._on_progress)
        self.bridge.done.connect(self._on_done)
        self.ok = False

    def start(self) -> None:
        self.btn.setEnabled(False); self.btn.setText("Instalando…")
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        try:
            bootstrap.install(lambda s, f, m: self.bridge.progress.emit(s, f, m))
            self.bridge.done.emit(True, "")
        except Exception as e:  # noqa: BLE001 — mostrado ao usuário com detalhes
            self.bridge.done.emit(False, f"{e}\n\n{traceback.format_exc()}")

    def _on_progress(self, step: str, frac: float, msg: str) -> None:
        i = bootstrap.STEPS.index(step) if step in bootstrap.STEPS else 0
        self.step.setText(f"Etapa {i + 1} de {len(bootstrap.STEPS)}: {bootstrap.STEP_LABELS.get(step, step)}")
        if frac >= 0:
            self.bar.setValue(int(1000 * (i + frac) / len(bootstrap.STEPS)))
            self.pct.setText(f"{self.bar.value() / 10:.0f}%")
        if msg:
            self.detail.append(msg)

    def _on_done(self, ok: bool, err: str) -> None:
        self.ok = ok
        if ok:
            self.accept()
            return
        self.btn.setEnabled(True); self.btn.setText("Tentar de novo")
        box = QMessageBox(self); box.setIcon(QMessageBox.Warning); box.setWindowTitle(APP_NAME)
        box.setText("A instalação do motor não terminou.")
        box.setInformativeText(err.split("\n\n")[0][:600])
        box.setDetailedText(err)
        box.exec()


def ensure_runtime(parent) -> bool:
    if bootstrap.runtime_ready():
        return True
    dlg = SetupDialog(parent)
    dlg.setWindowModality(Qt.ApplicationModal)
    return dlg.exec() == QDialog.Accepted and bootstrap.runtime_ready()
