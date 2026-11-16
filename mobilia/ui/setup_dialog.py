"""Assistente de primeiro uso: instala o motor (Python + PyTorch + Blender + modelos) com progresso real."""
from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
                               QTextEdit, QVBoxLayout)

from .. import APP_NAME, bootstrap, runtime


def intro_text(state: dict) -> tuple[str, str]:
    """Título e texto do assistente: instalação nova ou atualização (só as etapas pendentes)."""
    if not state.get("done"):
        return ("Primeiro uso: vamos instalar o motor de IA e renderização",
                "Tudo roda no seu computador, na sua GPU NVIDIA. O download (~7 GB) acontece uma única "
                "vez; depois o Mobília Studio funciona sem internet. No fim, a GPU é preparada (kernels de "
                "render e modelos de IA) para que o primeiro vídeo já saia rápido. Se a conexão cair, é só "
                "abrir de novo: a instalação continua de onde parou.")
    # só a inicial em minúscula: siglas e nomes próprios (GPU, IA, PyTorch, Mobília Studio) ficam
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
        self.setWindowTitle(f"{APP_NAME} — instalação do motor")
        self.setMinimumWidth(620)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 22, 24, 22)
        state = bootstrap.sync_with_payload(bootstrap.load_state(runtime.runtime_paths().state_file),
                                            bootstrap.payload_fingerprints(runtime.payload_root()))
        title, text = intro_text(state)
        t = QLabel(title)
        t.setObjectName("title")
        info = QLabel(text)
        info.setWordWrap(True); info.setObjectName("subtitle")
        self.step = QLabel("Pronto para começar.")
        self.bar = QProgressBar(); self.bar.setRange(0, 1000)
        self.detail = QTextEdit(); self.detail.setReadOnly(True); self.detail.setFixedHeight(140)
        self.btn = QPushButton("Instalar agora"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self.start)
        cancel = QPushButton("Sair"); cancel.clicked.connect(self.reject)
        row = QHBoxLayout(); row.addStretch(1); row.addWidget(cancel); row.addWidget(self.btn)
        for w in (t, info, self.step, self.bar, self.detail):
            lay.addWidget(w)
        lay.addLayout(row)
        self.bridge = _Bridge()
        self.bridge.progress.connect(self._on_progress)
        self.bridge.done.connect(self._on_done)
        self.ok = False

    def start(self) -> None:
        self.btn.setEnabled(False)
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
