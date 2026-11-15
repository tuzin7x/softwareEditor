"""Contrato de execução de um trabalho (foto -> vídeo) e protocolo de progresso.

A interface gráfica NUNCA roda o pipeline no próprio processo: ela inicia o
processo de trabalho (``mobilia-worker run ...`` ou, no executável congelado,
``MobiliaStudio.exe --worker run ...``) e lê eventos JSON, um por linha, na saída
padrão. Isso isola falhas, libera a VRAM ao final e permite cancelar com segurança.

Eventos (um objeto JSON por linha, campo "event"):
  {"event": "progress", "stage": "analise", "fraction": 0.12, "message": "..."}
  {"event": "preview", "path": "C:/.../preview.jpg"}           # imagem para mostrar na UI
  {"event": "log", "level": "info", "message": "..."}
  {"event": "done", "output": "C:/.../video.mp4", "report": "C:/.../report.json"}
  {"event": "error", "message": "...", "detail": "traceback"}
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Tipos de ambiente (chave interna -> rótulo da UI)
ROOM_TYPES: dict[str, str] = {
    "sala": "Sala de estar",
    "quarto": "Quarto",
    "cozinha": "Cozinha",
    "banheiro": "Banheiro",
    "escritorio": "Escritório",
    "sala_jantar": "Sala de jantar",
}

# Etapas e peso aproximado de cada uma no progresso total
STAGES: list[tuple[str, str, float]] = [
    ("preparo", "Preparando a foto", 0.02),
    ("analise", "Analisando profundidade, paredes e piso", 0.18),
    ("layout", "Escolhendo e posicionando os móveis", 0.05),
    ("assets", "Carregando modelos 3D", 0.05),
    ("render", "Renderizando a montagem (GPU)", 0.60),
    ("composicao", "Compondo sobre a foto original", 0.06),
    ("video", "Gerando o MP4", 0.04),
]


@dataclass
class JobRequest:
    photo_path: str
    room_type: str
    output_path: str
    quality: str = "auto"          # auto | previa | final
    work_dir: str | None = None    # pasta do trabalho (intermediários); padrão: DATA_DIR/jobs/<id>
    seed: int = 7

    def validate(self) -> None:
        if self.room_type not in ROOM_TYPES:
            raise ValueError(f"Tipo de ambiente inválido: {self.room_type}")
        if not Path(self.photo_path).is_file():
            raise FileNotFoundError(f"Foto não encontrada: {self.photo_path}")


@dataclass
class JobResult:
    output_path: str
    report_path: str
    work_dir: str
    timings: dict = field(default_factory=dict)


class ProgressEmitter:
    """Emite eventos JSON (para a UI) e/ou chama um callback Python."""

    def __init__(self, stream=None, callback=None):
        self.stream = stream
        self.callback = callback
        self._weights = {k: w for k, _, w in STAGES}
        self._order = [k for k, _, _ in STAGES]
        self._labels = {k: label for k, label, _ in STAGES}

    def _emit(self, ev: dict) -> None:
        if self.stream is not None:
            self.stream.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self.stream.flush()
        if self.callback is not None:
            self.callback(ev)

    def stage(self, stage: str, local_fraction: float = 0.0, message: str | None = None) -> None:
        done = sum(self._weights[s] for s in self._order[: self._order.index(stage)])
        total = sum(self._weights.values())
        frac = (done + self._weights[stage] * max(0.0, min(1.0, local_fraction))) / total
        self._emit({"event": "progress", "stage": stage, "fraction": round(frac, 4),
                    "message": message or self._labels[stage]})

    def preview(self, path: str | Path) -> None:
        self._emit({"event": "preview", "path": str(path)})

    def log(self, message: str, level: str = "info") -> None:
        self._emit({"event": "log", "level": level, "message": message})

    def done(self, result: JobResult) -> None:
        self._emit({"event": "done", "output": result.output_path, "report": result.report_path,
                    "work_dir": result.work_dir})

    def error(self, message: str, detail: str = "") -> None:
        self._emit({"event": "error", "message": message, "detail": detail})


def stdout_emitter() -> ProgressEmitter:
    return ProgressEmitter(stream=sys.stdout)


def request_to_json(req: JobRequest) -> str:
    return json.dumps(asdict(req), ensure_ascii=False)
