"""Contrato de execução de um trabalho (foto -> vídeo) e protocolo de progresso.

A interface gráfica NUNCA roda o pipeline no próprio processo: ela inicia o
processo de trabalho (``mobilia-worker run ...`` ou, no executável congelado,
``7xAi.exe --worker run ...``) e lê eventos JSON, um por linha, na saída
padrão. Isso isola falhas, libera a VRAM ao final e permite cancelar com segurança.

Eventos (um objeto JSON por linha, campo "event"):
  {"event": "progress", "stage": "analise", "fraction": 0.12, "message": "..."}
  {"event": "preview", "path": "C:/.../preview.jpg"}           # imagem para mostrar na UI
  {"event": "log", "level": "info", "message": "..."}
  {"event": "mode", "mode": "B", "reason": "...", "stages": [["preparo", "Preparando a foto"], ...]}
  {"event": "done", "output": "C:/.../video.mp4", "report": "C:/.../report.json", "mode": "B",
   "warning": "..." | null, "gate": {...} | null}
  # modo B com a sala vazia reprovada pelo portão de qualidade (sem ``aceitar_placa_fraca``): "output" vem null,
  # "warning" explica o motivo e a interface oferece gerar mesmo assim, usar o modo A ou trocar de foto.
  {"event": "error", "message": "...", "detail": "traceback"}
"""
from __future__ import annotations

import json
import sys
import threading
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

# Etapas do modo B (foto mobiliada, PLANO_V3 §3): sem layout/3D; o motor troca a lista com o evento "mode"
STAGES_B: list[tuple[str, str, float]] = [
    ("preparo", "Preparando a foto", 0.02),
    ("analise", "Analisando profundidade, paredes e piso", 0.20),
    ("pecas", "Separando os móveis da foto, peça por peça", 0.10),
    ("sala_vazia", "Montando a sala vazia (por trás, não aparece no vídeo)", 0.40),
    ("camadas", "Recortando os móveis reais e planejando a entrada", 0.08),
    ("video", "Compondo, verificando e gerando o MP4", 0.20),
]

MODES = ("auto", "A", "B")
PLATES = ("auto", "ia", "geometria")


@dataclass
class JobRequest:
    photo_path: str
    room_type: str
    output_path: str
    quality: str = "auto"          # auto | previa | final
    work_dir: str | None = None    # pasta do trabalho (intermediários); padrão: DATA_DIR/jobs/<id>
    seed: int = 7
    # "auto" = detectado pela foto (vazia -> A, mobiliada -> B); "A"/"B" = escolha manual (só a tela de aviso
    # do portão de qualidade oferece; a tela inicial pede apenas o tipo de ambiente)
    mode: str = "auto"
    # modo B: gera o vídeo mesmo com a sala vazia reprovada pelo portão (o usuário pediu, com aviso no relatório)
    accept_weak_plate: bool = False
    # modo B: como a sala vazia do início é feita. "auto" = gerada por IA quando o gerador de difusão está
    # instalado (GPU ou CPU), senão pela geometria; "ia" | "geometria" forçam
    plate: str = "auto"
    # modo B: o cômodo vazio já escolhido pelo usuário ("Esvaziar ambiente"), usado dentro da máscara de remoção
    empty_image: str | None = None
    # modo B: tecido (roupa de cama, almofadas, cortina) assenta com um movimento pequeno quando a peça pousa
    fabric: bool = True

    def validate(self) -> None:
        if self.room_type not in ROOM_TYPES:
            raise ValueError(f"Tipo de ambiente inválido: {self.room_type}")
        if self.mode not in MODES:
            raise ValueError(f"Modo inválido: {self.mode} (use auto, A ou B)")
        if self.plate not in PLATES:
            raise ValueError(f"Sala vazia inválida: {self.plate} (use auto, ia ou geometria)")
        if not Path(self.photo_path).is_file():
            raise FileNotFoundError(f"Foto não encontrada: {self.photo_path}")
        if self.empty_image is not None and not Path(self.empty_image).is_file():
            raise FileNotFoundError(f"Imagem do cômodo vazio não encontrada: {self.empty_image}")


@dataclass
class JobResult:
    output_path: str
    report_path: str
    work_dir: str
    timings: dict = field(default_factory=dict)
    mode: str = "A"
    warning: str | None = None       # aviso em português para a tela de resultado (portão de qualidade)
    gate: dict | None = None         # resumo do portão do modo B (aprovada, nota, estratégia)


class ProgressEmitter:
    """Emite eventos JSON (para a UI) e/ou chama um callback Python.

    Pode ser usado de várias threads (a composição em fluxo emite prévias enquanto o render
    emite o progresso): cada evento sai inteiro, numa linha."""

    def __init__(self, stream=None, callback=None):
        self.stream = stream
        self.callback = callback
        self._lock = threading.RLock()       # o callback pode emitir de novo
        self._set_stages(STAGES)

    def _set_stages(self, stages) -> None:
        self._weights = {k: w for k, _, w in stages}
        self._order = [k for k, _, _ in stages]
        self._labels = {k: label for k, label, _ in stages}

    def mode(self, mode: str, reason: str = "") -> None:
        """Anuncia o modo escolhido (A ou B) e troca a lista de etapas do progresso."""
        stages = STAGES_B if mode == "B" else STAGES
        self._set_stages(stages)
        self._emit({"event": "mode", "mode": mode, "reason": reason,
                    "stages": [[k, label] for k, label, _ in stages]})

    def _emit(self, ev: dict) -> None:
        with self._lock:
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

    def forecast(self, total_s: float, fits: bool = True) -> None:
        """Previsão do tempo total (s, desde o clique em "Gerar"), feita pelo controle de orçamento
        do render (PT7) depois do quadro final."""
        from .render.budget import format_duration

        self._emit({"event": "forecast", "total_s": round(float(total_s), 1), "cabe": bool(fits),
                    "message": f"Previsão: {format_duration(total_s)} no total"})

    def log(self, message: str, level: str = "info") -> None:
        self._emit({"event": "log", "level": level, "message": message})

    def done(self, result: JobResult) -> None:
        self._emit({"event": "done", "output": result.output_path or None, "report": result.report_path,
                    "work_dir": result.work_dir, "mode": result.mode, "warning": result.warning,
                    "gate": result.gate})

    def error(self, message: str, detail: str = "") -> None:
        self._emit({"event": "error", "message": message, "detail": detail})


def stdout_emitter() -> ProgressEmitter:
    return ProgressEmitter(stream=sys.stdout)


def request_to_json(req: JobRequest) -> str:
    return json.dumps(asdict(req), ensure_ascii=False)
