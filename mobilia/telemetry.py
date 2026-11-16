"""Telemetria do trabalho: tempos por subetapa, VRAM, estatísticas de render por quadro.

Só biblioteca padrão: é importado pelo pipeline, pelo worker do Blender e pela interface.

As chaves de ``report.json`` seguem o PLANO_V2 (seção 4). As etapas de primeiro nível de
``tempos_s`` (``ETAPAS``) PARTICIONAM o tempo total: a soma delas fica dentro de poucos por
cento de ``tempos_s.total`` (testado). ``analise`` é um dicionário de subetapas e entra na
soma pelo seu total.
"""
from __future__ import annotations

import os
import re
import statistics
import threading
import time
from contextlib import contextmanager

# Etapas de primeiro nível de ``tempos_s``, na ordem em que acontecem.
ETAPAS: tuple[str, ...] = (
    "inicio_motor",      # lançamento do processo + imports + detecção de hardware
    "carga_foto",        # leitura da foto e gravação da placa
    "analise",           # dicionário (ver ETAPAS_ANALISE)
    "layout",
    "ativos",            # cena (scene.json) e cache de modelos, sem a parte de rede
    "download",          # só o tempo de rede (Poly Haven + pesos dos modelos); 0 com o cache completo
    "blender_inicio",    # subprocesso do Blender até o dispositivo configurado
    "blender_kernels",   # render de aquecimento minúsculo: carga/compilação dos kernels
    "cena_montagem",     # sala proxy, luzes, importação e animação dos móveis
    "calibracao",        # renders pequenos de calibração da luz
    "render_final",      # quadros com a cena completa em repouso (hold final)
    "render_marcos",     # quadros estáticos intermediários
    "render_dinamico",   # quadros com peças em movimento
    "blender_fim",       # gravação de frames.json, encerramento do processo e folgas
    "composicao",        # composição bit a bit que sobra depois do render (o resto corre junto dele)
    "codificacao",       # envio ao FFmpeg depois do render + finalização do MP4
    "verificacao",       # 3 quadros do MP4 decodificados por seek + PSNR
    "relatorio",
)
ETAPAS_ANALISE: tuple[str, ...] = (
    "redimensionar", "geometria_carga", "geometria_inferencia", "segmentacao_carga",
    "segmentacao_inferencia", "sala", "gravacao", "sol", "oclusores",
)
# Variáveis de ambiente com o instante (time.time()) em que a interface pediu o trabalho e em
# que lançou o processo do motor; sem elas o relógio começa na importação da CLI.
ENV_T_CLIQUE = "MOBILIA_T_CLIQUE"
ENV_T_INICIO = "MOBILIA_T_INICIO"


def env_time(name: str, env=None) -> float | None:
    env = os.environ if env is None else env
    try:
        v = float(env.get(name, ""))
    except ValueError:
        return None
    # instante futuro ou muito antigo (> 1 dia): ignora (relógio trocado, variável herdada)
    now = time.time()
    return v if now - 86400 < v <= now + 1 else None


class Stopwatch:
    """Acumula durações por chave pontuada (``"analise.sala"`` vira ``{"analise": {"sala": ...}}``)."""

    def __init__(self) -> None:
        self.times: dict = {}

    def add(self, key: str, seconds: float) -> None:
        d = self.times
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = d.get(parts[-1], 0.0) + max(0.0, float(seconds))

    def set(self, key: str, seconds: float) -> None:
        d = self.times
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = max(0.0, float(seconds))

    def get(self, key: str, default: float = 0.0):
        d = self.times
        for p in key.split("."):
            if not isinstance(d, dict) or p not in d:
                return default
            d = d[p]
        return d

    @contextmanager
    def measure(self, key: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(key, time.perf_counter() - t)

    def rounded(self, nd: int = 2) -> dict:
        def r(v):
            return {k: r(x) for k, x in v.items()} if isinstance(v, dict) else round(float(v), nd)
        return r(self.times)


def value_total(v) -> float:
    """Total de uma entrada de ``tempos_s`` (número ou dicionário de subetapas)."""
    if isinstance(v, dict):
        return float(sum(value_total(x) for x in v.values()))
    return float(v or 0.0)


def stages_sum(tempos: dict) -> float:
    return float(sum(value_total(tempos.get(k, 0.0)) for k in ETAPAS))


# ============================================================================ VRAM (NVML)


class VramMonitor:
    """Amostra a memória usada da GPU (NVML) numa thread e guarda o pico por fase.

    É a memória do DISPOSITIVO (inclui o Blender, outros programas e a reserva do Windows),
    que é o que limita os 8 GB. Sem NVML (sem GPU NVIDIA ou sem ``nvidia-ml-py``) fica
    inativo e os picos saem ``None``.
    """

    def __init__(self, interval_s: float = 0.25, index: int | None = None):
        self.interval = float(interval_s)
        self.peaks: dict[str, int] = {}
        self.phase: str | None = None
        self.total_mb: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None
        try:
            import pynvml  # pacote nvidia-ml-py

            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            if n:
                if index is None:   # a GPU de maior memória (mesma regra de hardware.best_gpu)
                    mems = [pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).total
                            for i in range(n)]
                    index = max(range(n), key=lambda i: mems[i])
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                self.total_mb = int(pynvml.nvmlDeviceGetMemoryInfo(self._handle).total // 2**20)
                self._nvml = pynvml
            else:
                pynvml.nvmlShutdown()
        except Exception:
            self._nvml = None

    @property
    def active(self) -> bool:
        return self._nvml is not None

    def used_mb(self) -> int | None:
        if not self.active:
            return None
        try:
            return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used // 2**20)
        except Exception:
            return None

    def _sample(self) -> None:
        ph = self.phase
        if ph is None:
            return
        u = self.used_mb()
        if u is not None and u > self.peaks.get(ph, -1):
            self.peaks[ph] = u

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self) -> "VramMonitor":
        if self.active and self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    @contextmanager
    def track(self, phase: str):
        old = self.phase
        self.phase = phase
        self._sample()
        try:
            yield
        finally:
            self._sample()
            self.phase = old

    def peak(self, phase: str) -> int | None:
        return self.peaks.get(phase) if self.active else None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None


def torch_peak_mb(device: str) -> int | None:
    """Pico de memória reservada pelo PyTorch desde o último ``torch_reset_peak`` (só CUDA)."""
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        return int(torch.cuda.max_memory_reserved() // 2**20)
    except Exception:
        return None


def torch_reset_peak(device: str) -> None:
    if str(device).startswith("cuda"):
        try:
            import torch

            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


# ============================================================================ estatísticas do Cycles

_RE_SAMPLE = re.compile(r"Sample (\d+)/(\d+)")
_RE_MEM = re.compile(r"(?:Mem|Peak):\s*([\d.]+)\s*([KMG])")
_RE_SAVING = re.compile(r"Saving:\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)")
_KERNEL_WORDS = ("kernels", "Compiling", "compiling")
_MB = {"K": 1.0 / 1024.0, "M": 1.0, "G": 1024.0}


def _clock_s(h: str | None, m: str, s: str) -> float:
    return (int(h) * 3600 if h else 0) + int(m) * 60 + float(s)


class RenderStatsCollector:
    """Interpreta as mensagens do handler ``bpy.app.handlers.render_stats`` de UMA chamada de render.

    O Cycles informa a fase atual em texto ("Updating Scene BVH", "Loading render kernels",
    "Sample 12/24", "Finished", "Time: 00:01.20 (Saving: 00:00.09)"). Daqui saem:

    - ``sincronizacao_s``: do início até a primeira amostra (envio da cena ao dispositivo,
      inclui a carga de kernels);
    - ``kernels_s``: tempo nas mensagens de carga/compilação de kernels (custo de 1ª execução);
    - ``amostragem_s`` e ``denoise_s``: o Cycles só informa "Sample N/N" depois do denoiser,
      então a amostragem é extrapolada do ritmo das amostras intermediárias e o denoise é o
      resto (``denoise_estimado``). Sem amostra intermediária, tudo conta como amostragem;
    - ``gravacao_s``: tempo de gravação do EXR informado pelo próprio Blender ("Saving");
    - ``mem_mb``: maior "Mem:"/"Peak:" informado (no Cycles em GPU, memória do dispositivo).
    """

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.t0 = clock()
        self.events: list[tuple[float, str]] = []

    def start(self) -> None:
        self.t0 = self.clock()
        self.events = []

    def feed(self, msg, t: float | None = None) -> None:
        if msg is None:
            return
        self.events.append((self.clock() if t is None else float(t), str(msg)))

    def finish(self, denoise: bool = True, t_end: float | None = None) -> dict:
        t_end = self.clock() if t_end is None else float(t_end)
        ev = self.events
        res = {"parede_s": t_end - self.t0, "sincronizacao_s": None, "kernels_s": 0.0, "amostragem_s": None,
               "denoise_s": None, "denoise_estimado": False, "gravacao_s": None, "amostras": None,
               "amostras_max": None, "mem_mb": None}
        mems = [float(m.group(1)) * _MB[m.group(2)] for _, s in ev for m in _RE_MEM.finditer(s)]
        if mems:
            res["mem_mb"] = round(max(mems), 1)
        for i, (t, s) in enumerate(ev):
            if any(w in s for w in _KERNEL_WORDS):
                t_next = ev[i + 1][0] if i + 1 < len(ev) else t_end
                res["kernels_s"] += max(0.0, t_next - t)
        samples = [(t, int(m.group(1)), int(m.group(2))) for t, s in ev if (m := _RE_SAMPLE.search(s))]
        t_fin = next((t for t, s in ev if "Finished" in s), None)
        for t, s in ev:
            if (m := _RE_SAVING.search(s)):
                res["gravacao_s"] = _clock_s(m.group(1), m.group(2), m.group(3))
        if samples:
            t_s0 = samples[0][0]
            kmax = max(k for _, k, _ in samples)
            res["amostras"] = kmax
            res["amostras_max"] = samples[-1][2]
            if t_fin is None:
                t_fin = samples[-1][0] if kmax else t_end
            res["sincronizacao_s"] = max(0.0, t_s0 - self.t0)
            busy = max(0.0, t_fin - t_s0)
            mids = [(t, k) for t, k, n in samples if 0 < k < kmax]
            if denoise and mids and kmax:
                tk, k = mids[-1]
                rate = max(0.0, tk - t_s0) / k
                samp = min(busy, rate * kmax)
                res["amostragem_s"] = samp
                res["denoise_s"] = busy - samp
                res["denoise_estimado"] = True
            else:
                res["amostragem_s"] = busy
                res["denoise_s"] = 0.0 if not denoise else None
        if res["gravacao_s"] is None and t_fin is not None:
            res["gravacao_s"] = max(0.0, t_end - t_fin)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}


# ============================================================================ resumo do render


def _median(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def summarize_render(por_quadro: list[dict], width: int, height: int) -> dict:
    """Resumo por camada e desempenho medido, a partir dos registros por quadro do worker.

    ``amostras_por_s`` = Σ(área × amostras) / Σ amostragem nos quadros "final" (cena inteira
    parada; é o quadro que mais pesa e o mais estável), ou em todos se não houver;
    ``custo_fixo_quadro_s`` = mediana de (parede − amostragem) — sincronização, denoiser,
    gravação e preparo do quadro, que não diminuem com menos amostras.
    """
    area_img = max(1, int(width) * int(height))

    def agg(rows: list[dict]) -> dict | None:
        if not rows:
            return None
        areas = [r.get("area_px") or 0 for r in rows]
        return {"quadros": len(rows), "area_media_mp": round(sum(areas) / len(rows) / 1e6, 4),
                "cobertura_media": round(sum(areas) / len(rows) / area_img, 4),
                "spp": max(int(r.get("spp") or 0) for r in rows),
                "s": round(sum(float(r.get("parede_s") or 0.0) for r in rows), 2)}

    fin = [r for r in por_quadro if r.get("camada") == "final"]
    mar = [r for r in por_quadro if r.get("camada") == "marco"]
    din = [r for r in por_quadro if r.get("camada") == "dinamico"]
    marcos = [{"quadro": r.get("quadro"), "itens": r.get("itens"), "area_mp": round((r.get("area_px") or 0) / 1e6, 4),
               "spp": r.get("spp"), "s": round(float(r.get("parede_s") or 0.0), 2)} for r in mar]
    base = fin or por_quadro
    work = sum((r.get("area_px") or 0) * (r.get("amostras") or r.get("spp") or 0) for r in base
               if r.get("amostragem_s"))
    samp = sum(float(r.get("amostragem_s") or 0.0) for r in base if r.get("amostragem_s"))
    fixed = [float(r["parede_s"]) - float(r.get("amostragem_s") or 0.0) for r in por_quadro
             if r.get("parede_s") is not None and r.get("amostragem_s") is not None]
    den = [float(r["denoise_s"]) for r in por_quadro if r.get("denoise_s") is not None]
    sav = [float(r["gravacao_s"]) for r in por_quadro if r.get("gravacao_s") is not None]
    mem = [float(r["mem_mb"]) for r in por_quadro if r.get("mem_mb") is not None]
    areas = [r.get("area_px") or 0 for r in por_quadro]
    return {
        "camadas": {"final": agg(fin), "marcos": marcos, "marcos_resumo": agg(mar), "dinamico": agg(din)},
        "desempenho": {
            "amostras_por_s": round(work / samp) if samp > 0 else None,
            "medido_em": "final" if fin else "todos",
            "custo_fixo_quadro_s": round(_median(fixed), 4) if fixed else None,
            "denoise_mediano_s": round(_median(den), 4) if den else None,
            "gravacao_mediana_s": round(_median(sav), 4) if sav else None,
            "mem_cycles_pico_mb": round(max(mem), 1) if mem else None,
        },
        "cobertura_recorte_media": round(sum(areas) / len(areas) / area_img, 4) if areas else None,
    }
