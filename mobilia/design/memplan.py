"""Plano de memória do redesenho: resolução da geração, passada de detalhe e onde fica cada modelo.

Medido no notebook do dono (RTX 5070 Laptop, 8 GB de VRAM, ~650 MB de RAM livre durante a geração): a passada de
detalhe a 1536 px com CFG (lote 2) levava o pico a 7,6–7,9 GB e o driver passava a usar a memória do sistema
(*sysmem fallback*): 18–23 s por passo contra ~1,5 s a 1024 px. Este módulo decide, ANTES de gerar, a partir da VRAM
total/livre e da RAM livre:

* lado maior da geração (1024 na GPU) e se o pipeline precisa de ``enable_model_cpu_offload``;
* se a passada de detalhe roda, em que tamanho e se com CFG (sem CFG o lote é 1: metade das ativações);
* se o OneFormer da conferência sai da GPU enquanto o SDXL gera;
* em que precisão os codificadores de texto rodam na CPU (fp32 pede ~3,4 GB de RAM; com pouca RAM, bf16).

Calibrado com o teste real no mesmo notebook (commit 40c56da): a estimativa antiga (6985 MB para a geração a 1024 px
contra um orçamento de 6757 MB) ligava o *offload* e desligava o refino numa placa de 8 GB, mas o pico MEDIDO por
chamada (``torch.cuda.max_memory_allocated``) foi 5813 MB, com codificadores de texto na CPU (``nvidia-smi`` total
7,0–7,4 GB contando a área de trabalho). Modelo: pesos fp16 (UNet 4,9 GiB + ControlNet small + VAE ≈ 5380 MB) +
ativações ~280 MB por megapixel por item do lote (×1,1 de folga) + contexto CUDA (~400 MB, fora do alocador do torch
e dentro do orçamento, que é medido como o ``nvidia-smi``). Geração a 1024×768 com CFG: ~5,86 GB no alocador
(~6,26 GB no total) — sem offload numa placa de 8 GB. Refino: o de 1536 px transbordava (7,6–7,9 GB) e fica só para
placas de 12 GB ou mais; numa placa de 8 GB só um refino LEVE (1280 ou 1152 px, sem CFG) e só se a estimativa couber
no orçamento com folga — senão desligado (registrado nas ``notes``). O ``engine`` ainda confere o tempo por passo
da passada de detalhe e desliga o refino das versões seguintes se ele indicar *sysmem fallback*
(``refine_spilled``).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

# ---- estimativas (MiB)
WEIGHTS_MB = 5380          # UNet fp16 (4900) + ControlNet small (~320) + VAE fp16 (~160), no alocador do torch
ACT_MB_PER_MP = 280        # ativações do UNet+ControlNet por megapixel por item do lote (SDPA, fp16), MEDIDO
CUDA_CTX_MB = 400          # contexto CUDA/cuBLAS: fora do alocador do torch, mas ocupa a placa
MEASURED_GEN_PEAK_MB = 5813  # pico medido por chamada a 1024×768 com CFG (RTX 5070 Laptop 8 GB; referência dos testes)
SEG_GPU_MB = 700           # OneFormer swin-large fp16 + buffers, se ficasse na GPU durante a geração
RESERVE_MB = 1024          # o que o Windows/driver/tela usam numa GPU de notebook (WDDM)
SAFETY = 1.10              # fragmentação do alocador (sobre as ativações)
BUDGET_CAP_8GB_MB = 7168   # "cabe em ~7 GB": teto explícito da classe de 8 GB
TE_FP32_RAM_MB = 3600      # codificadores de texto (CLIP-L + bigG) em fp32 na CPU
REFINE_SIDES = (1536, 1408, 1280, 1152)
REFINE_SIDES_TIGHT = (1280, 1152)   # placas < 12 GB: refino leve ou nenhum (o de 1536 transbordava no teste real)
REFINE_HEADROOM_MB = 400  # o refino deixa esta folga a mais (a estimativa de ativações não é medida aqui)
SPILL_FACTOR = 3.0         # passo do refino N× mais lento (por megapixel) que o da geração = memória do sistema


@dataclass
class MemPlan:
    device: str
    gen_long_side: int
    refine: bool                       # passada de detalhe ligada por padrão (o usuário ainda pode forçar)
    refine_long_side: int              # lado maior da passada de detalhe (0 = sem refino)
    refine_cfg: bool                   # False = guidance 1 no refino (lote 1, metade das ativações)
    offload: bool                      # ``enable_model_cpu_offload``
    seg_off_gpu: bool                  # OneFormer da conferência sai da GPU enquanto o SDXL gera
    te_dtype: str                      # "float32" | "bfloat16" (codificadores de texto na CPU)
    budget_mb: int
    peak_gen_mb: int
    peak_refine_mb: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def act_mb(width: int, height: int, batch: int) -> float:
    return width * height / 1e6 * ACT_MB_PER_MP * batch


def peak_mb(width: int, height: int, cfg: bool = True, seg_on_gpu: bool = False) -> int:
    """Pico estimado de VRAM do processo (como o ``nvidia-smi`` mostra) num passo do SDXL+ControlNet em
    ``width × height`` (CFG = lote 2): o do alocador (:func:`torch_peak_mb`) + o contexto CUDA."""
    return torch_peak_mb(width, height, cfg) + CUDA_CTX_MB + (SEG_GPU_MB if seg_on_gpu else 0)


def torch_peak_mb(width: int, height: int, cfg: bool = True) -> int:
    """Pico estimado do alocador do torch (``max_memory_allocated``, o que o motor registra por chamada)."""
    return int(WEIGHTS_MB + SAFETY * act_mb(width, height, 2 if cfg else 1))


def _dims(long_side: int, aspect: float = 4 / 3) -> tuple[int, int]:
    """Tamanho do pior caso (paisagem 4:3) para um lado maior."""
    return long_side, int(round(long_side / aspect))


def plan(device: str, vram_total_mb: int | None = None, vram_free_mb: int | None = None,
         ram_free_mb: int | None = None, aspect: float = 4 / 3) -> MemPlan:
    """Decide o plano a partir dos números de memória (todos em MiB; ``None`` = desconhecido).

    GPU: orçamento = min(total − reserva, livre agora − 256, 7 GB se a placa tem até 8 GB). A geração é a 1024 px
    se couber (senão offload). A passada de detalhe usa o maior lado de ``REFINE_SIDES`` que cabe no orçamento sem
    CFG (lote 1) — numa placa de 8 GB no máximo 1280 px (``REFINE_SIDES_TIGHT``); só placas folgadas (≥ 12 GB)
    refinam a 1536 com CFG. O OneFormer da
    conferência fica fora da GPU durante a geração sempre que o orçamento for apertado. Estimativas calibradas
    com o pico medido (ver o cabeçalho do módulo)."""
    notes: list[str] = []
    ram_low = ram_free_mb is not None and ram_free_mb < TE_FP32_RAM_MB + 512
    te = "bfloat16" if ram_low else "float32"
    if ram_low:
        notes.append(f"pouca RAM livre ({ram_free_mb} MB): codificadores de texto em bf16 na CPU")
    if not str(device).startswith("cuda"):
        w, h = _dims(768, aspect)
        return MemPlan("cpu", 768, False, int(768 * 1.5), True, False, False, te, 0, 0, 0,
                       notes + ["CPU: sem passada de detalhe por padrão"])
    total = int(vram_total_mb or 0)
    budget = total - RESERVE_MB
    if vram_free_mb is not None:
        budget = min(budget, int(vram_free_mb) - 256)
    if total and total <= 8 * 1024 + 256:
        budget = min(budget, BUDGET_CAP_8GB_MB)
    gw, gh = _dims(1024, aspect)
    gen_peak = peak_mb(gw, gh, cfg=True)
    offload = gen_peak > budget
    if offload:
        notes.append(f"geração a 1024 px pede ~{gen_peak} MB e o orçamento é {budget} MB: offload para a RAM")
        if ram_free_mb is not None and ram_free_mb < 4096:
            notes.append(f"atenção: offload com só {ram_free_mb} MB de RAM livre fica lento")
        return MemPlan("cuda", 1024, False, 0, False, True, True, te, budget, gen_peak, 0, notes)
    seg_off = (gen_peak + SEG_GPU_MB) > budget or total < 12 * 1024
    roomy = total >= 12 * 1024
    ref_side, ref_cfg, ref_peak = 0, False, 0
    for side in (REFINE_SIDES if roomy else REFINE_SIDES_TIGHT):
        for cfg in ((True, False) if roomy else (False,)):
            p = peak_mb(*_dims(side, aspect), cfg=cfg)
            if p <= budget - REFINE_HEADROOM_MB:
                ref_side, ref_cfg, ref_peak = side, cfg, p
                break
        if ref_side:
            break
    if not ref_side:
        notes.append(f"sem folga para a passada de detalhe (1280 px sem CFG pediria ~"
                     f"{peak_mb(*_dims(1280, aspect), cfg=False)} MB, orçamento {budget} MB menos "
                     f"{REFINE_HEADROOM_MB} de folga): só a geração a 1024 px")
    else:
        notes.append(f"passada de detalhe a {ref_side} px {'com' if ref_cfg else 'sem'} CFG (~{ref_peak} MB)")
    return MemPlan("cuda", 1024, bool(ref_side), ref_side, ref_cfg, False, seg_off, te, budget, gen_peak, ref_peak,
                   notes)


def ram_free_mb() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available // 2 ** 20)
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def current_plan(device: str) -> MemPlan:
    """Plano desta máquina agora (VRAM pelo torch, RAM pelo psutil). ``MOBILIA_VRAM_MB`` simula outra placa."""
    total = free = None
    sim = os.environ.get("MOBILIA_VRAM_MB", "").strip()
    if sim:
        total = free = int(sim)
    elif str(device).startswith("cuda"):
        try:
            import torch

            f, t = torch.cuda.mem_get_info()
            total, free = int(t // 2 ** 20), int(f // 2 ** 20)
        except Exception:  # noqa: BLE001
            pass
    return plan(device, total, free, ram_free_mb())


def refine_spilled(gen_s_per_step: float, gen_px: int, ref_s_per_step: float, ref_px: int,
                   factor: float = SPILL_FACTOR) -> bool:
    """O passo do refino, normalizado por pixel, ficou ``factor`` vezes mais lento que o da geração: a VRAM
    transbordou para a memória do sistema (o refino das próximas versões é desligado)."""
    if gen_s_per_step <= 0 or gen_px <= 0 or ref_px <= 0:
        return False
    return (ref_s_per_step / ref_px) > factor * (gen_s_per_step / gen_px)


__all__ = ["MemPlan", "plan", "current_plan", "peak_mb", "torch_peak_mb", "ram_free_mb", "refine_spilled",
           "REFINE_SIDES", "REFINE_SIDES_TIGHT"]
