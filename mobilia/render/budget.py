"""Controle do orçamento de tempo do render (PT7, versão simples). Python puro.

O render em camadas (PT6) começa pelo quadro final (*hold*), sempre com as amostras do perfil.
Esse render é também a sonda de desempenho da máquina: dele saem a vazão de amostragem
(pixel·amostras por segundo), o custo fixo de cada chamada de render e o custo por pixel que não
depende das amostras (denoiser, gravação do EXR). Com o plano do restante (área de cada marco e de
cada render da camada dinâmica), ``choose_settings`` escolhe, UMA vez, antes do primeiro quadro
animado, as configurações que cabem no tempo que sobra, nesta ordem de corte (PLANO_V2, seção 2):

1. amostras da camada dinâmica (do perfil até ``DYN_SPP_MIN``);
2. escala de render só do CG em movimento (1,0 → ``SCALE_MIN``; reamostra só RGBA/sombra do
   recorte, nunca a placa — ver ``border.scaled_border`` e ``compositor.resample_crop``);
3. amostras dos marcos (do perfil até ``MARCO_SPP_MIN``).

O quadro final nunca perde amostras (já foi renderizado quando a escolha é feita) e a placa nunca
é tocada. Se nem tudo no mínimo cabe, o render segue no mínimo e ``cabe`` = False (a previsão é
mostrada assim mesmo). A previsão usa as amostras pedidas (a amostragem adaptativa só pode parar
antes): é um teto, não uma média.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field

BUDGET_S = 120.0              # do clique em "Gerar" até o MP4 verificado
POS_RENDER_S = 8.0            # cauda da composição + codificação + verificação + relatório
RESERVA_S = 5.0               # folga (estimativas de GPU ±)
DYN_SPP_MIN = 12
SCALE_MIN = 0.8
MARCO_SPP_MIN = 40
# os quadros dinâmicos têm desfoque de movimento (BVH com o passo de tempo) e a sonda (quadro
# final, em repouso) não: a amostra custa mais nas peças em movimento
MOTION_BLUR_FACTOR = 1.2
# "0" desliga o controle; um número troca o orçamento total (segundos)
ENV_ORCAMENTO = "MOBILIA_ORCAMENTO_S"

CUTS = ("spp_dinamico", "escala_cg", "spp_marcos")


def budget_spec(t_click: float | None, total_s: float | None = None) -> dict:
    """Bloco ``render.orcamento`` do ``scene.json`` (o worker decide com ele)."""
    total = BUDGET_S if total_s is None else float(total_s)
    env = os.environ.get(ENV_ORCAMENTO, "").strip()
    if env:
        try:
            total = float(env.replace(",", "."))
        except ValueError:
            pass
    return {"ativo": total > 0, "total_s": total, "t_clique": t_click, "pos_render_s": POS_RENDER_S,
            "reserva_s": RESERVA_S, "spp_dinamico_min": DYN_SPP_MIN, "escala_min": SCALE_MIN,
            "spp_marcos_min": MARCO_SPP_MIN, "fator_desfoque": MOTION_BLUR_FACTOR}


@dataclass(frozen=True)
class Overhead:
    """Custo de cada render que não depende das amostras: ``fixed_s`` por chamada (sincronização,
    preparo, kernels já quentes) + ``per_px_s`` por pixel do recorte (denoiser, gravação)."""

    fixed_s: float
    per_px_s: float = 0.0


@dataclass(frozen=True)
class Probe:
    """Medida da sonda (quadro final): vazão de amostragem e custo fixo."""

    throughput: float             # pixel·amostras por segundo de amostragem
    overhead: Overhead
    area_px: int = 0
    spp: int = 0
    estimado: bool = False        # sem tempo de amostragem separado: tudo dividido pela parede


def measure(row: dict) -> Probe | None:
    """Sonda a partir do registro do quadro final no ``por_quadro`` do worker (PT1).

    Vazão = área × amostras atingidas / tempo de amostragem. Denoiser e gravação viram custo por
    pixel (escalam com o recorte); o resto da parede (sincronização, preparo) é o custo fixo."""
    area = int(row.get("area_px") or 0)
    wall = float(row.get("parede_s") or row.get("render_s") or 0.0)
    spp = int(row.get("amostras") or row.get("spp") or 0)
    if area <= 0 or wall <= 0 or spp <= 0:
        return None
    samp = row.get("amostragem_s")
    if samp is None or float(samp) <= 0:
        # sem a divisão do tempo: 90% amostragem, 10% custo fixo (conservador nos dois lados)
        return Probe(area * spp / (0.9 * wall), Overhead(0.1 * wall), area, spp, estimado=True)
    samp = min(float(samp), wall)
    per_px_total = float(row.get("denoise_s") or 0.0) + float(row.get("gravacao_s") or 0.0)
    per_px_total = min(per_px_total, wall - samp)
    fixed = max(0.0, wall - samp - per_px_total)
    return Probe(area * spp / max(samp, 1e-6), Overhead(fixed, per_px_total / area), area, spp)


@dataclass(frozen=True)
class RenderPlan:
    """O que falta renderizar depois do quadro final (plano do PT6).

    ``marcos_px``: área (px) de cada marco ainda por renderizar; ``dinamicos_px``: área, em escala
    1, de cada render da camada dinâmica (incluindo as camadas antigas das transições)."""

    marcos_px: tuple = ()
    dinamicos_px: tuple = ()
    spp_marcos: int = 64
    spp_dinamico: int = 24
    motion_blur: bool = True


@dataclass
class Settings:
    spp_dinamico: int
    escala_cg: float              # 1.0 = sem reamostragem
    spp_marcos: int
    previsto_s: float             # render restante (s), teto
    orcamento_s: float            # tempo disponível para o render restante (s)
    cabe: bool
    cortes: list = field(default_factory=list)
    sem_corte_s: float = 0.0      # previsão com as configurações do perfil

    @property
    def render_pct(self) -> int:
        return int(round(self.escala_cg * 100))

    def layers(self, spl: dict) -> dict:
        """Amostras por camada com a escolha aplicada. O quadro final fica como está."""
        out = dict(spl)
        out["marco"] = int(min(int(spl.get("final", self.spp_marcos)), self.spp_marcos))
        out["dinamico"] = int(min(out["marco"], self.spp_dinamico))
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["render_pct"] = self.render_pct
        return d


def _overhead(o) -> Overhead:
    if isinstance(o, Overhead):
        return o
    if isinstance(o, (tuple, list)):
        return Overhead(float(o[0]), float(o[1]) if len(o) > 1 else 0.0)
    return Overhead(float(o or 0.0))


def forecast(throughput: float, overhead, plan: RenderPlan, spp_dinamico: int, escala: float,
             spp_marcos: int) -> float:
    """Tempo (s) do render restante com essas configurações (teto: todas as amostras pedidas)."""
    oh = _overhead(overhead)
    thr = max(float(throughput), 1e-9)
    mb = MOTION_BLUR_FACTOR if plan.motion_blur else 1.0
    s2 = float(escala) ** 2
    t = sum(oh.fixed_s + a * (oh.per_px_s + spp_marcos / thr) for a in plan.marcos_px)
    t += sum(oh.fixed_s + a * s2 * (oh.per_px_s + spp_dinamico * mb / thr) for a in plan.dinamicos_px)
    return float(t)


def choose_settings(throughput: float, overhead, plan: RenderPlan, budget: float) -> Settings:
    """Configurações do restante do render que cabem em ``budget`` segundos (ordem de corte do
    módulo). ``throughput`` em pixel·amostras/s, ``overhead`` em s por render (número ou
    ``Overhead``). Nunca sobe nada acima do perfil e nunca mexe no quadro final."""
    oh = _overhead(overhead)
    thr = max(float(throughput), 1e-9)
    budget = float(budget)
    d0 = max(1, int(plan.spp_dinamico))
    m0 = max(1, int(plan.spp_marcos))
    d_min, m_min = min(d0, DYN_SPP_MIN), min(m0, MARCO_SPP_MIN)
    mb = MOTION_BLUR_FACTOR if plan.motion_blur else 1.0
    # T(d, s, m) = FM + m·WM + FD + s²·(PD + d·WD)
    FM = sum(oh.fixed_s + oh.per_px_s * a for a in plan.marcos_px)
    WM = sum(plan.marcos_px) / thr
    FD = oh.fixed_s * len(plan.dinamicos_px)
    PD = oh.per_px_s * sum(plan.dinamicos_px)
    WD = sum(plan.dinamicos_px) * mb / thr

    def T(d, s, m):
        return FM + m * WM + FD + s * s * (PD + d * WD)

    full = T(d0, 1.0, m0)

    def done(d, s, m, cuts):
        t = T(d, s, m)
        return Settings(int(d), float(s), int(m), round(t, 3), round(budget, 3), bool(t <= budget + 1e-9),
                        cuts, round(full, 3))

    if full <= budget:
        return done(d0, 1.0, m0, [])
    if not plan.dinamicos_px:
        # nada dinâmico a cortar: direto aos marcos
        m = m0
        if WM > 0 and m_min < m0:
            m = max(m_min, min(m0, int(math.floor((budget - FM - FD) / WM + 1e-9))))
        return done(d0, 1.0, m, ["spp_marcos"] if m < m0 else [])
    # 1. amostras dinâmicas
    if WD > 0:
        d_star = (budget - FM - m0 * WM - FD - PD) / WD
        d = int(math.floor(d_star + 1e-9))
        if d >= d_min:
            return done(min(d, d0), 1.0, m0, ["spp_dinamico"])
    cuts = ["spp_dinamico"] if d_min < d0 else []
    # 2. escala do CG em movimento (porcentagem inteira, como o Blender)
    var = PD + d_min * WD
    if var > 0:
        s2 = (budget - FM - m0 * WM - FD) / var
        if s2 >= SCALE_MIN ** 2:
            pct = int(math.floor(100 * math.sqrt(min(s2, 1.0)) + 1e-9))
            pct = max(int(round(SCALE_MIN * 100)), min(99, pct))
            return done(d_min, pct / 100, m0, cuts + ["escala_cg"])
        cuts = cuts + ["escala_cg"]
        s = SCALE_MIN
    else:
        s = 1.0
    # 3. amostras dos marcos
    if WM > 0 and m_min < m0:
        m_star = (budget - FM - FD - s * s * var) / WM
        m = max(m_min, min(m0, int(math.floor(m_star + 1e-9))))
        cuts = cuts + ["spp_marcos"]
    else:
        m = m0
    return done(d_min, s, m, cuts)


def format_duration(seconds: float) -> str:
    """"≈ 1 min 40 s" (arredonda para 5 s acima de 1 min)."""
    s = max(0, int(round(float(seconds))))
    if s >= 60:
        s = int(5 * round(s / 5))
        m, r = divmod(s, 60)
        return f"≈ {m} min {r:02d} s" if r else f"≈ {m} min"
    return f"≈ {s} s"
