"""Amostras do Cycles por camada de quadro (Python puro: usado pelo pipeline e pelo worker).

Camadas (as mesmas da telemetria do worker):
- ``final``: a cena completa em repouso (o *hold* do fim, o quadro que o espectador mais olha);
- ``marco``: quadro parado com só parte dos itens (pausa entre entradas);
- ``dinamico``: alguma peça em movimento (o desfoque de movimento esconde o ruído residual).

O quadro final nunca perde amostras. Os dinâmicos usam bem menos; a amostragem adaptativa
tem mínimo explícito (8 nos dinâmicos, 32 nos estáticos) e limiar 0,02. Até existir o render em
camadas (PT6), cada quadro ainda é a cena inteira: para o salto de ruído entre o último quadro
dinâmico e o quadro parado seguinte não "pular", os ``RAMP_FRAMES`` quadros dinâmicos que vêm
logo antes de um quadro parado sobem as amostras em progressão geométrica até as dele.
"""
from __future__ import annotations

LAYERS = ("final", "marco", "dinamico")
ADAPTIVE_THRESHOLD = 0.02
ADAPTIVE_MIN = {"final": 32, "marco": 32, "dinamico": 8}
RAMP_FRAMES = 3


def samples_by_layer(final: int, preview: bool = False) -> dict:
    """Amostras por camada a partir das do quadro final (perfil de hardware/qualidade)."""
    final = max(1, int(final))
    if preview:
        return {"final": final, "marco": final, "dinamico": max(min(final, 12), final // 2)}
    return {"final": final, "marco": max(min(final, 32), final // 2),
            "dinamico": max(min(final, 16), final // 4)}


def clamp_layers(layers: dict | None, final: int) -> dict:
    """Normaliza o dicionário de camadas: nenhuma camada passa do quadro final."""
    final = max(1, int(final))
    base = samples_by_layer(final)
    out = {k: int(min(final, max(1, int((layers or {}).get(k, base[k]))))) for k in LAYERS}
    out["final"] = final
    return out


def adaptive_min(layer: str, samples: int) -> int:
    """``adaptive_min_samples`` explícito da camada (nunca acima das amostras do quadro)."""
    return int(min(int(samples), ADAPTIVE_MIN.get(layer, ADAPTIVE_MIN["dinamico"])))


def frame_samples(layers_seq: list[str], spl: dict, ramp: int = RAMP_FRAMES) -> list[int]:
    """Amostras de cada quadro RENDERIZADO, na ordem, a partir da camada de cada um.

    Os ``ramp`` quadros dinâmicos imediatamente antes de um quadro parado (final/marco) com
    mais amostras sobem em progressão geométrica: o j-ésimo antes dele recebe
    ``dyn · (alvo/dyn)^((ramp + 1 − j)/(ramp + 1))``."""
    out = [int(spl[layer]) for layer in layers_seq]
    if ramp <= 0:
        return out
    for i, layer in enumerate(layers_seq):
        if layer == "dinamico" or i == 0 or layers_seq[i - 1] != "dinamico":
            continue
        target = int(spl[layer])
        for j in range(1, ramp + 1):
            k = i - j
            if k < 0 or layers_seq[k] != "dinamico":
                break
            dyn = int(spl["dinamico"])
            if target <= dyn:
                break
            s = int(round(dyn * (target / dyn) ** ((ramp + 1 - j) / (ramp + 1))))
            out[k] = max(out[k], min(target, s))
    return out
