"""Sala vazia do modo B ("placa"), PLANO_V3 §3.2, §3.6 e PT-V3-3.

A placa é a foto sem os móveis. Ela é o 1º quadro do vídeo e, fora da máscara de remoção, é **a foto byte a byte**
(cópia de volta no fim, conferida com ``np.array_equal``). Etapas:

1. **Máscara de efeitos (2 passadas).** Núcleo = peças (máscara dura ∪ alpha > 0,02) dilatadas 0,4% da diagonal.
   Candidatos de sombra/reflexo/luz = anel de 1,2% da diagonal + faixa espelhada abaixo dos pés (até 45% da altura
   da peça; 60% e também na parede para os planejados: luz de LED sob o aéreo, brilho no backsplash). Uma 1ª
   passada barata (inpainting no espaço da imagem) dá uma placa provisória; só ficam na máscara os candidatos em
   que a foto difere dela **e** que estão ligados à peça. Os efeitos são separados em "sombra" (foto mais escura)
   e "luz" (foto mais clara; reflexo no piso polido e luz emitida por planejados): os dois saem da placa.
2. **Ray-cast na caixa vazia** (``RoomModel``): cada pixel mascarado vira piso, teto, parede *i*, "outro" (visto
   por uma porta, pela profundidade do MoGe) ou **protegido** (janela, porta, cortina: dentro de uma abertura do
   modelo da sala ou colado a pixels dessas classes).
3. **Preenchimento por plano** em vistas retificadas (piso e teto de cima, paredes de frente), **sempre com a
   máscara inteira como buraco** (senão o LaMa reconstrói a peça a partir dos pedaços dela que caem no plano
   vizinho). Contexto do piso/teto: só o próprio material. Em cada vista:
   - revestimento com padrão (azulejo, porcelanato): a grade é detectada por autocorrelação e as placas visíveis
     são repetidas (síntese periódica, rejunte alinhado);
   - senão LaMa + reinjeção do grão (o LaMa borra buracos grandes);
   - **harmonização**: membrana tipo Poisson (a diferença de tom na borda é interpolada para dentro do buraco) e
     corte de "ilhas" de tom fora da faixa do anel original.
4. **Parede quase toda escondida** (cozinha planejada de ponta a ponta): pintura lisa com o perfil de cor por
   altura das paredes visíveis (inclui o rodapé) e o gradiente de luz medido na parte visível, mais grão leve.
   Nas paredes preenchidas pelo LaMa, o **rodapé** detectado no perfil é redesenhado reto na altura certa.
5. **Protegidos e "outro"**: inpainting no espaço da imagem, com contexto local (recorte em volta). Nunca
   recebem textura de parede retificada; nada de porta, tomada ou vista nova.
6. **Cópia de volta** byte a byte fora da máscara.

A **nota de qualidade** (:class:`PlateQuality`) mede, por plano: ΔE de costura (faixa interna x anel) e ΔE do
preenchimento todo, razão de grão/textura, distância e continuidade da linha parede/piso projetada, e (opcional)
o OneFormer rodado de novo na placa (nenhum móvel com área > 0,5% pode reaparecer dentro da máscara, nem porta ou
janela nova), mais névoa, fantasma e placas soltas dentro do preenchimento (:mod:`plate_score`). A nota é contínua
(100 menos os pontos perdidos por defeito; reprovada = 0,6 × nota sem teto). Ela alimenta o portão de qualidade (:func:`build_plate_gated`): placa ruim refaz mantendo os planejados e, se ainda
ruim, avisa; nunca entrega uma placa ruim em silêncio.

O inpainter é injetável (``inpaint(rgb8, mask) -> rgb8``): no produto é o big-lama (:class:`Lama`, CUDA se houver,
senão CPU); nos testes rápidos, :func:`telea_inpaint` (OpenCV, sem modelo).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np
from scipy import ndimage

from ..analysis.segmentation import GROUP_IDS

InpaintFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

# rótulos do ray-cast (paredes usam o id >= 0)
FLOOR, OTHER, CEIL, PROTECTED = -1, -2, -3, -4
LABEL_NAMES = {FLOOR: "piso", OTHER: "outro", CEIL: "teto", PROTECTED: "protegido"}

G_FLOOR, G_WALL, G_CEIL = GROUP_IDS["floor"], GROUP_IDS["wall"], GROUP_IDS["ceiling"]
G_PROTECTED = (GROUP_IDS["window"], GROUP_IDS["window_covering"], GROUP_IDS["door"])

# --- máscara de efeitos ----------------------------------------------------------------------------------------
CORE_DILATE = 0.004          # fração da diagonal
EFFECT_RING = 0.012
EFFECT_RING_BUILTIN = 0.025
BAND_DOWN = 0.45             # faixa espelhada abaixo dos pés (fração da altura da peça)
BAND_DOWN_BUILTIN = 0.6
# peças penduradas na parede: a sombra da moldura cai NA PAREDE (embaixo e de lado). Antes a faixa só valia sobre o
# piso, e o retângulo claro/escuro do quadro ficava fora da máscara (fantasma do quadro, avaliação SDXL rodada 1)
WALL_HUNG = ("painting", "poster", "picture", "mirror", "clock", "photo", "frame", "sconce", "tv", "television")
BAND_DOWN_WALL_HUNG = 0.25
EFFECT_REL = 0.07            # diferença relativa (borrada) foto x placa provisória
EFFECT_ABS = 12.0            # ou diferença absoluta em níveis

# --- preenchimento ---------------------------------------------------------------------------------------------
PLAIN_WALL_VISIBLE = 0.25    # parede com menos que isto visível -> pintura lisa
PROTECT_REACH = 0.012        # pixel mascarado a menos disto (diag) de janela/porta/cortina -> protegido
PERIOD_MIN_ACF = 0.40
MAX_SIDE = 1024              # lado máximo de uma vista retificada
MIN_CONTEXT_PX = 1500        # material visível mínimo do plano na vista retificada para preencher por plano
MIN_CONTEXT_FRAC = 0.01      # ... e em fração do buraco

# --- limites da nota (portão) ----------------------------------------------------------------------------------
# estratégias do portão (:func:`build_plate_gated`): "planos" = LaMa por plano retificado (padrão);
# "imagem" = LaMa direto no espaço da imagem (melhor quando o modelo da sala erra: linhas tortas por plano errado)
# "geometria" = sala refeita pela geometria (:mod:`geofill`: só material real da foto, plano a plano; padrão do portão)
STRATEGIES = ("planos", "imagem", "geometria", "ia")
GATE_WIDE_DILATE = 2.5       # máscara ampla do portão: núcleo dilatado 2,5× (sobra de borda/sombra do móvel)
# Planejados mantidos **por padrão** (a 1ª tentativa já os deixa na foto, sem tentar a sala crua): quando a remoção
# completa passa de 35% da foto ou só os planejados cobrem 10% ou mais. Atrás de armários, estantes e bancadas grandes
# não há parede visível para copiar: o LaMa inventa "pedra derretida" e a nota pode até aprovar (cor média certa).
KEEP_BUILTINS_TOTAL = 0.35
KEEP_BUILTINS_AREA = 0.10
KEEP_BUILTINS_MIN = 0.01     # planejados abaixo disto não mudam a decisão
# Decisão do produto (§3.6, revista na rodada 1 de correções): a sala crua (planejados fora) é SEMPRE a 1ª tentativa;
# os planejados só ficam se o portão (agora com falhas duras: estrutura inventada, material, contorno) reprovar a
# crua. True volta ao comportamento antigo (mantém por padrão quando ``keep_builtins_reason`` acusa risco).
KEEP_BUILTINS_FIRST = False
# Sala vazia pela geometria (:mod:`geofill`) é a 1ª estratégia do portão quando há modelo da sala: nada imaginado, só o
# material real copiado plano a plano. As tentativas por LaMa ("crua_por_plano", "lama_imagem") só entram se a sala
# não tem modelo (sem planos não há o que copiar) ou com ``GATE_LAMA_ALTERNATIVES`` ligado.
GEOMETRY_FIRST = True
GATE_LAMA_ALTERNATIVES = False
# peça vista por uma abertura: profundidade > BEYOND_RATIO × a da parede da caixa em >= BEYOND_FRAC da peça (a peça
# encostada na parede fica em ~0,9-1,0; o ruído do MoGe é de poucos %)
BEYOND_RATIO = 1.07
BEYOND_FRAC = 0.6
RANK_HARD_PTS = 3.0           # desempate do portão: falha dura (0..1) e cada motivo valem poucos pontos de nota
RANK_REASON_PTS = 1.0
RESTING_MAX = 0.002           # objeto não detectado apoiado na peça que sai: até 0,2% da foto sai junto
HALO_MIN_Z = 0.30             # ... só acima da faixa do rodapé (m sobre o piso)
HALO_CAP = 0.015              # sombra/borda escura de peça na parede: a máscara cresce até 1,5% da diagonal
MASK_HOLE_MAX = 0.005          # buraco fechado da máscara até 0,5% da foto entra na máscara (estratégia geometria)

LIMITS = {
    "costura_de_max": 6.0,           # ΔE76 entre faixa interna e anel, por plano (aceite PT-V3-3: <= 3)
    "grao_log2_max": 1.0,            # |log2(grão dentro / grão no anel)|
    "linha_px_max": 4.0,             # distância mediana da borda à linha parede/piso projetada (aceite: <= 3)
    "linha_continuidade_min": 0.5,   # fração dos pontos da linha (dentro da máscara) com borda achada
    "preenchimento_de_max": 6.0,     # ΔE76 preenchimento x anel (média global ou referência local), por plano
    "preenchimento_de_max_relaxado": 9.0,   # ... quando a costura é ótima (<= 2) e o grão bom: gradiente de luz
    "costura_relaxa_de": 4.5,        # costura a partir da qual o limite do preenchimento começa a subir
    "grao_relaxa_log2": 0.6,         # |log2 grão| acima disto o limite do preenchimento não sobe
    "relaxa_area_max": 0.6,          # ... nem quando o preenchimento passa de 60% do material visível do plano
                                     # (tapete grande removido: o piso de baixo é inventado, não iluminado)
    "costura_sem_plano_max": 16.0,   # ΔE76 da costura do que não tem plano (vãos, protegidos, sobre bancadas)
    "area_gerada_max": 0.70,         # fração da foto reconstruída
    "sem_referencia_max": 0.02,      # parede sem nada à vista (geometria) acima disto da foto: veto
    # geometria da sala não achada (HDR, retrato, piso todo coberto): quase todo o preenchimento fica "sem plano",
    # que só tem o limite frouxo de costura. Área grande e maioria sem plano = não medido -> reprova
    "sem_plano_area_min": 0.30,      # área gerada (fração da foto) a partir da qual a regra vale
    "sem_plano_frac_max": 0.50,      # fração da área gerada que fica "sem plano" (vãos, protegidos, plano perdido)
    "redetectado_area_min": 0.005,   # móvel reencontrado pelo OneFormer na placa (> 0,5% da foto)
    # medidas de realismo (``plate_score``, PT-V3-4): severidades 0..1 (placas soltas: fração da máscara)
    "nevoa_max": 0.6,                # névoa/borrão dentro do preenchimento
    "fantasma_max": 0.6,             # estrutura da peça removida que sobrou
    "placas_soltas_max": 0.35,       # manchas cuja cor destoa da cor esperada do plano
    # falhas "duras" (rodada 1 de correções do SDXL): não se compensam com costura/cor boas
    "estrutura_nova_max": 0.6,       # bordas fortes inventadas (rejunte, tijolo, cano, placa solta) x anel
    "material_max": 0.6,             # orientação das bordas do preenchimento x material visível do mesmo plano
    "contorno_max": 0.6,             # contorno da peça removida marcado na borda da máscara
    "nota_min": 60.0,                # nota contínua mínima para aprovar
    "exige_referencia": True,        # plano preenchido sem material visível reprova (False na sala por IA)
}

# Pesos da nota contínua (pontos tirados de 100 pela severidade 0..1 de cada defeito). Os defeitos que o crítico vê
# primeiro (névoa, fantasma, placas soltas, objeto novo) pesam mais que costura e grão.
SCORE_WEIGHTS = {
    "costura": 10.0, "cor": 10.0, "grao": 8.0, "linha": 8.0, "sem_plano": 10.0, "mancha_sem_plano": 8.0,
    "plano_sem_referencia": 10.0, "area": 8.0, "nevoa": 25.0, "fantasma": 25.0, "placas_soltas": 20.0,
    "movel_reencontrado": 30.0, "abertura_nova": 30.0,
    "estrutura_nova": 25.0, "material": 20.0, "contorno": 20.0,
}
# falhas duras: a pior delas (severidade 0..1) vai para ``PlateQuality.hard`` e decide a escolha de semente/tentativa
# antes da nota média (uma placa lisa com rejunte inventado não ganha de uma placa com costura pior)
HARD_KEYS = ("nevoa", "fantasma", "estrutura_nova", "material", "contorno")

# Portão da sala vazia "ia" (o gerador de difusão refina a sala refeita pela geometria, ``design/empty.py``). As
# medidas de textura do portão foram calibradas no preenchimento pela geometria (cópia do material visível): o
# grão e as bordas de uma imagem gerada são outros, então ``material``/``estrutura_nova``/``grao`` medem diferente
# sem que haja defeito. O que não muda: móvel reencontrado, porta/janela nova e costura reprovam como antes.
LIMITS_IA = {**LIMITS,
             # Portão próprio da sala gerada. Medido na CPU (sala com sofás, foto do usuário, quarto): costura,
             # cor média, grão, "placas soltas" e "material" comparam o preenchimento com o material visível
             # COPIADO (é assim que a geometria preenche) e reprovaram todas as salas geradas, inclusive as sem
             # defeito visível. Aqui decide o que a IA pode errar de verdade: móvel que volta, porta/janela nova,
             # fantasma e contorno do móvel, linha parede/piso fora do lugar, estrutura inventada e geometria da
             # sala não medida; costura só reprova acima de 9 (ΔE) e a área vai até 85% da foto.
             "costura_de_max": 9.0, "preenchimento_de_max": 99.0, "preenchimento_de_max_relaxado": 99.0,
             "grao_log2_max": 9.0, "placas_soltas_max": 1.01, "material_max": 1.01, "estrutura_nova_max": 0.85,
             # "névoa" lê parede lisa pintada como véu (o quarto gerado limpo deu 1,0): na "ia" só tira pontos
             "nevoa_max": 1.01, "area_gerada_max": 0.85, "costura_sem_plano_max": 24.0,
             "nota_min": 0.0,
             # parede escondida sem nada à vista (a cozinha que sai inteira): a geometria não tem o que copiar e
             # reprova; gerar essa parede é justamente o papel da IA (só tira pontos)
             "exige_referencia": False}


def disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _dil(m: np.ndarray, r: int) -> np.ndarray:
    return cv2.dilate(m.astype(np.uint8), disk(r)) > 0


def _ero(m: np.ndarray, r: int) -> np.ndarray:
    return cv2.erode(m.astype(np.uint8), disk(r)) > 0


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def _nblur(img: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Média gaussiana normalizada só sobre os pixels com peso ``w`` (convolução normalizada)."""
    wf = w.astype(np.float32)
    num = cv2.GaussianBlur(img.astype(np.float32) * (wf[..., None] if img.ndim == 3 else wf), (0, 0), sigma)
    den = cv2.GaussianBlur(wf, (0, 0), sigma)
    den = np.maximum(den, 1e-6)
    return num / (den[..., None] if img.ndim == 3 else den)


# ============================================================================================================
# Inpainters
# ============================================================================================================
class Lama:
    """big-lama TorchScript (Apache-2.0). CUDA se disponível, senão CPU (mesmo resultado, mais lento)."""

    def __init__(self, path, device: str = "auto"):
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = torch.jit.load(str(path), map_location=device).eval()
        self.calls: list[tuple[tuple[int, int], float]] = []

    def __call__(self, rgb8: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch

        t0 = time.perf_counter()
        h, w = mask.shape
        if not mask.any():
            return rgb8.copy()
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        img = np.pad(rgb8, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        m = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
        ti = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div(255.0).to(self.device)
        tm = torch.from_numpy(np.ascontiguousarray(m))[None, None].float().to(self.device)
        with torch.inference_mode():
            out = self.model(ti, tm)
        out = (out[0].permute(1, 2, 0).float().cpu().numpy()[:h, :w] * 255.0).round().clip(0, 255).astype(np.uint8)
        res = rgb8.copy()
        res[mask] = out[mask]
        self.calls.append(((w, h), time.perf_counter() - t0))
        return res

    def release(self) -> None:
        import gc

        del self.model
        gc.collect()
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


def load_lama(device: str = "auto", offline: bool = False) -> Lama:
    """Carrega o big-lama do diretório de pesos (SHA-256 conferido); ``WeightsError`` se ausente/corrompido."""
    from ..analysis.weights import big_lama_path

    return Lama(big_lama_path(offline=offline), device)


class LazyLama:
    """big-lama carregado só no 1º uso (a estratégia "geometria" não chama o inpainter; a LaMa fica como último
    recurso das outras estratégias)."""

    def __init__(self, device: str = "auto", offline: bool = False):
        self.device, self.offline = device, offline
        self._m: Lama | None = None

    @property
    def model(self):
        return None if self._m is None else self._m.model

    @property
    def calls(self) -> list:
        return [] if self._m is None else self._m.calls

    @property
    def loaded(self) -> bool:
        return self._m is not None

    def __call__(self, rgb8: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._m is None:
            self._m = load_lama(self.device, self.offline)
        return self._m(rgb8, mask)

    def release(self) -> None:
        if self._m is not None:
            self._m.release()
            self._m = None


def telea_inpaint(rgb8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpainting clássico do OpenCV (Telea). Sem modelo: usado nos testes rápidos e como último recurso."""
    if not mask.any():
        return rgb8.copy()
    res = cv2.inpaint(np.ascontiguousarray(rgb8), mask.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    out = rgb8.copy()
    out[mask] = res[mask]
    return out


# ============================================================================================================
# Caixa vazia: ray-cast e vistas retificadas
# ============================================================================================================
@dataclass
class Plane:
    key: int                    # FLOOR, CEIL ou id da parede
    A: np.ndarray               # (4, 3): (a, b, 1) -> ponto homogêneo da sala
    point: np.ndarray           # ponto do plano (3,)
    normal: np.ndarray          # normal (3,)
    wall: object | None = None


class RoomBox:
    """Projeções da caixa da sala (piso, teto, paredes do ``RoomModel``) na foto de tamanho (W, H)."""

    def __init__(self, room, W: int, H: int):
        self.room = room
        intr = room.intr()
        if (intr.width, intr.height) != (W, H):
            intr = intr.scaled(W, H)
        self.intr = intr
        self.K = intr.matrix().copy()
        self.K[0, 2] -= 0.5        # convenção de borda -> centro do pixel (OpenCV)
        self.K[1, 2] -= 0.5
        self.M = room.M
        self.Minv = np.linalg.inv(self.M)
        self.W, self.H = W, H
        self.rays = intr.rays() @ self.M[:3, :3].T          # direções no referencial da sala (H, W, 3)
        self.o = self.M[:3, 3].copy()
        ch = float(room.ceiling_height)
        self.planes: dict[int, Plane] = {
            FLOOR: Plane(FLOOR, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1.0]]), np.zeros(3),
                         np.array([0, 0, 1.0])),
            CEIL: Plane(CEIL, np.array([[1, 0, 0], [0, 1, 0], [0, 0, ch], [0, 0, 1.0]]), np.array([0, 0, ch]),
                        np.array([0, 0, 1.0])),
        }
        for w in room.walls:
            d, p0 = w.direction, np.asarray(w.p0, float)
            self.planes[w.id] = Plane(w.id, np.array([[d[0], 0, p0[0]], [d[1], 0, p0[1]], [0, 1, 0], [0, 0, 1.0]]),
                                      np.array([p0[0], p0[1], 0.0]),
                                      np.array([w.normal[0], w.normal[1], 0.0]), w)
        self.label, self.t, self.S, self.Z = self._raycast()

    def _raycast(self, margin: float = 0.35):
        d = self.rays
        H, W = d.shape[:2]
        best = np.full((H, W), np.inf)
        label = np.full((H, W), OTHER, np.int32)
        S = np.full((H, W), np.nan, np.float32)
        Z = np.full((H, W), np.nan, np.float32)
        ch = float(self.room.ceiling_height)
        with np.errstate(divide="ignore", invalid="ignore"):
            tf = -self.o[2] / d[..., 2]
            ok = (d[..., 2] < 0) & (tf > 0)
            best[ok], label[ok] = tf[ok], FLOOR
            tc = (ch - self.o[2]) / d[..., 2]
            ok = (d[..., 2] > 0) & (tc > 0)
            best[ok], label[ok] = tc[ok], CEIL
            for w in self.room.walls:
                n = np.asarray(w.normal, float)
                p0 = np.asarray(w.p0, float)
                tw = ((p0 - self.o[:2]) @ n) / (d[..., 0] * n[0] + d[..., 1] * n[1])
                hit = self.o + tw[..., None] * d
                s = (hit[..., :2] - p0) @ w.direction
                okw = ((tw > 0) & (s > -margin) & (s < w.length + margin) & (hit[..., 2] > -0.05)
                       & (hit[..., 2] < ch + 0.05) & (tw < best))
                best[okw], label[okw] = tw[okw], w.id
                S[okw], Z[okw] = s[okw], hit[..., 2][okw]
        return label, best, S, Z

    def coords(self, plane: Plane, us: np.ndarray, vs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coordenadas (a, b) no plano e distância ao longo do raio para os pixels (us, vs)."""
        r = self.rays[vs, us]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((plane.point - self.o) @ plane.normal) / (r @ plane.normal)
        hit = self.o + t[:, None] * r
        if plane.wall is None:
            a, b = hit[:, 0], hit[:, 1]
        else:
            a = (hit[:, :2] - np.asarray(plane.wall.p0)) @ plane.wall.direction
            b = hit[:, 2]
        return a, b, t * np.linalg.norm(r, axis=1)

    def homography(self, plane: Plane, a0: float, b1: float, ppm: float) -> np.ndarray:
        """Pixel (u, v) da vista retificada -> pixel da foto. a = u/ppm + a0 ; b = b1 - v/ppm."""
        G = np.array([[1 / ppm, 0, a0], [0, -1 / ppm, b1], [0, 0, 1.0]])
        return self.K @ self.Minv[:3, :] @ plane.A @ G

    def rect_for(self, plane: Plane, sel: np.ndarray, max_side: int = MAX_SIDE, pad_px: int = 40):
        """Homografia e tamanho da vista retificada que cobre ``sel`` (dilatado) com densidade ~ a da foto."""
        p = self.rect_params(plane, sel, max_side, pad_px)
        if p is None:
            return None
        a0, b1, ppm, size = p
        return self.homography(plane, a0, b1, ppm), size

    def rect_params(self, plane: Plane, sel: np.ndarray, max_side: int = MAX_SIDE, pad_px: int = 40,
                    pct: float = 0.5, near_pct: float = 50.0):
        """(a0, b1, ppm, tamanho) da vista retificada de :meth:`rect_for` (u = (a - a0)·ppm, v = (b1 - b)·ppm)."""
        vs, us = np.nonzero(_dil(sel, pad_px))
        a, b, dist = self.coords(plane, us, vs)
        ok = np.isfinite(a) & np.isfinite(b) & (dist > 0) & (dist < 15.0)
        if ok.sum() < 20:
            return None
        a, b, dist = a[ok], b[ok], dist[ok]
        a0, a1 = np.percentile(a, [pct, 100 - pct])
        b0, b1 = np.percentile(b, [pct, 100 - pct])
        if plane.wall is not None:
            a0, a1 = max(a0, -0.5), min(a1, plane.wall.length + 0.5)
            b0, b1 = max(b0, -0.3), min(b1, float(self.room.ceiling_height) + 0.2)
        ext = max(a1 - a0, b1 - b0, 1e-3)
        sv, su = np.nonzero(sel)
        _, _, dsel = self.coords(plane, su, sv)
        dsel = dsel[np.isfinite(dsel) & (dsel > 0)]
        med = float(np.percentile(dsel, near_pct)) if dsel.size else float(np.median(dist))
        ppm = min(self.K[0, 0] / max(med, 0.3), max_side / ext)
        size = (max(16, int((a1 - a0) * ppm) + 1), max(16, int((b1 - b0) * ppm) + 1))
        return float(a0), float(b1), float(ppm), size

    def floor_line(self, wall, step_px: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
        """Pontos (N, 2) da linha parede/piso projetada (centro de pixel) e a normal da linha na imagem (N, 2)."""
        L = wall.length
        n = max(8, int(L * self.K[0, 0] / step_px))
        s = np.linspace(0, L, n)
        p = np.asarray(wall.p0)[None] + s[:, None] * wall.direction[None]
        pts = np.c_[p, np.zeros(n), np.ones(n)]
        cam = pts @ self.Minv.T
        z = cam[:, 2]
        uv = (cam[:, :3] @ self.K.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uv[:, :2] / uv[:, 2:3]
        tang = np.gradient(uv, axis=0)
        nrm = np.c_[-tang[:, 1], tang[:, 0]]
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        ok = (z > 0.1) & np.isfinite(uv).all(1)
        return uv[ok], nrm[ok]


def _warp_in(img, Hm, size, interp=cv2.INTER_LINEAR, border=cv2.BORDER_REFLECT):
    return cv2.warpPerspective(img, Hm, size, flags=interp | cv2.WARP_INVERSE_MAP, borderMode=border)


def _warp_mask_in(m, Hm, size, thr=0):
    return _warp_in(m.astype(np.uint8) * 255, Hm, size, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT) > thr


# ============================================================================================================
# Textura: grade periódica, grão, harmonização
# ============================================================================================================
def detect_period(gray: np.ndarray, known: np.ndarray, min_p: int = 6) -> tuple[int, int] | None:
    """Período (py, px) de um revestimento com padrão, por autocorrelação normalizada só nos pixels conhecidos.
    ``None`` se não houver grade nos dois eixos (piso liso, madeira corrida, parede pintada)."""
    if known.sum() < 400:
        return None
    g = gray.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 8.0)          # tira o gradiente de luz: fica o rejunte
    m = known.astype(np.float32)
    g = (g - g[known].mean()) * m
    var = float((g[known] ** 2).mean())
    if var < 1.0:
        return None
    h, w = g.shape
    s = (2 * h, 2 * w)
    F = np.fft.rfft2(g, s)
    Mf = np.fft.rfft2(m, s)
    num = np.fft.irfft2(np.abs(F) ** 2, s)
    den = np.fft.irfft2(np.abs(Mf) ** 2, s)
    acf = num / np.maximum(den, 1.0) / var
    overlap = den / max(den[0, 0], 1.0)

    def axis_period(vals, ov):
        n = len(vals)
        best = None
        for p in range(min_p, n):
            if ov[p] < 0.25:
                break
            if vals[p] >= PERIOD_MIN_ACF and vals[p] >= vals[p - 1] and (p + 1 >= n or vals[p] >= vals[p + 1]):
                if vals[1:p].min() < vals[p] - 0.25:
                    best = p
                    break
        return best

    px = axis_period(acf[0, : w // 2], overlap[0, : w // 2])
    py = axis_period(acf[: h // 2, 0], overlap[: h // 2, 0])
    if px is None or py is None:
        return None
    return int(py), int(px)


def periodic_fill(img: np.ndarray, known: np.ndarray, hole: np.ndarray, period: tuple[int, int],
                  reach: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Copia, para cada pixel do buraco, o pixel conhecido mais próximo deslocado de um múltiplo da grade
    (rejunte alinhado). Retorna (imagem, pixels que ficaram sem fonte)."""
    py, px = period
    out = img.copy()
    todo = hole.copy()
    src_ok = _ero(known, 2)
    h, w = hole.shape
    offs = [(i * py, j * px) for i in range(-reach, reach + 1) for j in range(-reach, reach + 1) if i or j]
    offs.sort(key=lambda o: o[0] ** 2 + o[1] ** 2)
    for dy, dx in offs:
        ys, xs = np.nonzero(todo)
        if ys.size == 0:
            break
        sy, sx = ys + dy, xs + dx
        ok = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
        ys, xs, sy, sx = ys[ok], xs[ok], sy[ok], sx[ok]
        ok = src_ok[sy, sx]
        out[ys[ok], xs[ok]] = img[sy[ok], sx[ok]]
        todo[ys[ok], xs[ok]] = False
    return out, todo


def grain_transfer(filled: np.ndarray, known: np.ndarray, hole: np.ndarray, rng: np.random.Generator,
                   patch: int = 48, sigma: float = 2.0) -> np.ndarray:
    """Reinjeta no buraco o grão (passa-alta) de retalhos aleatórios do material visível."""
    f = filled.astype(np.float32)
    low = cv2.GaussianBlur(f, (0, 0), sigma)
    hp = f - low
    th, tw = hole.shape
    ok = _ero(known, patch // 2 + 1)
    ys, xs = np.nonzero(ok)
    if len(ys) < 50:
        return filled
    acc = np.zeros_like(f)
    wsum = np.zeros((th, tw, 1), np.float32)
    win = np.outer(np.hanning(patch + 2)[1:-1], np.hanning(patch + 2)[1:-1]).astype(np.float32)[..., None]
    hb = _dil(hole, patch // 2)
    step = patch // 2
    for y in range(0, th, step):
        for x in range(0, tw, step):
            if not hb[y:y + patch, x:x + patch].any():
                continue
            k = rng.integers(len(ys))
            sy, sx = ys[k] - patch // 2, xs[k] - patch // 2
            src = hp[max(sy, 0):sy + patch, max(sx, 0):sx + patch]
            h2, w2 = min(patch, th - y), min(patch, tw - x)
            if sy < 0 or sx < 0 or src.shape[0] < h2 or src.shape[1] < w2:
                continue
            acc[y:y + h2, x:x + w2] += src[:h2, :w2] * win[:h2, :w2]
            wsum[y:y + h2, x:x + w2] += win[:h2, :w2]
    g = acc / np.maximum(wsum, 1e-3)
    # o LaMa já deixa algum grão: soma só o que falta (mantém o desvio do anel)
    lowh = cv2.GaussianBlur(f, (0, 0), sigma)
    rstd = lambda a: float(1.4826 * np.median(np.abs(a - np.median(a)))) if a.size else 0.0   # robusto ao rejunte
    have = rstd((f - lowh)[hole])
    want = rstd(hp[ok])
    gain = float(np.sqrt(max(want ** 2 - have ** 2, 0.0)) / max(rstd(g[hole]) if hole.any() else 1.0, 1e-3))
    out = f.copy()
    out[hole] = (f + min(gain, 1.0) * g)[hole]
    return out.round().clip(0, 255).astype(np.uint8)


def push_pull(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Interpolação "push-pull" (pirâmide) dos valores com peso > 0 para o resto da imagem (suave)."""
    v = values.astype(np.float32)
    w = np.clip(weights.astype(np.float32), 0, 1)
    h, wd = w.shape
    if w.max() <= 0:
        return v
    if min(h, wd) <= 4 or w.min() >= 1:
        if w.min() >= 1:
            return v
        mean = (v * w[..., None]).sum((0, 1)) / w.sum() if v.ndim == 3 else (v * w).sum() / w.sum()
        return w[..., None] * v + (1 - w[..., None]) * mean if v.ndim == 3 else w * v + (1 - w) * mean
    size = (max(1, (wd + 1) // 2), max(1, (h + 1) // 2))
    vw = cv2.resize(v * (w[..., None] if v.ndim == 3 else w), size, interpolation=cv2.INTER_AREA)
    ws = cv2.resize(w, size, interpolation=cv2.INTER_AREA)
    vc = vw / (np.maximum(ws, 1e-6)[..., None] if v.ndim == 3 else np.maximum(ws, 1e-6))
    coarse = push_pull(vc, np.clip(ws * 4, 0, 1))
    up = cv2.resize(coarse, (wd, h), interpolation=cv2.INTER_LINEAR)
    if v.ndim == 3 and up.ndim == 2:
        up = up[..., None]
    return (w[..., None] * v + (1 - w[..., None]) * up) if v.ndim == 3 else (w * v + (1 - w) * up)


def harmonize(img: np.ndarray, known: np.ndarray, hole: np.ndarray, band: int = 4, island_sigma: float = 0.0,
              tol: float = 3.0) -> np.ndarray:
    """Harmonização de um preenchimento: (1) membrana tipo Poisson: a diferença de tom (passa-baixa) entre o anel
    original e a borda interna do buraco é interpolada para dentro do buraco; (2) corte de ilhas: o passa-baixa
    largo do preenchimento é limitado à faixa (p2..p98) do passa-baixa do anel original."""
    if not hole.any() or known.sum() < 30:
        return img
    f = img.astype(np.float32)
    lk = _nblur(f, known, 3.0)
    lf = _nblur(f, hole, 3.0)
    inner = hole & _dil(known, band)
    if inner.sum() >= 10:
        d = np.zeros_like(f)
        d[inner] = (lk - lf)[inner]
        corr = push_pull(d, inner.astype(np.float32))
        corr = _nblur(corr, hole, 2.0)
        f[hole] += corr[hole]
    if island_sigma > 0:
        low = _nblur(f, hole, island_sigma)
        ring = known & _dil(hole, int(3 * island_sigma))
        if ring.sum() >= 50:
            lowk = _nblur(f, known, island_sigma)[ring]
            lo = np.percentile(lowk, 2, axis=0) - tol
            hi = np.percentile(lowk, 98, axis=0) + tol
            f[hole] += (np.clip(low, lo, hi) - low)[hole]
    out = img.copy()
    out[hole] = f[hole].round().clip(0, 255).astype(np.uint8)
    return out


def _finish_surface(orig: np.ndarray, filled: np.ndarray, known: np.ndarray, hole: np.ndarray,
                    rng: np.random.Generator, periodic: bool = True, grain: bool = True) -> tuple[np.ndarray, dict]:
    """Pós-processo numa vista retificada: grade periódica OU grão, e harmonização."""
    info: dict = {"metodo": "lama"}
    if periodic:
        per = detect_period(cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY), known)
        if per is not None:
            filled, left = periodic_fill(filled, known, hole, per)
            info.update(metodo="periodico", periodo_px=list(per), sem_fonte_px=int(left.sum()))
    if grain and info["metodo"] == "lama":
        filled = grain_transfer(filled, known, hole, rng)
    diag = float(np.hypot(*hole.shape))
    # casamento de estatística por plano: mediana (Lab) do preenchimento -> mediana do anel do mesmo material
    ring = known & _dil(hole, max(8, int(0.08 * diag)))
    if ring.sum() >= 200 and hole.sum() >= 50:
        lab = _lab(filled)
        shift = np.median(lab[ring], axis=0) - np.median(lab[hole], axis=0)
        info["ajuste_tom_de"] = round(float(np.linalg.norm(shift)), 2)
        if np.linalg.norm(shift) > 1.0:
            lab[hole] += shift
            rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB) * 255.0
            filled = filled.copy()
            filled[hole] = np.clip(rgb[hole], 0, 255).round().astype(np.uint8)
    sig = max(4.0, 0.02 * diag)
    filled = harmonize(filled, known, hole, island_sigma=sig)
    return filled, info


# ============================================================================================================
# Máscara de efeitos
# ============================================================================================================
def core_mask(pieces: Sequence, shape: tuple[int, int], dilate: float = CORE_DILATE) -> np.ndarray:
    H, W = shape
    m = np.zeros(shape, bool)
    for p in pieces:
        m |= p.mask | (p.alpha > 0.02)
    return _dil(m, max(3, int(dilate * np.hypot(H, W))))


def effect_candidates(pieces: Sequence, groups: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Região candidata a sombra/reflexo/luz: anel em volta + faixa espelhada abaixo dos pés (sobre o piso; para
    planejados também sobre a parede, onde cai a luz do LED sob o aéreo)."""
    H, W = core.shape
    diag = np.hypot(H, W)
    floorish = _dil(groups == G_FLOOR, 6)
    wallish = _dil(np.isin(groups, (G_FLOOR, G_WALL)), 6)
    cand = np.zeros((H, W), bool)
    for p in pieces:
        builtin = getattr(p, "kind", "") == "planejado"
        hung = str(getattr(p, "label", "")).split(",")[0].strip().lower() in WALL_HUNG
        m = p.mask
        if not m.any():
            continue
        cand |= _dil(m, int((EFFECT_RING_BUILTIN if builtin else EFFECT_RING) * diag))
        ys, xs = np.nonzero(m)
        bottom = np.full(W, -1)
        top = np.full(W, H)
        np.maximum.at(bottom, xs, ys)
        np.minimum.at(top, xs, ys)
        frac = BAND_DOWN_BUILTIN if builtin else (BAND_DOWN_WALL_HUNG if hung else BAND_DOWN)
        band = np.zeros((H, W), bool)
        for c in np.nonzero(bottom >= 0)[0]:
            hgt = bottom[c] - top[c]
            band[bottom[c] + 1:min(H, bottom[c] + 1 + int(frac * hgt)), c] = True
        band = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_CLOSE, disk(6)) > 0
        cand |= band & (wallish if (builtin or hung) else floorish)
    prot = np.isin(groups, G_PROTECTED)
    for p in pieces:                   # painel fixo recortado da peça (cabeceira-painel): fica na foto
        panel = getattr(p, "wall_panel", None)
        if panel is not None:
            prot = prot | panel
    return cand & ~core & ~prot


def verify_effects(photo: np.ndarray, plate1: np.ndarray, cand: np.ndarray, core: np.ndarray):
    """Só ficam os candidatos em que a foto difere da placa provisória e que estão ligados à peça.
    Retorna (efeito, sombra, luz)."""
    lum = lambda a: cv2.GaussianBlur(a.astype(np.float32), (0, 0), 3.0)
    lp, l1 = lum(photo), lum(plate1)
    d = np.abs(lp - l1).max(axis=2)
    rel = d / (l1.mean(axis=2) + 10.0)
    eff = cand & ((rel > EFFECT_REL) | (d > EFFECT_ABS))
    eff = cv2.morphologyEx(eff.astype(np.uint8), cv2.MORPH_OPEN, disk(2)) > 0
    n, lab = cv2.connectedComponents(eff.astype(np.uint8))
    touch = set(np.unique(lab[_dil(core, 3)]).tolist()) - {0}
    eff = np.isin(lab, list(touch)) if touch else np.zeros_like(eff)
    eff = cv2.morphologyEx(eff.astype(np.uint8), cv2.MORPH_CLOSE, disk(4)) > 0
    eff &= (cand | _dil(cand, 2)) & ~core
    darker = lp.mean(axis=2) < l1.mean(axis=2)
    return eff, eff & darker, eff & ~darker


# ============================================================================================================
# Parede lisa (quase toda escondida) e rodapé
# ============================================================================================================
@dataclass
class WallProfile:
    """Cor mediana das paredes visíveis por altura (bins de ``step`` m) + rodapé detectado."""
    z: np.ndarray
    rgb: np.ndarray             # (n, 3)
    count: np.ndarray
    baseboard_h: float = 0.0    # 0 = sem rodapé detectado
    grain: float = 1.0
    flat: np.ndarray | None = None   # cor lisa (mediana) acima do rodapé

    def color_at(self, z: np.ndarray) -> np.ndarray:
        """Rodapé: perfil por altura; acima dele: a cor lisa (§3.6: "pintura lisa na cor média das paredes
        visíveis"; o perfil por altura acima do rodapé copiaria a faixa iluminada do backsplash e portas de
        armário mal segmentadas como parede)."""
        idx = np.clip(np.round((z - self.z[0]) / (self.z[1] - self.z[0])).astype(int), 0, len(self.z) - 1)
        col = self.rgb[idx]
        if self.flat is not None:
            col = np.where((z >= self.baseboard_h)[:, None], self.flat[None], col)
        return col


def wall_profile(photo: np.ndarray, box: RoomBox, visible_wall: np.ndarray, step: float = 0.01) -> WallProfile | None:
    ys, xs = np.nonzero(visible_wall & np.isfinite(box.Z))
    if ys.size < 300:
        return None
    z = box.Z[ys, xs]
    ch = float(box.room.ceiling_height)
    bins = np.arange(0.0, ch + step, step)
    idx = np.clip(((z - bins[0]) / step).astype(int), 0, len(bins) - 1)
    rgb = np.zeros((len(bins), 3), np.float32)
    cnt = np.bincount(idx, minlength=len(bins))
    px = photo[ys, xs].astype(np.float32)
    order = np.argsort(idx, kind="stable")
    sidx, spx = idx[order], px[order]
    starts = np.searchsorted(sidx, np.arange(len(bins)))
    ends = np.searchsorted(sidx, np.arange(len(bins)), side="right")
    for k in range(len(bins)):
        if ends[k] > starts[k]:
            rgb[k] = np.median(spx[starts[k]:ends[k]], axis=0)
    good = cnt >= 15
    if good.sum() < 5:
        return None
    gi = np.nonzero(good)[0]
    for k in range(len(bins)):                         # bins vazios: vizinho válido mais próximo
        if not good[k]:
            rgb[k] = rgb[gi[np.argmin(np.abs(gi - k))]]
    # rodapé: faixa contígua a partir do chão com cor bem diferente da parede (z 0,35..1,2 m)
    ref_sel = good & (bins >= 0.35) & (bins <= 1.2)
    bb = 0.0
    if ref_sel.sum() >= 5 and good[: int(0.3 / step)].sum() >= 3:
        ref = _lab(np.median(rgb[ref_sel], axis=0)[None, None])[0, 0]
        lab = _lab(rgb[None])[0]
        de = np.linalg.norm(lab - ref, axis=1)
        top = 0.0
        miss = 0
        for k in range(int(0.32 / step)):
            if not good[k]:
                miss += 1
                if miss > 3:
                    break
                continue
            if de[k] > 8.0:
                top = bins[k] + step
            else:
                break
        if 0.03 <= top <= 0.30:
            bb = float(top)
    # grão (desvio robusto do passa-alta da luminância das paredes visíveis)
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = (gray - cv2.GaussianBlur(gray, (0, 0), 1.5))[visible_wall]
    grain = float(min(6.0, 1.4826 * np.median(np.abs(hp - np.median(hp))))) if hp.size else 1.0
    above = z >= bb + 0.05
    flat = np.median(px[above], axis=0) if above.sum() >= 50 else np.median(px, axis=0)
    return WallProfile(bins, rgb, cnt, bb, grain, flat.astype(np.float32))


def _light_model(photo, box, prof: WallProfile, sel: np.ndarray, rng) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Gradiente de luz medido: razão luminância/perfil ajustada por um polinômio de 2º grau em (u, v)."""
    ys, xs = np.nonzero(sel & np.isfinite(box.Z))
    H, W = sel.shape
    if ys.size > 20000:
        k = rng.choice(ys.size, 20000, replace=False)
        ys, xs = ys[k], xs[k]
    if ys.size < 200:
        return lambda u, v: np.ones(u.shape, np.float32)
    base = prof.color_at(box.Z[ys, xs]).mean(axis=1)
    ratio = photo[ys, xs].astype(np.float32).mean(axis=1) / np.maximum(base, 1.0)
    u, v = xs / W - 0.5, ys / H - 0.5
    A = np.c_[np.ones_like(u), u, v, u * u, v * v, u * v]
    keep = np.ones(len(ratio), bool)
    for _ in range(3):                    # ajuste robusto: tira faixas iluminadas, quadros, sombras fortes
        coef, *_ = np.linalg.lstsq(A[keep], ratio[keep], rcond=None)
        res = ratio - A @ coef
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-3
        keep = np.abs(res) < 2.0 * mad
        if keep.sum() < 100:
            break

    def f(uu, vv):
        uu, vv = uu / W - 0.5, vv / H - 0.5
        r = coef[0] + coef[1] * uu + coef[2] * vv + coef[3] * uu * uu + coef[4] * vv * vv + coef[5] * uu * vv
        return np.clip(r, 0.6, 1.4).astype(np.float32)
    return f


def paint_plain(out: np.ndarray, photo: np.ndarray, box: RoomBox, prof: WallProfile, region: np.ndarray,
                light_sel: np.ndarray, rng: np.random.Generator) -> None:
    """Pinta ``region`` (pixels de parede) como parede lisa: perfil por altura × gradiente de luz + grão leve."""
    ys, xs = np.nonzero(region & np.isfinite(box.Z))
    if ys.size == 0:
        return
    light = _light_model(photo, box, prof, light_sel, rng)
    col = prof.color_at(box.Z[ys, xs]) * light(xs.astype(np.float32), ys.astype(np.float32))[:, None]
    noise = np.zeros(region.shape, np.float32)
    noise[ys, xs] = rng.normal(0, prof.grain, ys.size)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.7)[ys, xs]
    out[ys, xs] = np.clip(col + noise[:, None], 0, 255).round().astype(np.uint8)


# ============================================================================================================
# Resultado e nota
# ============================================================================================================
@dataclass
class PlateQuality:
    score: float                         # 0..100
    ok: bool
    per_plane: dict                      # nome -> {area_px, costura_de, preenchimento_de, grao_log2}
    line: dict                           # distância/continuidade da linha parede/piso
    area_generated: float
    redetected: list = field(default_factory=list)   # [{rotulo, area_pct}] reencontrados pelo OneFormer
    reasons: list = field(default_factory=list)       # motivos de reprovação (português)
    raw_score: float | None = None                    # nota antes da escala das reprovadas (ordena tentativas)
    unplanned: float = 0.0                            # fração da área gerada sem plano medido
    components: dict = field(default_factory=dict)    # pontos perdidos por defeito (nota contínua)
    realism: dict = field(default_factory=dict)       # névoa, fantasma e placas soltas (``plate_score``)
    openings: list = field(default_factory=list)      # portas/janelas novas que o OneFormer achou na placa
    hard: float = 0.0                                 # pior severidade entre as falhas duras (``HARD_KEYS``)
    hard_detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"nota": round(self.score, 1), "aprovada": self.ok,
                "nota_sem_teto": None if self.raw_score is None else round(self.raw_score, 1),
                "pontos_perdidos": self.components, "realismo": self.realism, "por_plano": self.per_plane,
                "linha_parede_piso": self.line, "area_gerada_pct": round(100 * self.area_generated, 2),
                "sem_plano_pct": round(100 * self.unplanned, 1),
                "moveis_reencontrados": self.redetected, "aberturas_novas": self.openings,
                "falha_dura": round(self.hard, 3), "falhas_duras": self.hard_detail,
                "motivos": list(self.reasons)}


@dataclass
class PlateResult:
    plate: np.ndarray
    mask: np.ndarray                     # máscara de remoção final (placa == foto fora dela)
    core: np.ndarray                     # peças dilatadas
    effect: np.ndarray                   # sombra ∪ luz removidas
    shadow: np.ndarray
    light: np.ndarray
    labels: np.ndarray                   # rótulo de plano por pixel (válido dentro da máscara)
    quality: PlateQuality | None
    info: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    kept_builtins: bool = False

    def report(self) -> dict:
        return {"placa": {"mascara_pct": round(100 * float(self.mask.mean()), 2),
                          "efeitos_pct": round(100 * float(self.effect.mean()), 2),
                          "sombra_px": int(self.shadow.sum()), "luz_px": int(self.light.sum()),
                          "planejados_mantidos": self.kept_builtins,
                          "planejados_por_padrao": self.info.get("planejados_por_padrao"),
                          "risco_planejados": self.info.get("risco_planejados"),
                          "cobertura": self.info.get("cobertura"),
                          "planos": self.info.get("planos", {}),
                          "tempos_s": {k: round(v, 2) for k, v in self.info.get("tempos", {}).items()},
                          "qualidade": None if self.quality is None else self.quality.to_dict(),
                          "estrategia": self.info.get("estrategia_portao", self.info.get("estrategia", "planos")),
                          "refino_sdxl": self.info.get("refino_sdxl"),
                          "tentativas": list(self.info.get("tentativas", [])),
                          "avisos": list(self.warnings)}}


def _bands(mask: np.ndarray, width: int):
    """Faixa interna (dentro da máscara, até ``width`` px da borda) e anel externo."""
    dist_in = ndimage.distance_transform_edt(mask)
    dist_out = ndimage.distance_transform_edt(~mask)
    return mask & (dist_in <= width), ~mask & (dist_out <= width)


def _fill_local_de(lab: np.ndarray, sel: np.ndarray, ring: np.ndarray, diag: float) -> float:
    """ΔE do preenchimento contra a cor **esperada no lugar**: o passa-baixa do anel do mesmo material estendido
    para dentro do buraco (push-pull), e não a média do anel inteiro. Uma faixa de sol, a queda de luz para o fundo
    ou um material escuro vizinho em um lado do anel puxam a média global, mas não a referência local de cada pixel.
    Média de ‖passa-baixa(preenchimento) − referência‖ sobre o preenchimento: uma mancha no meio continua contando."""
    s = max(4.0, 0.02 * diag)
    ref = push_pull(_nblur(lab, ring, s), ring.astype(np.float32))
    fl = _nblur(lab, sel, s)
    return float(np.linalg.norm((fl - ref)[sel], axis=1).mean())


def fill_de_limit(seam: float, grain_log2: float, relax: bool = True) -> float:
    """Limite do ΔE do preenchimento conforme a costura: com a costura e o grão bons, a cor já casa com o anel na
    borda e a diferença no meio é quase sempre gradiente de luz (sol, fundo da sala, sombra sob a bancada), então o
    limite sobe de ``preenchimento_de_max`` (costura >= ``costura_relaxa_de``) até ``preenchimento_de_max_relaxado``
    (costura <= 2). Grão ruim, ou ``relax=False`` (a placa tem mancha forte em outro lugar), não relaxa nada."""
    lo, hi = LIMITS["preenchimento_de_max"], LIMITS["preenchimento_de_max_relaxado"]
    if not relax or abs(grain_log2) > LIMITS["grao_relaxa_log2"]:
        return lo
    t = float(np.clip((LIMITS["costura_relaxa_de"] - seam) / (LIMITS["costura_relaxa_de"] - 2.0), 0.0, 1.0))
    return lo + t * (hi - lo)


def plane_stats(photo: np.ndarray, plate: np.ndarray, mask: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                min_area: float = 0.002) -> dict:
    """ΔE de costura e do preenchimento e razão de grão por plano (piso, teto, cada parede)."""
    H, W = mask.shape
    diag = np.hypot(H, W)
    w_seam = max(6, int(0.015 * diag))
    w_ring = max(10, int(0.03 * diag))
    lab_p = _lab(plate)
    gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.5))
    inner, ring = _bands(mask, w_seam)
    _, ring_w = _bands(mask, w_ring)
    out = {}
    for L in np.unique(labels[mask]):
        if L in (OTHER, PROTECTED):
            continue
        sel = mask & (labels == L)
        if sel.sum() < min_area * H * W:
            continue
        grp = G_FLOOR if L == FLOOR else (G_CEIL if L == CEIL else G_WALL)
        same = (labels == L) & (groups == grp)
        rin, rout = inner & sel, ring & same
        if rin.sum() < 30 or rout.sum() < 30:
            continue
        # costura local: passa-baixa do anel original x passa-baixa do preenchimento, pixel a pixel na faixa
        # interna (robusto a parede de 2 materiais dentro do mesmo plano)
        sig = max(3.0, w_seam / 2.0)
        lk = _nblur(lab_p, rout | (ring & (labels == L)), sig)
        lf = _nblur(lab_p, sel, sig)
        valid = rin & (cv2.GaussianBlur(rout.astype(np.float32), (0, 0), sig) > 0.05)
        if valid.sum() < 30:
            continue
        seam = float(np.linalg.norm((lk - lf)[valid], axis=1).mean())
        rw = ring_w & same
        fill_g = fill = float(np.linalg.norm(lab_p[sel].mean(0) - lab_p[rw].mean(0))) if rw.sum() >= 30 else seam
        if rw.sum() >= 30:      # média global ou referência local, a menor (anel poluído só de um lado)
            fill = min(fill_g, _fill_local_de(lab_p, sel, rw, diag))
        g_in = float(np.median(hp[sel])) + 0.3
        g_out = float(np.median(hp[rw if rw.sum() >= 30 else rout])) + 0.3
        name = LABEL_NAMES.get(int(L), f"parede{int(L)}")
        out[name] = {"area_px": int(sel.sum()), "costura_de": round(seam, 2), "preenchimento_de": round(fill, 2),
                     "preenchimento_global_de": round(fill_g, 2),
                     "razao_material": round(float(sel.sum()) / max(1, int((same & ~mask).sum())), 2),
                     "grao_log2": round(float(np.log2(g_in / g_out)), 2)}
    # o que não tem plano (vãos, protegidos, peças apoiadas em algo que fica): costura local contra o anel inteiro
    # (qualquer material), para que uma mancha ali também conte na nota
    sel = mask & np.isin(labels, (OTHER, PROTECTED))
    if sel.sum() >= min_area * H * W:
        sig = max(3.0, w_seam / 2.0)
        lk = _nblur(lab_p, ring, sig)
        lf = _nblur(lab_p, sel, sig)
        valid = inner & sel & (cv2.GaussianBlur(ring.astype(np.float32), (0, 0), sig) > 0.05)
        if valid.sum() >= 30:
            de = np.linalg.norm((lk - lf)[valid], axis=1)
            out["sem_plano"] = {"area_px": int(sel.sum()), "costura_de": round(float(de.mean()), 2),
                                "costura_p90_de": round(float(np.percentile(de, 90)), 2)}
    return out


def line_stats(plate: np.ndarray, photo: np.ndarray, mask: np.ndarray, box: RoomBox, radius: int = 8,
               labels: np.ndarray | None = None) -> dict:
    """Linha parede/piso projetada: na placa, dentro da máscara, procura a borda mais forte na normal da linha.
    Retorna a distância mediana (px), a continuidade (fração de pontos com borda) e, para calibrar, a mesma
    distância na parte visível da foto."""
    H, W = mask.shape
    g_plate = cv2.GaussianBlur(cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    g_photo = cv2.GaussianBlur(cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    offs = np.arange(-radius, radius + 1, dtype=np.float32)
    res_in, res_vis = [], []
    for w in box.room.walls:
        uv, nrm = box.floor_line(w)
        if len(uv) == 0:
            continue
        m = (uv[:, 0] > radius + 2) & (uv[:, 0] < W - radius - 3) & (uv[:, 1] > radius + 2) & (uv[:, 1] < H - radius - 3)
        uv, nrm = uv[m], nrm[m]
        if len(uv) == 0:
            continue
        # o ponto só conta se a caixa diz "parede w" logo acima e "piso" logo abaixo (linha visível na caixa)
        up = np.round(uv - 4 * nrm * np.sign(nrm[:, 1:2] + 1e-9)).astype(int)
        dn = np.round(uv + 4 * nrm * np.sign(nrm[:, 1:2] + 1e-9)).astype(int)
        lab = box.label if labels is None else np.where(mask, labels, box.label)
        valid = (lab[np.clip(up[:, 1], 0, H - 1), np.clip(up[:, 0], 0, W - 1)] == w.id) & \
                (lab[np.clip(dn[:, 1], 0, H - 1), np.clip(dn[:, 0], 0, W - 1)] == FLOOR)
        uv, nrm = uv[valid], nrm[valid]
        if len(uv) == 0:
            continue
        xs = uv[:, 0:1] + offs[None] * nrm[:, 0:1]
        ys = uv[:, 1:2] + offs[None] * nrm[:, 1:2]
        ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        inm = mask[vi, ui] & _ero(mask, 3)[vi, ui]
        for g, sel, acc in ((g_plate, inm, res_in), (g_photo, ~mask[vi, ui], res_vis)):
            if not sel.any():
                continue
            prof = ndimage.map_coordinates(g, [ys[sel].ravel(), xs[sel].ravel()], order=1).reshape(-1, len(offs))
            der = np.abs(np.diff(prof, axis=1))
            # a borda forte mais perto da linha (o topo do rodapé costuma ser mais forte que a borda do piso)
            strong = der >= 0.5 * der.max(1, keepdims=True)
            k = np.where(strong, np.abs(offs[:-1] + 0.5)[None], np.inf).argmin(1)
            acc.extend(zip(np.abs(offs[k] + 0.5).tolist(), der[np.arange(len(k)), k].tolist()))
    out: dict = {"pontos_mascara": len(res_in), "pontos_visiveis": len(res_vis)}
    if res_vis:
        v = np.array(res_vis)
        strong_vis = float(np.median(v[:, 1]))
        out["distancia_visivel_px"] = round(float(np.median(v[:, 0])), 2)
    else:
        strong_vis = 6.0
    if len(res_in) >= 8:
        a = np.array(res_in)
        found = a[:, 1] >= max(2.0, 0.35 * strong_vis)
        out["continuidade"] = round(float(found.mean()), 3)
        out["distancia_px"] = round(float(np.median(a[found, 0])) if found.any() else float(radius), 2)
    return out


def redetected_furniture(seg, pan, mask: np.ndarray, min_area: float = LIMITS["redetectado_area_min"]) -> list:
    """Móveis removíveis que o OneFormer encontra na placa, com área > ``min_area`` e metade dentro da máscara."""
    from . import instances as ri

    H, W = mask.shape
    found = []
    for c in ri.gather_candidates(seg, pan):
        a = c.mask.sum()
        if a < min_area * H * W:
            continue
        if (c.mask & mask).sum() >= 0.5 * a:
            found.append({"rotulo": c.label, "area_pct": round(100.0 * a / (H * W), 2)})
    return found


def rug_mask(seg, shape: tuple[int, int]) -> np.ndarray | None:
    """Pixels de tapete na segmentação da foto (o grupo "piso" inclui o tapete). None sem os rótulos ADE20K."""
    lbl, names = getattr(seg, "labels", None), getattr(seg, "id2label", None) or {}
    ids = [int(k) for k, v in names.items() if str(v) == "rug"]
    if lbl is None or not ids:
        return None
    H, W = shape
    if lbl.shape != (H, W):
        lbl = cv2.resize(lbl.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return np.isin(lbl, ids)


def assess(photo: np.ndarray, plate: np.ndarray, mask: np.ndarray, labels: np.ndarray, groups: np.ndarray,
           box: RoomBox | None, redetect: Callable | None = None, fill_info: dict | None = None,
           min_area: float = 0.002, core: np.ndarray | None = None, rug: np.ndarray | None = None,
           limits: dict | None = None) -> PlateQuality:
    """Nota automática da placa (0..100) e aprovação pelo :data:`LIMITS`. Plano preenchido que não pôde ser
    medido (nenhum pixel visível do próprio material em volta) reprova: o que não se mede não é aprovado.

    **Nota contínua** (PT-V3-4): 100 menos os pontos perdidos por defeito (:data:`SCORE_WEIGHTS` × severidade
    0..1): costura, cor, grão, linha parede/piso, área sem plano, área gerada e, dentro do preenchimento, **névoa,
    fantasma e placas soltas** (:mod:`plate_score`), mais móvel ou porta/janela que o OneFormer reencontra na
    placa. Reprovada (algum motivo, ou nota sem teto abaixo de ``nota_min``): a nota vira 0,6 × a nota sem teto,
    continua e sempre abaixo de 60, para ordenar tentativas e sementes e mostrar quão perto ficou. ``limits``:
    outro conjunto de limites (padrão :data:`LIMITS`; a sala vazia "ia" usa :data:`LIMITS_IA`)."""
    from . import plate_score as ps

    lim_ = LIMITS if limits is None else limits     # ``LIMITS_IA``: portão da sala vazia gerada ("ia")

    per = plane_stats(photo, plate, mask, labels, groups, min_area=min_area)
    line = line_stats(plate, photo, mask, box, labels=labels) if box is not None else {}
    area = float(mask.mean())
    reasons = []
    pen: dict[str, float] = {}

    def lose(key: str, sev: float, share: float = 1.0) -> None:
        pen[key] = pen.get(key, 0.0) + SCORE_WEIGHTS[key] * float(np.clip(sev, 0.0, 1.0)) * share

    H, W = mask.shape
    n_mask = int(mask.sum())
    unplanned = float((mask & np.isin(labels, (OTHER, PROTECTED))).sum()) / n_mask if n_mask else 0.0
    if area >= lim_["sem_plano_area_min"]:
        # cor média certa e estrutura errada (névoa, fantasma translúcido, parede inventada) passam pela costura;
        # sem plano medido não há como aprovar uma sala tão grande inventada
        if unplanned >= lim_["sem_plano_frac_max"]:
            reasons.append(f"geometria da sala não encontrada: {100 * unplanned:.0f}% do preenchimento sem plano "
                           f"medido ({100 * area:.0f}% da foto)")
        lose("sem_plano", (unplanned - 0.2) / 0.3)
    for L in np.unique(labels[mask]):
        if L in (OTHER, PROTECTED):
            continue
        name = LABEL_NAMES.get(int(L), f"parede{int(L)}")
        n = int((mask & (labels == L)).sum())
        if name in per or n < min_area * H * W or (fill_info or {}).get(name, {}).get("metodo") == "liso":
            continue
        if lim_.get("exige_referencia", True):
            reasons.append(f"{name} preenchido sem referência visível do material ({100 * n / (H * W):.1f}% da foto)")
        lose("plano_sem_referencia", 1.0)
    # sala pela geometria: parede escondida sem NADA dela à vista (atrás de planejados de ponta a ponta) não tem o que
    # copiar; o que sobra ali é membrana + grão (superfície cinza sem estrutura). Veto: reprova como falha dura, e o
    # portão prefere a tentativa que mantém os planejados (nada inventado)
    blind = 0
    for name, fi in (fill_info or {}).items():
        if isinstance(fi, dict) and fi.get("metodo") in ("sem_contexto", "sem_vista"):
            blind += int(fi.get("buraco_px", 0))
    sem_ref = blind / float(H * W)
    if sem_ref >= lim_["sem_referencia_max"] and lim_.get("exige_referencia", True):
        reasons.append(f"parede escondida sem nenhuma referência visível ({100 * sem_ref:.1f}% da foto): seria "
                       "inventada")
        lose("plano_sem_referencia", 1.0)
    # o limite da cor do preenchimento só relaxa se o que não tem plano não tem mancha forte (resto de móvel que a
    # reconferência não pegou): senão uma tentativa com fantasma passaria antes da tentativa limpa
    relax_fill = per.get("sem_plano", {}).get("costura_p90_de", 0.0) <= lim_["costura_sem_plano_max"]
    planes = {k: v for k, v in per.items() if k != "sem_plano"}
    tot = float(sum(v["area_px"] for v in planes.values())) or 1.0
    for name, s in per.items():
        if name == "sem_plano":
            if s["costura_de"] > lim_["costura_sem_plano_max"]:
                reasons.append(f"mancha no preenchimento sem plano (ΔE {s['costura_de']:.1f})")
            lose("mancha_sem_plano", (s["costura_de"] - 4.0) / 12.0)
            continue
        relax = relax_fill and s.get("razao_material", 0.0) <= lim_["relaxa_area_max"]
        lim = fill_de_limit(s["costura_de"], s["grao_log2"], relax)
        if limits is not None:
            lim = max(lim, lim_["preenchimento_de_max"])
        if s["preenchimento_de"] > lim:
            reasons.append(f"cor do preenchimento destoa no(a) {name} (ΔE {s['preenchimento_de']:.1f})")
        if s["costura_de"] > lim_["costura_de_max"]:
            reasons.append(f"costura visível no(a) {name} (ΔE {s['costura_de']:.1f})")
        if abs(s["grao_log2"]) > lim_["grao_log2_max"]:
            reasons.append(f"textura diferente no(a) {name} (grão {2 ** s['grao_log2']:.2f}×)")
        share = s["area_px"] / tot                    # cada plano pesa pela área que ocupa no preenchimento
        lose("costura", (s["costura_de"] - 2.0) / 8.0, share)
        lose("cor", (s["preenchimento_de"] - 0.5 * lim) / lim, share)
        lose("grao", (abs(s["grao_log2"]) - 0.3) / 1.2, share)
    if "distancia_px" in line:
        if line["distancia_px"] > lim_["linha_px_max"]:
            reasons.append(f"linha parede/piso fora do lugar ({line['distancia_px']:.1f} px)")
        if line["continuidade"] < lim_["linha_continuidade_min"]:
            reasons.append(f"rodapé/linha parede-piso interrompido ({100 * line['continuidade']:.0f}% contínuo)")
        lose("linha", 0.5 * (line["distancia_px"] - 2.0) / 6.0 + 0.5 * (1 - line["continuidade"] / 0.8))
    if area > lim_["area_gerada_max"]:
        reasons.append(f"área reconstruída muito grande ({100 * area:.0f}% da foto)")
    lose("area", (area - 0.5) / 0.4)
    # dentro do preenchimento: névoa, fantasma e placas soltas
    real = ps.realism(photo, plate, mask, labels, groups, core, rug)
    tex = real["textura"]
    # névoa: a pior entre o contraste/véu (por plano) e a energia de alta frequência (vale também sem plano)
    # e as manchas de véu/fumaça (``haze_metric``), que valem também onde não há plano medido
    fog = max(real["nevoa"]["severidade"], tex["nevoa_af"], real.get("veu", {}).get("severidade", 0.0))
    ghost = real["fantasma"]["severidade"]
    slab = real["placas_soltas"]["fracao"]
    # material: orientação das retas longas (``texture_metric``) ou cor/energia/orientação do piso preenchido contra
    # o piso visível sem tapete (``floor_material_metric``)
    struct, cont = tex["estrutura_nova"], real["contorno"]["severidade"]
    mat = max(tex["material"], real.get("material_piso", {}).get("severidade", 0.0))
    lose("nevoa", fog)
    lose("fantasma", ghost)
    lose("placas_soltas", slab / 0.3)
    lose("estrutura_nova", struct)
    lose("material", mat)
    lose("contorno", cont)
    if fog > lim_["nevoa_max"]:
        reasons.append(f"névoa/borrão no preenchimento (severidade {fog:.2f})")
    if ghost > lim_["fantasma_max"]:
        reasons.append(f"fantasma do móvel removido na sala vazia (severidade {ghost:.2f})")
    if slab > lim_["placas_soltas_max"]:
        reasons.append(f"manchas/placas soltas no preenchimento ({100 * slab:.0f}% da área gerada)")
    if struct > lim_["estrutura_nova_max"]:
        reasons.append(f"estrutura inventada no preenchimento (bordas {struct:.2f}: rejunte, bloco, cano ou placa)")
    if mat > lim_["material_max"]:
        reasons.append(f"material do preenchimento diferente do visível (severidade {mat:.2f})")
    if cont > lim_["contorno_max"]:
        reasons.append(f"contorno da peça removida visível na sala vazia (severidade {cont:.2f})")
    hard_detail = {"nevoa": round(fog, 3), "fantasma": round(ghost, 3), "estrutura_nova": round(struct, 3),
                   "material": round(mat, 3), "contorno": round(cont, 3)}
    if sem_ref >= lim_["sem_referencia_max"] and lim_.get("exige_referencia", True):
        hard_detail["sem_referencia"] = 1.0
    red, opn = [], []
    if redetect is not None:
        seg2, pan2 = redetect(plate)
        red = redetected_furniture(seg2, pan2, mask)
        opn = ps.new_openings(seg2, mask, groups, labels)
        if red:
            reasons.append("móvel reaparece na sala vazia: " + ", ".join(r["rotulo"] for r in red))
            lose("movel_reencontrado", 1.0)
        if opn:
            reasons.append("porta/janela nova na sala vazia: " + ", ".join(o["rotulo"] for o in opn))
            lose("abertura_nova", 1.0)
    raw = float(np.clip(100.0 - sum(pen.values()), 0.0, 100.0))
    if not reasons and raw < lim_["nota_min"]:
        top = sorted(pen.items(), key=lambda kv: -kv[1])[:3]
        reasons.append(f"nota baixa ({raw:.0f}/100; mais pontos perdidos em: "
                       + ", ".join(f"{k} {v:.0f}" for k, v in top) + ")")
    score = 0.6 * raw if reasons else raw
    comps = {k: round(v, 1) for k, v in sorted(pen.items(), key=lambda kv: -kv[1]) if v >= 0.05}
    hard = max(hard_detail.values()) if hard_detail else 0.0
    return PlateQuality(score, not reasons, per, line, area, red, reasons, raw, unplanned, comps, real, opn, hard,
                        hard_detail)


# ============================================================================================================
# Preenchimento por plano
# ============================================================================================================
def _protected_labels(mask, labels, groups, box: RoomBox | None, diag) -> np.ndarray:
    """Pixels mascarados que pertencem a janela/porta/cortina: dentro de uma abertura do modelo da sala ou a
    menos de PROTECT_REACH de pixels visíveis dessas classes."""
    prot_vis = np.isin(groups, G_PROTECTED) & ~mask
    near = mask & _dil(prot_vis, max(2, int(PROTECT_REACH * diag)))
    if box is not None:
        for op in box.room.openings:
            sel = mask & (labels == op.wall) & np.isfinite(box.S)
            ins = (box.S >= op.s0) & (box.S <= op.s1)
            z0 = op.z0
            # a borda de baixo da abertura foi medida só nos pixels visíveis: se um móvel a esconde, a janela
            # (ou porta de sacada) pode continuar até o chão atrás dele
            edge = (labels == op.wall) & ins & (box.Z >= op.z0 - 0.05) & (box.Z <= op.z0 + 0.10)
            if edge.sum() >= 20 and (edge & mask).sum() >= 0.3 * edge.sum():
                z0 = 0.0
            near |= sel & ins & (box.Z >= z0) & (box.Z <= op.z1)
    return near


def _resting_on_kept(mask, labels, groups, diag, ring_frac: float = 0.015) -> np.ndarray:
    """Pixels mascarados cujo plano da caixa (piso/teto/parede) está escondido atrás de algo que FICA na foto
    (copo sobre a bancada mantida, vaso sobre a ilha): o anel visível em volta quase não tem o material do plano
    e é dominado por objetos. Preencher na vista do plano aí inventa uma mancha de piso/parede no meio da bancada;
    esses pixels vão para o espaço da imagem com contexto local."""
    out = np.zeros(mask.shape, bool)
    plane = mask & ((labels >= 0) | (labels == FLOOR) | (labels == CEIL))
    if not plane.any():
        return out
    surfaces = np.isin(groups, (G_FLOOR, G_WALL, G_CEIL) + G_PROTECTED)
    r = max(4, int(ring_frac * diag))
    H, W = mask.shape
    for L in np.unique(labels[plane]):
        grp = G_FLOOR if L == FLOOR else (G_CEIL if L == CEIL else G_WALL)
        n, cc, st, _ = cv2.connectedComponentsWithStats((plane & (labels == L)).astype(np.uint8), connectivity=8)
        for k in range(1, n):
            x, y, w, h, _ = st[k]
            x0, y0, x1, y1 = max(0, x - r - 1), max(0, y - r - 1), min(W, x + w + r + 1), min(H, y + h + r + 1)
            comp = cc[y0:y1, x0:x1] == k
            ring = _dil(comp, r) & ~mask[y0:y1, x0:x1]
            nr = int(ring.sum())
            if nr < 30:
                continue
            g = groups[y0:y1, x0:x1][ring]
            if (g == grp).mean() < 0.15 and surfaces[y0:y1, x0:x1][ring].mean() < 0.5:
                out[y0:y1, x0:x1] |= comp
    return out


def clutter_mask(photo: np.ndarray, surface: np.ndarray, max_area: float = 0.0015, de: float = 14.0) -> np.ndarray:
    """Objetos pequenos e contrastantes sobre uma superfície (tomada, interruptor, quadro pequeno): não servem de
    contexto para o preenchimento, senão o LaMa os repete na parede (objeto inventado)."""
    H, W = surface.shape
    lab = _lab(photo)
    med = _nblur(lab, surface, max(6.0, 0.02 * np.hypot(H, W)))
    dev = np.linalg.norm(lab - med, axis=2) > de
    blobs = _dil(dev & surface, 2)
    n, cc, st, _ = cv2.connectedComponentsWithStats(blobs.astype(np.uint8), connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = st[1:, cv2.CC_STAT_AREA] <= max_area * H * W
    return small[cc] & surface


def _inpaint_local(inpaint: InpaintFn, img: np.ndarray, hole: np.ndarray, margin_frac: float = 0.25) -> np.ndarray:
    """Inpainting no espaço da imagem, um recorte por componente (contexto local)."""
    out = img.copy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
    H, W = hole.shape
    for k in range(1, n):
        x, y, w, h, _ = stats[k]
        mg = max(24, int(margin_frac * max(w, h)))
        x0, y0, x1, y1 = max(0, x - mg), max(0, y - mg), min(W, x + w + mg), min(H, y + h + mg)
        sub = lab[y0:y1, x0:x1] == k
        res = inpaint(out[y0:y1, x0:x1], sub)
        out[y0:y1, x0:x1][sub] = res[sub]
    return out


def fill_planes(inpaint: InpaintFn, photo: np.ndarray, mask: np.ndarray, groups: np.ndarray, box: RoomBox | None,
                depth: np.ndarray | None = None, rng: np.random.Generator | None = None,
                max_side: int = MAX_SIDE, rug: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Preenche ``mask`` por plano. Retorna (placa, rótulos, informações por plano). ``rug``: tapete que ficou
    na foto; nunca é contexto do piso (o grupo "piso" da segmentação inclui o tapete, e o LaMa copiava a estampa
    ou a cor dele para o piso de baixo)."""
    rng = rng or np.random.default_rng(0)
    H, W = mask.shape
    diag = float(np.hypot(H, W))
    info: dict = {}
    if box is None:                                   # sem modelo da sala: só espaço da imagem
        out = inpaint(photo, mask)
        out[~mask] = photo[~mask]
        info["imagem"] = {"metodo": "lama-imagem", "area_px": int(mask.sum())}
        labels = np.full((H, W), OTHER, np.int32)
        # janela/porta/cortina atrás do móvel continua "protegida" mesmo sem modelo da sala (o refino SDXL não a
        # repinta: antes todo pixel ficava "outro" e o SDXL trocava a janela escondida por parede)
        labels[_protected_labels(mask, labels, groups, None, diag)] = PROTECTED
        return out, labels, info
    labels, linfo = assign_labels(mask, groups, box, depth)
    info.update(linfo)
    out = photo.copy()
    remaining = mask.copy()
    return _fill_planes_body(inpaint, photo, mask, groups, box, labels, info, out, remaining, rng, max_side, rug, diag)


def assign_labels(mask: np.ndarray, groups: np.ndarray, box: RoomBox, depth: np.ndarray | None = None,
                  wall_to_floor: bool = True) -> tuple[np.ndarray, dict]:
    """Rótulo de plano de cada pixel mascarado: ray-cast na caixa, "outro" para o que a profundidade mostra além da
    caixa (vista por uma porta), parede da caixa que vira piso onde a foto mostra piso (``wall_to_floor``),
    protegidos (janela/porta/cortina) e o que está apoiado em objeto que fica."""
    H, W = mask.shape
    diag = float(np.hypot(H, W))
    info: dict = {}
    labels = box.label.copy()
    if depth is not None:                             # visto por uma porta: bem mais longe que a caixa
        dep = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST) \
            if depth.shape != (H, W) else depth
        beyond = (dep > 1.2 * box.t) & ~np.isin(groups, G_PROTECTED) & np.isfinite(dep)
        agree = np.abs(dep - box.t) / np.maximum(box.t, 1e-3) < 0.12
        known = ~_dil(mask, 3)
        for L in np.unique(labels[mask]):
            bad = known & (labels == L) & beyond
            if L == OTHER or bad.sum() == 0:
                continue
            d_ok = cv2.distanceTransform((~(known & (labels == L) & agree)).astype(np.uint8), cv2.DIST_L2, 5)
            d_bad = cv2.distanceTransform((~bad).astype(np.uint8), cv2.DIST_L2, 5)
            labels[mask & (labels == L) & (d_bad < d_ok)] = OTHER
    # parede da caixa onde a foto mostra piso (MoGe/RoomModel erram muito em cozinhas planejadas): o plano do piso
    # é a parte confiável; os pixels mascarados dessa parede cujo vizinho visível é piso viram piso
    vis_surf = ~_dil(mask, 2) & np.isin(groups, (G_FLOOR, G_WALL, G_CEIL))
    if wall_to_floor and vis_surf.sum() >= 100:
        idx = ndimage.distance_transform_edt(~vis_surf, return_distances=False, return_indices=True)
        near = groups[idx[0], idx[1]]
        down = box.rays[..., 2] < 0
        for w in box.room.walls:
            visw = vis_surf & (labels == w.id)
            if visw.sum() < 50:
                continue
            ffrac = float((groups[visw] == G_FLOOR).mean())
            if ffrac > 0.3:
                sel = mask & (labels == w.id) & (near == G_FLOOR) & down
                labels[sel] = FLOOR
                info.setdefault("parede_virou_piso", {})[f"parede{w.id}"] = {"piso_visivel": round(ffrac, 2),
                                                                             "px": int(sel.sum())}
    prot = _protected_labels(mask, labels, groups, box, diag)
    labels[prot] = PROTECTED
    on_obj = _resting_on_kept(mask, labels, groups, diag)
    if on_obj.any():
        labels[on_obj] = OTHER
        info["apoiado_em_objeto_px"] = int(on_obj.sum())
    return labels, info


def _fill_planes_body(inpaint, photo, mask, groups, box, labels, info, out, remaining, rng, max_side, rug, diag):
    H, W = mask.shape
    # ---- piso e teto (vista de cima / de baixo), contexto só do próprio material ------------------------------
    for key, grp in ((FLOOR, G_FLOOR), (CEIL, G_CEIL)):
        sel = mask & (labels == key)
        if sel.sum() < 20:
            continue
        r = box.rect_for(box.planes[key], sel, max_side)
        if r is None:
            labels[sel] = OTHER
            continue
        Hm, size = r
        same = (labels == key) & ~mask & (groups == grp)
        if key == FLOOR and rug is not None:
            same &= ~_dil(rug, 2)
        back, cover, pinfo = _fill_rect(inpaint, out, photo, mask, same, Hm, size, rng, ctx_restrict=True)
        done = sel & cover
        out[done] = back[done]
        remaining &= ~done
        labels[sel & ~cover] = OTHER
        info[LABEL_NAMES[key]] = {**pinfo, "area_px": int(done.sum()), "vista": list(size)}

    # ---- paredes ------------------------------------------------------------------------------------------------
    visible_wall = ~mask & (groups == G_WALL) & (labels >= 0)
    prof = wall_profile(photo, box, visible_wall)
    for w in box.room.walls:
        sel = mask & (labels == w.id)
        if sel.sum() < 20:
            continue
        tot = int((labels == w.id).sum())
        vis = visible_wall & (labels == w.id)
        frac = float(vis.sum()) / max(tot, 1)
        own = wall_profile(photo, box, vis) if vis.sum() >= 1500 else None
        if own is not None:               # cor da própria parede (paredes vizinhas podem ser de outro material)
            if own.baseboard_h == 0 and prof is not None:
                own.baseboard_h = prof.baseboard_h
            wprof = own
        else:
            wprof = prof
        if wprof is not None and (frac < PLAIN_WALL_VISIBLE or vis.sum() < 0.003 * H * W):
            light_sel = vis if vis.sum() >= 2000 else visible_wall
            paint_plain(out, photo, box, wprof, sel, light_sel, rng)
            ring_known = ~mask & (labels == w.id) & (groups == G_WALL)
            out = harmonize(out, ring_known, sel)
            remaining &= ~sel
            info[f"parede{w.id}"] = {"metodo": "liso", "visivel": round(frac, 3), "area_px": int(sel.sum()),
                                     "rodape_m": round(wprof.baseboard_h, 3)}
            continue
        r = box.rect_for(box.planes[w.id], sel, max_side, pad_px=60)
        if r is None:
            labels[sel] = OTHER
            continue
        Hm, size = r
        same = (labels == w.id) & ~mask & (groups == G_WALL)
        same &= ~clutter_mask(photo, same)
        back, cover, pinfo = _fill_rect(inpaint, out, photo, remaining | sel, same, Hm, size, rng,
                                        ctx_restrict=True)
        done = sel & cover
        out[done] = back[done]
        # rodapé reto: redesenha a faixa do rodapé (perfil das paredes visíveis) na altura certa
        if wprof is not None and wprof.baseboard_h > 0:
            bb = done & (box.Z < wprof.baseboard_h) & np.isfinite(box.Z)
            if bb.any():
                paint_plain(out, photo, box, WallProfile(wprof.z, wprof.rgb, wprof.count, wprof.baseboard_h,
                                                         wprof.grain * 0.5, None), bb, vis if vis.sum() >= 2000
                            else visible_wall, rng)
                pinfo["rodape_redesenhado_px"] = int(bb.sum())
        remaining &= ~done
        labels[sel & ~cover] = OTHER
        info[f"parede{w.id}"] = {**pinfo, "visivel": round(frac, 3), "area_px": int(done.sum()),
                                 "vista": list(size)}

    # ---- "outro" (vãos de porta, sem plano) e protegidos: espaço da imagem, contexto local --------------------
    for key in (OTHER, PROTECTED):
        sel = remaining & ((labels == key) if key == PROTECTED else (labels != PROTECTED))
        if sel.sum() == 0:
            continue
        out = _inpaint_local(inpaint, out, sel)
        remaining &= ~sel
        info[LABEL_NAMES[key]] = {"metodo": "lama-local", "area_px": int(sel.sum())}
    out[~mask] = photo[~mask]
    return out, labels, info


def _fill_rect(inpaint, cur, photo, hole_img, same_img, Hm, size, rng, ctx_restrict: bool):
    """Vista retificada: buraco = máscara inteira (+ fora da foto [+ outro material]); LaMa; textura; volta."""
    tw, th = size
    H, W = hole_img.shape
    rect = _warp_in(cur, Hm, size)
    orig = _warp_in(photo, Hm, size)
    rh = _dil(_warp_mask_in(hole_img, Hm, size), 2)
    inside = _warp_in(np.full((H, W), 255, np.uint8), Hm, size, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT) > 0
    same = _warp_mask_in(same_img, Hm, size, 200) & ~rh & inside
    n_hole = int((rh & inside).sum())
    if int(same.sum()) < max(MIN_CONTEXT_PX, MIN_CONTEXT_FRAC * n_hole):
        # quase nada do material do plano à vista: o LaMa com o buraco do tamanho da vista devolve uma cor
        # inventada (cinza/verde-azulado). Esses pixels vão para o espaço da imagem com contexto local.
        return cur, np.zeros((H, W), bool), {"metodo": "sem_contexto", "contexto_px": int(same.sum())}
    rmask = rh | ~inside
    if ctx_restrict:
        rmask |= ~same
    filled = inpaint(rect, rmask)
    filled, pinfo = _finish_surface(orig, filled, same, rmask & inside, rng)
    back = cv2.warpPerspective(filled, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    full = np.full((th, tw), 255, np.uint8)
    cover = cv2.warpPerspective(cv2.erode(full, disk(2)), Hm, (W, H), flags=cv2.INTER_NEAREST) > 0
    return back, cover, pinfo


# ============================================================================================================
# Pontos de entrada
# ============================================================================================================
def build_plate(photo: np.ndarray, pieces: Sequence, seg, room=None, depth: np.ndarray | None = None,
                inpaint: InpaintFn | None = None, redetect: Callable | None = None, seed: int = 0,
                max_side: int = MAX_SIDE, assess_quality: bool = True, strategy: str = "planos",
                core_dilate: float = CORE_DILATE, ai_fill: Callable | None = None) -> PlateResult:
    """Sala vazia a partir das peças a remover (``FurnitureInstance`` ou qualquer objeto com ``mask``, ``alpha``
    e ``kind``). ``seg``: segmentação semântica da foto. ``room``: ``RoomModel`` (sem ele, só espaço da
    imagem). ``depth``: profundidade do MoGe (detecta o que é visto por portas). ``redetect(placa) ->
    (seg, pan)``: OneFormer para a nota. ``strategy``: "planos" (LaMa por plano) ou "imagem" (LaMa no espaço da
    imagem; a nota usa os mesmos planos da caixa, para medir as duas do mesmo jeito). ``core_dilate``: dilatação
    do núcleo (fração da diagonal). Garante ``placa[~mask] == foto[~mask]`` byte a byte.

    ``strategy="ia"``: máscara e preenchimento como "geometria" (sem modelo da sala, por plano) e depois
    ``ai_fill(foto, placa, mascara, grupos, profundidade, semente) -> (placa, info)`` (o gerador de difusão,
    ``design.empty.PlateFiller``) refina só dentro da máscara; a nota usa :data:`LIMITS_IA`."""
    if strategy not in STRATEGIES:
        raise ValueError(f"estratégia de placa desconhecida: {strategy}")
    ai = strategy == "ia"
    if ai:
        if ai_fill is None:
            raise ValueError("estratégia \"ia\" sem o preenchimento por IA")
        strategy = "geometria"
    if inpaint is None:
        inpaint = LazyLama()
    t: dict = {}
    t0 = time.perf_counter()
    H, W = photo.shape[:2]
    groups = seg.groups
    warnings: list[str] = []
    rng = np.random.default_rng(seed)
    if not pieces:
        return PlateResult(photo.copy(), np.zeros((H, W), bool), np.zeros((H, W), bool), np.zeros((H, W), bool),
                           np.zeros((H, W), bool), np.zeros((H, W), bool), np.full((H, W), OTHER, np.int32),
                           None, {"planos": {}, "tempos": {}}, ["nenhuma peça para remover"])
    box = None
    if room is not None and room.walls is not None:
        try:
            box = RoomBox(room, W, H)
        except Exception as e:  # modelo da sala degenerado: segue no espaço da imagem
            warnings.append(f"modelo da sala inválido ({e}); sala vazia só no espaço da imagem")
    else:
        warnings.append("sem modelo da sala: sala vazia só no espaço da imagem (linhas podem entortar)")
    beyond_ids: list = []
    if strategy == "geometria" and box is not None and depth is not None:
        # peça vista por uma porta/abertura (cama do quarto ao lado): não há plano desta sala atrás dela; tirá-la
        # deixaria uma silhueta lisa inventada. Fica na foto (e no vídeo não entra: já está lá)
        beyond_ids = pieces_beyond_openings(pieces, box, depth, groups)
        if beyond_ids:
            pieces = [p for p in pieces if getattr(p, "id", None) not in beyond_ids]
            warnings.append(f"{len(beyond_ids)} peça(s) vista(s) por uma porta/abertura ficaram na foto")
            if not pieces:
                return PlateResult(photo.copy(), np.zeros((H, W), bool), np.zeros((H, W), bool),
                                   np.zeros((H, W), bool), np.zeros((H, W), bool), np.zeros((H, W), bool),
                                   np.full((H, W), OTHER, np.int32), None,
                                   {"planos": {}, "tempos": {}, "pecas_alem_da_abertura": beyond_ids}, warnings)
    core = core_mask(pieces, (H, W), core_dilate)
    holes = np.zeros((H, W), bool)
    if strategy == "geometria":
        core2 = fill_mask_holes(core, MASK_HOLE_MAX * H * W)
        core2 |= resting_on_removed(core2, groups, RESTING_MAX * H * W)
        if box is not None:
            core2 |= wall_halo(photo, pieces, groups, box.label, core2, zmap=box.Z)
        holes = core2 & ~core          # devolvidos com as peças (região de efeito), como sombra/reflexo
        core = core2
    rug = rug_mask(seg, (H, W))
    cand = effect_candidates(pieces, groups, core)
    t["mascaras"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if strategy == "geometria" and box is not None:
        # passada 1 pela própria geometria: a placa provisória não copia o reflexo/sombra do móvel (o LaMa
        # reproduzia o reflexo escuro no piso polido e ele ficava fora da máscara)
        from .geofill import fill_geometry

        plate1 = fill_geometry(photo, core | cand, groups, box, depth, np.random.default_rng(seed), None,
                               max_side, rug=rug)[0]
    else:
        plate1 = inpaint(photo, core | cand)            # passada 1 (barata): só para medir os efeitos
    t["passada1"] = time.perf_counter() - t0
    effect, shadow, light = verify_effects(photo, plate1, cand, core)
    effect = effect | holes
    del plate1
    mask = core | _dil(effect, 3)
    t0 = time.perf_counter()
    if strategy == "imagem":
        plate = np.array(inpaint(photo, mask), np.uint8, copy=True)
        plate[~mask] = photo[~mask]
        labels = box.label.copy() if box is not None else np.full((H, W), OTHER, np.int32)
        # mesmos protegidos da estratégia por plano: a janela/porta escondida pelo móvel não vira parede no refino
        labels[_protected_labels(mask, labels, groups, box, float(np.hypot(H, W)))] = PROTECTED
        pinfo = {"imagem": {"metodo": "lama-imagem", "area_px": int(mask.sum())}}
    elif strategy == "geometria" and box is not None:
        from .geofill import fill_geometry

        plate, labels, pinfo = fill_geometry(photo, mask, groups, box, depth, rng, inpaint, max_side, rug=rug)
    else:
        if strategy == "geometria":
            warnings.append("sem modelo da sala: a sala vazia não pôde ser refeita pela geometria")
        plate, labels, pinfo = fill_planes(inpaint, photo, mask, groups, box, depth, rng, max_side, rug=rug)
    t["passada2_planos"] = time.perf_counter() - t0
    panels = [pp.wall_panel for pp in pieces if getattr(pp, "wall_panel", None) is not None]
    if panels:
        n_ext = extend_wall_panels(plate, mask, labels, np.logical_or.reduce(panels))
        if n_ext:
            pinfo["painel_estendido"] = {"metodo": "painel espelhado para baixo", "area_px": n_ext}
    ai_info = None
    if ai:
        t0 = time.perf_counter()
        plate, ai_info = ai_fill(photo, plate, mask, groups, depth, seed)
        plate = np.array(plate, np.uint8, copy=True)
        plate[~mask] = photo[~mask]
        t["ia"] = time.perf_counter() - t0
        strategy = "ia"
    if not np.array_equal(plate[~mask], photo[~mask]):     # regra absoluta: nunca deveria acontecer
        raise AssertionError("placa difere da foto fora da máscara")
    q = None
    if assess_quality:
        t0 = time.perf_counter()
        q = assess(photo, plate, mask, labels, groups, box, redetect, pinfo, core=core, rug=rug,
                   limits=LIMITS_IA if ai else None)
        t["nota"] = time.perf_counter() - t0
    info = {"planos": pinfo, "tempos": t, "estrategia": strategy, "dilatacao_nucleo": core_dilate,
            "pecas_alem_da_abertura": beyond_ids}
    if ai_info is not None:
        info["ia"] = ai_info
    return PlateResult(plate, mask, core, effect, shadow, light, labels, q, info, warnings)


def attempt_rank(q: PlateQuality) -> tuple:
    """Ordem das tentativas do portão: aprovada > área gerada no limite > nota sem teto, com as falhas duras e o
    nº de motivos só como penalidade pequena (desempate dentro de poucos pontos). Antes a falha dura arredondada e
    o nº de motivos vinham antes da nota: uma tentativa de nota 0,5 ganhava de outra de 17,6."""
    raw = q.raw_score if q.raw_score is not None else q.score
    return (q.ok, q.area_generated <= LIMITS["area_gerada_max"],
            float(raw) - RANK_HARD_PTS * float(getattr(q, "hard", 0.0) or 0.0) - RANK_REASON_PTS * len(q.reasons))


def pieces_beyond_openings(pieces: Sequence, box: RoomBox, depth: np.ndarray, groups: np.ndarray | None = None,
                           ratio: float = BEYOND_RATIO, frac: float = BEYOND_FRAC) -> list:
    """Ids das peças vistas **além** de uma parede da caixa (por uma porta ou abertura, em outro cômodo): a
    profundidade do MoGe na maior parte da peça fica bem atrás do plano da parede que o raio atravessa. Não há
    superfície desta sala atrás delas para copiar; ficam na foto (regra geral, sem ajuste por foto)."""
    H, W = box.label.shape
    dep = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST) \
        if depth.shape != (H, W) else depth.astype(np.float32)
    wall = box.label >= 0
    # piso da caixa além de todo o piso visível desta sala (caixa sem parede do fundo: o "piso" segue pela porta)
    far_t = np.inf
    if groups is not None and pieces:
        allm = np.zeros((H, W), bool)
        for p in pieces:
            allm |= np.asarray(p.mask, bool)
        vis = (box.label == FLOOR) & ~_dil(allm, 3) & (groups == G_FLOOR) & np.isfinite(box.t)
        vis &= ~((dep > ratio * box.t) & np.isfinite(dep))          # piso visto pela porta não conta
        if vis.sum() >= 200:
            far_t = float(np.percentile(box.t[vis], 97)) * 1.08
    out = []
    for p in pieces:
        m = np.asarray(p.mask, bool)
        n = int(m.sum())
        if n < 50:
            continue
        # sobre o piso da caixa além do piso visível, e a própria peça (profundidade) também lá longe
        on_far = m & (box.label == FLOOR) & np.isfinite(box.t) & (box.t > far_t) & np.isfinite(dep) & (dep > far_t)
        sel = m & wall & np.isfinite(dep) & (dep > 0) & np.isfinite(box.t)
        if on_far.sum() >= frac * n:
            out.append(getattr(p, "id", None))
            continue
        if sel.sum() < 0.6 * n:                    # a peça cai sobre o piso/teto da caixa: está nesta sala
            continue
        r = dep[sel] / np.maximum(box.t[sel], 1e-3)
        if float(np.mean(r > ratio)) >= frac and float(np.median(r)) > ratio:
            out.append(getattr(p, "id", None))
    return [i for i in out if i is not None]


def resting_on_removed(core: np.ndarray, groups: np.ndarray, max_area: float) -> np.ndarray:
    """Objeto pequeno não detectado **apoiado** numa peça que sai (bola decorativa na mesinha, enfeite na
    cômoda): componente pequena de material que não é superfície da sala nem abertura, com a base encostada na
    máscara por baixo. Ficaria flutuando no ar da sala vazia; sai junto (volta com a peça)."""
    surf = np.isin(groups, (G_FLOOR, G_WALL, G_CEIL) + G_PROTECTED + (GROUP_IDS["ceiling_fixture"],))
    cand = ~core & ~surf
    if not cand.any():
        return np.zeros_like(core)
    n, cc, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    below = np.zeros_like(core)
    for k in range(1, 5):
        below[:-k] |= core[k:]
    out = np.zeros_like(core)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a > max_area or a < 15:
            continue
        comp = cc[y:y + h, x:x + w] == i
        # base da componente: o pixel mais baixo de cada coluna
        cols = np.nonzero(comp.any(0))[0]
        bot = y + (comp.shape[0] - 1 - np.argmax(comp[::-1], axis=0))[cols]
        if float(below[bot, x + cols].mean()) >= 0.5:
            out[y:y + h, x:x + w] |= comp
    return out


def wall_halo(photo: np.ndarray, pieces: Sequence, groups: np.ndarray, box_label: np.ndarray, core: np.ndarray,
              cap: float = HALO_CAP, k_grain: float = 3.0, min_dark: float = 6.0,
              zmap: np.ndarray | None = None) -> np.ndarray:
    """Sombra projetada, oclusão e borda escura (antisserrilhado) de peça na parede ou encostada nela (quadro,
    espelho, sofá, estante), logo fora da máscara e só sobre a parede: a parede lisa no anel é prevista pela parede um pouco mais longe; o anel cresce enquanto fica
    mais escuro que essa previsão (além de ``k_grain`` × grão), até ``cap`` × diagonal. Sem isso a parede lisa
    refeita copiava a borda escura e o contorno do quadro (ou a faixa escura em cima do sofá) ficava desenhado na
    sala vazia. O que entra volta no vídeo com a peça (região de efeito)."""
    H, W = core.shape
    diag = float(np.hypot(H, W))
    capr = max(3, int(cap * diag))
    lum = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32)
    wall = (groups == G_WALL) & (box_label >= 0)
    if zmap is not None:                               # a faixa do rodapé é escura de propósito: nunca é sombra
        wall &= ~(np.isfinite(zmap) & (zmap < HALO_MIN_Z))
    out = np.zeros_like(core)
    for p in pieces:
        m = np.asarray(p.mask, bool)
        if m.sum() < 100 or float((groups[_dil(m, 3) & ~m] == G_WALL).mean()) < 0.1:
            continue                                   # só peça encostada na parede (quadro, sofá, estante)
        near = _dil(m, capr) & ~core & wall
        outer = _dil(m, 3 * capr) & ~_dil(m, 2 * capr) & wall & ~core
        if outer.sum() < 50 or not near.any():
            continue
        ref = _nblur(lum, outer, max(3.0, capr))
        hp = lum - cv2.GaussianBlur(lum, (0, 0), 1.0)
        g = 1.4826 * float(np.median(np.abs(hp[outer]))) + 0.5
        dark = near & (lum < ref - max(min_dark, k_grain * g))
        # só o que está ligado à máscara (crescimento a partir da borda)
        grown = core.copy()
        for _ in range(capr):
            nxt = grown | (_dil(grown, 1) & dark)
            if nxt.sum() == grown.sum():
                break
            grown = nxt
        out |= grown & ~core
    return out


def fill_mask_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    """Buracos fechados pequenos dentro da máscara (etiqueta, canto de tapete, vão entre pés) entram na máscara:
    ilhas da foto no meio do preenchimento viram manchas soltas na sala vazia."""
    inv = ~mask
    n, cc, st, _ = cv2.connectedComponentsWithStats(inv.astype(np.uint8), connectivity=4)
    out = mask.copy()
    H, W = mask.shape
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a <= max_area and x > 0 and y > 0 and x + w < W and y + h < H:
            out[cc == k] = True
    return out


def extend_wall_panels(plate: np.ndarray, mask: np.ndarray, labels: np.ndarray, panels: np.ndarray) -> int:
    """Continua para baixo o painel fixo que ficou na foto (cabeceira-painel atrás da cama): o pedaço dele que a
    cama e as almofadas escondiam é parede da mesma madeira, não a parede lisa em volta que o LaMa copiaria.
    Por componente do painel, as linhas logo acima da base dele são espelhadas para baixo, coluna a coluna, sobre a
    máscara e só sobre parede (rótulo >= 0), até o piso. Altera ``plate`` no lugar; devolve os pixels trocados."""
    keep = panels & ~mask
    if keep.sum() < 50:
        return 0
    H, W = mask.shape
    lab, n = ndimage.label(_dil(keep, 4))
    done = 0
    for k in range(1, n + 1):
        comp = (lab == k) & keep
        cols = np.nonzero(comp.any(0))[0]
        if len(cols) < 8 or comp.sum() < 0.003 * H * W:  # pedaço pequeno do painel: o espelho viraria listras
            continue
        tops = np.array([np.nonzero(comp[:, c])[0].min() for c in cols])
        bots = np.array([np.nonzero(comp[:, c])[0].max() for c in cols])
        b0 = int(np.percentile(bots, 30))
        hb = int(np.clip(0.5 * (b0 - np.median(tops)), 4, 40))
        for c, bc in zip(cols, bots):
            y = int(bc) + 1
            while y < H and mask[y, c] and labels[y, c] >= 0:
                # acima da base comum (coluna em que o painel acabou antes): repete o último pixel do painel
                src = int(bc) if y <= b0 else b0 - ((y - b0 - 1) % hb)
                plate[y, c] = plate[src, c]
                done += 1
                y += 1
    return done


def keep_builtins_reason(inst_set, shape: tuple[int, int]) -> str:
    """Motivo para manter os planejados já na 1ª tentativa ("" = tenta a sala crua primeiro). Mede a área das
    peças que saem com e sem os planejados (``inst_set.removal(keep)``), antes de qualquer preenchimento."""
    def area(keep: bool) -> float:
        m = np.zeros(shape, bool)
        for i in inst_set.removal(keep):
            m |= np.asarray(i.mask, bool)
            a = getattr(i, "alpha", None)
            if a is not None:
                m |= np.asarray(a) > 0
        return float(m.mean())

    full, loose = area(False), area(True)
    built = full - loose
    if built < KEEP_BUILTINS_MIN:
        return ""
    if built >= KEEP_BUILTINS_AREA:
        return f"os planejados cobrem {100 * built:.0f}% da foto"
    if full >= KEEP_BUILTINS_TOTAL:
        return f"tirar tudo reconstruiria {100 * full:.0f}% da foto"
    return ""


# planejados/louças que ficaram na placa sem terem sido detectados como peça (o vaso que o detector não achou): a sala
# não está crua e o portão não pode aprovar (revisão rodada 2: banheiro aprovado com vaso, cuba e gabinete)
LEFTOVER_FIXTURE_MIN = 0.015     # fração da foto de um componente da classe para contar
LEFTOVER_FIXTURE_CLASSES = frozenset({
    "counter", "countertop", "kitchen island", "sink", "toilet", "tub", "bathtub", "shower", "stove", "oven",
    "refrigerator", "dishwasher", "microwave", "hood", "washer", "cabinet", "wardrobe", "chest of drawers",
    "buffet",
})


def leftover_fixtures(seg, mask: np.ndarray, keep: np.ndarray | None = None,
                      min_area: float = LEFTOVER_FIXTURE_MIN) -> list:
    """Planejados e louças (classes do ADE20K em :data:`LEFTOVER_FIXTURE_CLASSES`) que a segmentação da foto mostra
    FORA da máscara de remoção: ficaram na sala vazia. ``keep``: pixels que ficam de propósito (painel fixo de parede,
    planejados mantidos pelo portão, que já têm o próprio motivo). Componentes abaixo de ``min_area`` não contam."""
    lbl, names = getattr(seg, "labels", None), getattr(seg, "id2label", None) or {}
    if lbl is None or not names:
        return []
    H, W = mask.shape
    if lbl.shape != (H, W):
        lbl = cv2.resize(lbl.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    free = ~mask if keep is None else ~mask & ~keep
    out = []
    for cid, name in names.items():
        if str(name) not in LEFTOVER_FIXTURE_CLASSES:
            continue
        pix = (lbl == int(cid)) & free
        if pix.sum() < min_area * H * W:
            continue
        # sem as tiras finas que sobram em volta de uma peça removida (borda do OneFormer x máscara do SAM)
        pix = _dil(_ero(pix, max(2, int(0.005 * np.hypot(H, W)))), max(2, int(0.005 * np.hypot(H, W)))) & pix
        n, cc, st, _ = cv2.connectedComponentsWithStats(pix.astype(np.uint8), connectivity=8)
        big = st[1:, cv2.CC_STAT_AREA] >= min_area * H * W
        if big.any():
            out.append({"rotulo": str(name), "area_pct": round(100.0 * float(st[1:, cv2.CC_STAT_AREA][big].sum())
                                                               / (H * W), 2)})
    return out


def fail_quality(q: PlateQuality | None, reason: str) -> None:
    """Reprova a placa por uma regra do produto (fora da nota contínua): o motivo entra, ``ok`` cai e a nota vira
    0,6 × a nota sem teto, como toda reprovada."""
    if q is None or reason in q.reasons:
        return
    q.reasons.append(reason)
    q.ok = False
    raw = q.raw_score if q.raw_score is not None else q.score
    q.score = 0.6 * raw


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def build_plate_gated(photo: np.ndarray, inst_set, seg, room=None, depth=None, inpaint: InpaintFn | None = None,
                      redetect: Callable | None = None, seed: int = 0, max_side: int = MAX_SIDE,
                      alternatives: bool = True, on_attempt: Callable | None = None,
                      refine: Callable | None = None, geometry: bool | None = None,
                      ai_fill: Callable | None = None) -> PlateResult:
    """Portão de qualidade (§3.6 e PT-V3-8). Tentativas, parando na primeira aprovada:

    1. ``crua_por_plano``: soltos + planejados, LaMa por plano;
    2. ``planejados_mantidos`` (só se há planejados): só os soltos saem;
    3. ``lama_imagem`` (``alternatives``): LaMa direto na imagem, com o conjunto da melhor tentativa até aqui;
    4. ``mascara_ampla`` (``alternatives``): por plano com o núcleo dilatado :data:`GATE_WIDE_DILATE` vezes.

    A sala crua é sempre a 1ª tentativa (§3.6); :func:`keep_builtins_reason` só vai para o relatório como risco
    (``info["risco_planejados"]``), a menos que :data:`KEEP_BUILTINS_FIRST` esteja ligado. ``info["cobertura"]``
    diz quais planejados ficaram na foto e por quê. Erro no ``refine`` nunca derruba o trabalho: fica a LaMa.

    Com ``geometry`` (padrão :data:`GEOMETRY_FIRST`) e modelo da sala, as tentativas usam a estratégia "geometria"
    (:mod:`geofill`): ``crua_geometria`` → ``planejados_mantidos`` → ``mascara_ampla`` (todas pela geometria); as do
    LaMa só com :data:`GATE_LAMA_ALTERNATIVES`.

    Fica a melhor (aprovada > área no limite > menor falha dura > menos motivos > nota sem teto). Se nenhuma passa, devolve a melhor com aviso (a
    interface oferece gerar mesmo assim, o modo A ou outra foto); nunca entrega uma placa ruim em silêncio.
    ``info["tentativas"]`` registra cada tentativa (estratégia, nota, motivos, tempo). ``on_attempt(nome, placa)``
    é chamado depois de cada tentativa (progresso na interface).

    ``refine(placa, nome) -> placa`` (perfil "B com SDXL", PT-V3-4): estágio depois da LaMa em **cada** tentativa
    (refino SDXL só dentro da máscara, :func:`mobilia.gen.sdxl.refine_plate`); ele mesmo cai para a placa LaMa
    quando nenhuma semente é melhor. A tentativa registra a nota da LaMa e a do refino.

    ``ai_fill`` (sala vazia "ia", padrão do modo B quando o gerador de difusão está instalado): a 1ª tentativa é
    ``ia`` (sala crua pela geometria refinada pelo gerador, só dentro da máscara, com o portão :data:`LIMITS_IA`).
    Aprovada, para aí; reprovada (ou com erro no gerador), segue a sequência de sempre (geometria → LaMa) e no fim
    fica a melhor de todas as tentativas."""
    if inpaint is None:
        inpaint = LazyLama()
    attempts: list[dict] = []

    def run(name: str, keep: bool, strategy: str, dil: float) -> PlateResult:
        t0 = time.perf_counter()
        r = build_plate(photo, inst_set.removal(keep), seg, room, depth, inpaint, redetect, seed, max_side,
                        strategy=strategy, core_dilate=dil, ai_fill=ai_fill if strategy == "ia" else None)
        r.kept_builtins = keep
        r.info["estrategia_portao"] = name
        sd = None
        if refine is not None and strategy != "ia":
            q_lama = r.quality
            try:
                r = refine(r, name)
            except Exception as e:  # noqa: BLE001 - qualquer erro no refino (falta de VRAM na nota, driver...)
                # nunca derruba o trabalho: fica a placa LaMa desta tentativa, com o motivo no relatório
                _free_cuda()
                r.info["refino_sdxl"] = {"usado": False,
                                         "motivo": f"erro no refino SDXL ({type(e).__name__}: {str(e)[:200]}); "
                                                   "ficou a placa LaMa"}
                if r.quality is None:
                    r.quality = q_lama
            r.kept_builtins = keep
            r.info["estrategia_portao"] = name
            ri_ = r.info.get("refino_sdxl") or {}
            sd = {"usado": bool(ri_.get("usado")), "semente": ri_.get("semente"), "motivo": ri_.get("motivo"),
                  "nota_lama": None if q_lama is None else round(q_lama.score, 1), "tempo_s": ri_.get("tempo_s")}
        q = r.quality
        # regra do produto (§3.6): no modo B a sala vazia é CRUA. Planejado mantido ou planejado/louça que ficou sem
        # ser detectado reprova a placa, por melhor que seja a nota (antes o banheiro com vaso e cuba aprovava 82)
        if q is not None:
            kept_m = np.zeros(r.mask.shape, bool)
            for i in inst_set.instances:
                wp = getattr(i, "wall_panel", None)
                if wp is not None:
                    kept_m |= np.asarray(wp, bool)
                if keep and i.kind == "planejado":
                    kept_m |= np.asarray(i.mask, bool)
            if keep and any(i.kind == "planejado" for i in inst_set.instances):
                fail_quality(q, "planejados mantidos na foto: a sala vazia não está crua (o modo B tira armários, "
                                "bancadas e louças)")
            left = leftover_fixtures(seg, r.mask, kept_m)
            if left:
                r.info["planejados_que_ficaram"] = left
                fail_quality(q, "planejado/louça que ficou na sala vazia (não foi detectado como peça): "
                             + ", ".join(f"{x['rotulo']} {x['area_pct']:.1f}%" for x in left))
        attempts.append({"estrategia": name, "planejados_mantidos": keep, "metodo": strategy,
                         "dilatacao_nucleo": round(dil, 4),
                         "nota": None if q is None else round(q.score, 1),
                         "nota_sem_teto": None if q is None or q.raw_score is None else round(q.raw_score, 1),
                         "aprovada": None if q is None else q.ok, "motivos": [] if q is None else list(q.reasons),
                         "area_gerada_pct": round(100 * float(r.mask.mean()), 2),
                         "refino_sdxl": sd, "ia": r.info.get("ia"), "tempo_s": round(time.perf_counter() - t0, 2)})
        if on_attempt is not None:
            on_attempt(name, r)
        return r

    # as reprovadas têm a nota limitada a 59: compara pela aprovação, pela área gerada dentro do limite (a máscara
    # larga não pode "ganhar" trocando uma costura por uma sala quase toda inventada), nº de motivos e nota sem teto
    rank = lambda r: attempt_rank(r.quality)
    risk = keep_builtins_reason(inst_set, photo.shape[:2])
    keep_why = risk if KEEP_BUILTINS_FIRST else ""
    ai_best = None
    if ai_fill is not None:
        # uma tentativa por variante do preenchimento (``PlateFiller.variants``: refino da geometria, depois a
        # livre); a 1ª aprovada encerra; a melhor delas disputa no fim com as da geometria
        base_fill = ai_fill
        wide = CORE_DILATE * GATE_WIDE_DILATE
        variants = getattr(base_fill, "variants", None)
        if variants and "refino" in variants:
            # máscara larga primeiro (com o núcleo justo sobrava a borda fina do quadro/luminária: contorno escuro
            # no quarto); depois a justa (na sala com sofás a larga fez o gerador inventar tábuas onde a justa
            # saiu limpa); por último a livre (ruído puro, quando a geometria de partida é ruim)
            plan = [("ia", "refino", wide), ("ia_justa", "refino", CORE_DILATE)]
            plan += [(f"ia_{v}", v, wide) for v in variants if v != "refino"]
        else:
            plan = [("ia" if k == 0 else f"ia_{v}", v, wide) for k, v in enumerate(variants or (None,))]
        for name, var, dil in plan:
            ai_fill = base_fill if var is None else (lambda *a, _v=var: base_fill(*a, variant=_v))
            t_ai = time.perf_counter()
            try:
                r = run(name, False, "ia", dil)
            except Exception as e:  # noqa: BLE001 - gerador sem memória, pesos corrompidos...: segue a geometria
                _free_cuda()
                attempts.append({"estrategia": name, "planejados_mantidos": False, "metodo": "ia", "nota": None,
                                 "aprovada": False, "erro": f"{type(e).__name__}: {str(e)[:300]}",
                                 "tempo_s": round(time.perf_counter() - t_ai, 2)})
                break
            if ai_best is None or (r.quality is not None and ai_best.quality is not None
                                   and attempt_rank(r.quality) > attempt_rank(ai_best.quality)):
                ai_best = r
            if r.quality is None or r.quality.ok:
                return _gate_finish(r, inst_set, attempts, "", risk)
        ai_fill = base_fill
    geo = (GEOMETRY_FIRST if geometry is None else geometry) and room is not None \
        and getattr(room, "walls", None) is not None
    if geo:
        return _gated_geometry(run, rank, inst_set, alternatives, risk, attempts, ai_best)
    if keep_why:
        best = run("planejados_mantidos", True, "planos", CORE_DILATE)
        best.warnings.append(f"planejados mantidos na foto por padrão ({keep_why}): a parede atrás deles seria "
                             "toda inventada")
    else:
        best = run("crua_por_plano", False, "planos", CORE_DILATE)
    first_name = best.info["estrategia_portao"]
    if best.quality is not None and not best.quality.ok:
        first_reasons = list(best.quality.reasons)
        if not keep_why and any(i.kind == "planejado" for i in inst_set.instances):
            r2 = run("planejados_mantidos", True, "planos", CORE_DILATE)
            r2.warnings.append("sala crua reprovada (" + "; ".join(first_reasons) + "): planejados mantidos na foto")
            if rank(r2) >= rank(best):
                best = r2
        # área acima do limite: nenhuma alternativa passa (LaMa na imagem não muda a área e a máscara larga só a
        # aumenta), mesmo com outros motivos junto; não gasta ~2 preenchimentos + OneFormer à toa (foto lotada)
        area_blocked = any(r.startswith("área reconstruída") for r in best.quality.reasons)
        if alternatives and not best.quality.ok and not area_blocked:
            keep = best.kept_builtins
            for name, strategy, dil in (("lama_imagem", "imagem", CORE_DILATE),
                                        ("mascara_ampla", "planos", CORE_DILATE * GATE_WIDE_DILATE)):
                r = run(name, keep, strategy, dil)
                if rank(r) > rank(best):
                    r.warnings.extend(w for w in best.warnings if w not in r.warnings and "planejados" in w)
                    best = r
                if best.quality.ok:
                    break
        if best.quality.ok and best.info["estrategia_portao"] != first_name:
            best.warnings.append(f"sala vazia aprovada na tentativa \"{best.info['estrategia_portao']}\" "
                                 f"(a primeira reprovou: {'; '.join(first_reasons)})")
    return _gate_finish(best, inst_set, attempts, keep_why, risk, ai_best)


def _gated_geometry(run, rank, inst_set, alternatives: bool, risk: str, attempts: list,
                    ai_best: PlateResult | None = None) -> PlateResult:
    """Sequência do portão com a sala refeita pela geometria (ver :func:`build_plate_gated`)."""
    best = run("crua_geometria", False, "geometria", CORE_DILATE)
    first_reasons = [] if best.quality is None else list(best.quality.reasons)
    if best.quality is not None and not best.quality.ok:
        if any(i.kind == "planejado" for i in inst_set.instances):
            r2 = run("planejados_mantidos", True, "geometria", CORE_DILATE)
            r2.warnings.append("sala crua reprovada (" + "; ".join(first_reasons) + "): planejados mantidos na foto")
            if rank(r2) >= rank(best):
                best = r2
        area_blocked = any(r.startswith("área reconstruída") for r in best.quality.reasons)
        if alternatives and not best.quality.ok and not area_blocked:
            keep = best.kept_builtins
            alts = [("mascara_ampla", "geometria", CORE_DILATE * GATE_WIDE_DILATE)]
            if GATE_LAMA_ALTERNATIVES:
                alts += [("crua_por_plano", "planos", CORE_DILATE), ("lama_imagem", "imagem", CORE_DILATE)]
            for name, strategy, dil in alts:
                r = run(name, keep, strategy, dil)
                if rank(r) > rank(best):
                    r.warnings.extend(w for w in best.warnings if w not in r.warnings and "planejados" in w)
                    best = r
                if best.quality.ok:
                    break
        if best.quality.ok and best.info["estrategia_portao"] != "crua_geometria":
            best.warnings.append(f"sala vazia aprovada na tentativa \"{best.info['estrategia_portao']}\" "
                                 f"(a primeira reprovou: {'; '.join(first_reasons)})")
    return _gate_finish(best, inst_set, attempts, "", risk, ai_best)


def _gate_finish(best: PlateResult, inst_set, attempts: list, keep_why: str, risk: str,
                 ai_best: PlateResult | None = None) -> PlateResult:
    if ai_best is not None and ai_best is not best and ai_best.quality is not None and best.quality is not None \
            and attempt_rank(ai_best.quality) > attempt_rank(best.quality):
        best = ai_best                      # todas reprovaram e a "ia" ficou mais perto
    elif ai_best is not None and ai_best is not best and best.quality is not None and best.quality.ok:
        best.warnings.append("sala vazia por IA reprovada (" + "; ".join(ai_best.quality.reasons or ["erro"])
                             + f"): ficou a tentativa \"{best.info.get('estrategia_portao')}\"")
    best.info["tentativas"] = attempts
    best.info["planejados_por_padrao"] = keep_why or None
    best.info["risco_planejados"] = risk or None
    # cobertura do produto (§3.6: no modo B saem também os planejados): a placa entregue diz o que ficou na foto
    kept = [i.label for i in inst_set.instances if best.kept_builtins and i.kind == "planejado"]
    best.info["cobertura"] = {"planejados_mantidos": kept, "tudo_removido": not kept,
                              "motivo": None if not kept else (keep_why or "a sala crua reprovou no portão")}
    if kept and not any("planejados mantidos" in w for w in best.warnings):
        best.warnings.append("planejados mantidos na foto (a sala crua reprovou no portão): a sala vazia não está "
                             "totalmente crua")
    if best.quality is not None and not best.quality.ok:
        best.warnings.append("a sala vazia ficou abaixo da qualidade mínima (" + "; ".join(best.quality.reasons)
                             + "): use o modo A ou envie outra foto")
    return best
