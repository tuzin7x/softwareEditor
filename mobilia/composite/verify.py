"""Verificação independente da regra da placa (PT5).

Fora da **máscara de influência** (os pixels a menos de um raio de algum pixel de móvel), o quadro
composto tem de ser byte a byte a foto: ``np.array_equal(out8[~mascara], placa8[~mascara])``.

A máscara é calculada aqui, a partir só do alfa das camadas 3D, sem nada da lógica interna do
compositor (recorte, pesos de sombra, cópia dos bytes): um erro no recorte, na colagem ou na
conta da sombra que encoste em pixels fora dela faz a verificação falhar.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# lado do bloco da máscara: ela é montada numa grade de blocos (barata) e é um SUPERCONJUNTO da
# vizinhança euclidiana exata (nunca acusa um pixel que a composição podia mudar)
TILE_PX = 4
# folga além do raio de influência: antisserrilhado/desfoque de nitidez do alfa (≤ 4 px) e margem
MARGIN_PX = 8


def influence_mask(alpha: np.ndarray | None, origin: tuple[int, int], shape: tuple[int, int],
                   radius_px: float, tile: int = TILE_PX) -> np.ndarray:
    """Máscara (H, W) que contém todo pixel a menos de ``radius_px`` de algum pixel com alfa > 0.

    ``alpha`` (h×w) está na posição ``origin`` = (x, y) do quadro ``shape`` = (H, W). Um bloco da
    grade entra quando a distância euclidiana entre blocos até algum bloco com CG é ≤ raio/bloco
    + 1,5 (pixels a distância d ficam a menos de d/bloco + √2 blocos): superconjunto garantido."""
    H, W = int(shape[0]), int(shape[1])
    nty, ntx = -(-H // tile), -(-W // tile)
    tiles = np.zeros((nty, ntx), bool)
    if alpha is not None:
        a = np.asarray(alpha) > 0
        if a.any():
            ox, oy = int(origin[0]), int(origin[1])
            h, w = a.shape
            py, px = oy % tile, ox % tile                     # alinha o recorte à grade do quadro
            th, tw = -(-(py + h) // tile), -(-(px + w) // tile)
            buf = np.zeros((th * tile, tw * tile), bool)
            buf[py:py + h, px:px + w] = a
            blk = buf.reshape(th, tile, tw, tile).any(axis=(1, 3))
            ty, tx = oy // tile, ox // tile
            tiles[ty:ty + th, tx:tx + tw] |= blk[:nty - ty, :ntx - tx]
            tiles = ndimage.distance_transform_edt(~tiles) <= float(radius_px) / tile + 1.5
    return tiles[np.arange(H) // tile][:, np.arange(W) // tile]


def outside_mask_equal(out8: np.ndarray, plate8: np.ndarray, mask: np.ndarray) -> tuple[bool, int, int]:
    """(``np.array_equal(out8[~mask], plate8[~mask])``, pixels divergentes, maior diferença).

    Mesma resposta do ``array_equal`` sobre o complemento da máscara, comparando o quadro inteiro
    canal a canal (sem copiar os pixels de fora da máscara)."""
    if out8.shape != plate8.shape:
        return False, int(mask.size), 255
    neq = out8 != plate8
    changed = neq[..., 0]
    for i in range(1, neq.shape[-1]):
        changed = changed | neq[..., i]
    bad = changed & ~mask
    if not bad.any():
        return True, 0, 0
    d = np.abs(out8[bad].astype(np.int16) - plate8[bad].astype(np.int16))
    return False, int(np.count_nonzero(bad)), int(d.max())


# ============================================================================================================
# Modo B (PLANO_V3 §3.5, PT-V3-7): sala vazia → peças reais da foto → foto exata
# ============================================================================================================
class ModeBVerificationError(RuntimeError):
    """Uma regra do modo B falhou: o trabalho para sem MP4."""


def moving_boxes(stack, placements, shape: tuple[int, int]) -> np.ndarray:
    """Máscara (H, W) das caixas das peças EM MOVIMENTO no quadro (deslocadas e/ou escaladas). Calculada só da
    geometria (caixa de repouso + ``dx``/``dy`` + escala em torno da âncora, com 1 px de folga), sem o compositor.
    Peça em repouso não entra: ela só pode mudar pixels dentro da máscara de remoção."""
    from ..layers2d.compose import moving, scaled_box

    H, W = int(shape[0]), int(shape[1])
    m = np.zeros((H, W), bool)
    if not placements:
        return m
    for i, q in placements.items():
        if not moving(q):
            continue
        x0, y0, x1, y1 = scaled_box(stack.layers[i].box, q)
        x0, x1 = max(0, x0 + q.dx), min(W, x1 + q.dx)
        y0, y1 = max(0, y0 + q.dy), min(H, y1 + q.dy)
        if x0 < x1 and y0 < y1:
            m[y0:y1, x0:x1] = True
    return m


class ModeBVerifier:
    """Verificação em fluxo (quadro a quadro, sem guardar o vídeo) das regras do §3.5:

    1. quadro 0 ``array_equal`` à placa vazia (``stack.plate``);
    2. último quadro ``array_equal`` à foto (na resolução do vídeo);
    3. em TODO quadro, fora de (máscara de remoção ∪ caixa das peças em movimento), quadro == foto;
    4. placa == foto fora da máscara de remoção;
    5. pixel oculto visível = 0 (medido por ``choreo.verify``; o resultado entra em :meth:`finish`), também na
       verificação independente do grafo de oclusão (5b, ``choreo.verify_independent``);
    6. anti-morph: fora de (região dinâmica ∪ caixas em movimento do quadro e do anterior) nada muda entre
       quadros consecutivos;
    7. o relatório (:meth:`finish`) registra os números para o ``report.json``.

    Com ``strict`` (padrão) a 1ª falha levanta :class:`ModeBVerificationError` (o chamador aborta o MP4)."""

    def __init__(self, photo: np.ndarray, stack, removal_mask: np.ndarray, n_frames: int, strict: bool = True):
        self.photo = photo
        self.stack = stack
        self.mask = np.asarray(removal_mask, bool)
        self.n_frames = int(n_frames)
        self.strict = strict
        H, W = stack.photo_shape
        self.shape = (H, W)
        self.dynamic = stack.dynamic if stack.dynamic is not None else self.mask
        self.failures: list[str] = []
        self.n = 0
        self.max_outside = 0
        self.morph_px = 0
        self.max_moving_pct = 0.0
        self._prev = None
        self._prev_mov = None
        if photo.shape[:2] != (H, W):
            self._fail(f"foto {photo.shape[:2]} e camadas {(H, W)} com tamanhos diferentes")
        ok4, n4, _ = outside_mask_equal(stack.plate, photo, self.mask)
        self.plate_outside_px = n4
        if not ok4:
            self._fail(f"regra 4: a placa difere da foto em {n4} pixels fora da máscara de remoção")

    def _fail(self, msg: str) -> None:
        self.failures.append(msg)
        if self.strict:
            raise ModeBVerificationError(msg)

    def check(self, frame: np.ndarray, placements=None) -> None:
        """Verifica o próximo quadro (na ordem do vídeo). ``placements``: as do quadro (None = todas em repouso
        ou nenhuma camada, p. ex. dissolução e hold)."""
        f = self.n
        if frame.shape != self.photo.shape or frame.dtype != np.uint8:
            self._fail(f"quadro {f}: forma/tipo {frame.shape}/{frame.dtype} inesperados")
            self.n += 1
            return
        if f == 0 and not np.array_equal(frame, self.stack.plate):
            self._fail("regra 1: o quadro 0 não é a placa vazia byte a byte")
        mov = moving_boxes(self.stack, placements, self.shape)
        self.max_moving_pct = max(self.max_moving_pct, 100.0 * float(mov.mean()) if mov.any() else 0.0)
        # o hold final devolve o próprio array da foto e quadros repetidos o mesmo array: comparação trivial
        ok, nbad, dmax = (True, 0, 0) if frame is self.photo else outside_mask_equal(frame, self.photo,
                                                                                     self.mask | mov)
        if not ok:
            self.max_outside = max(self.max_outside, dmax)
            self._fail(f"regra 3: quadro {f} difere da foto em {nbad} pixels fora da máscara ∪ caixa móvel "
                       f"(máx {dmax} níveis)")
        if self._prev is not None and frame is not self._prev:
            allow = self.dynamic | mov | self._prev_mov
            ok6, n6, _ = outside_mask_equal(frame, self._prev, allow)
            if not ok6:
                self.morph_px += n6
                self._fail(f"regra 6: {n6} pixels fora da região dinâmica mudaram do quadro {f - 1} para o {f}")
        if f == self.n_frames - 1 and not np.array_equal(frame, self.photo):
            self._fail("regra 2: o último quadro não é a foto byte a byte")
        self._prev, self._prev_mov = frame, mov
        self.n += 1

    def finish(self, choreo_stats: dict | None = None) -> dict:
        if self.n != self.n_frames:
            self._fail(f"{self.n} quadros verificados, esperados {self.n_frames}")
        hidden = hidden_ind = None
        if choreo_stats is not None:
            hidden = int(choreo_stats.get("oculto_visivel_px", 0))
            if hidden:
                self._fail(f"regra 5: {hidden} pixels de parte oculta ficaram visíveis")
            # 5b: a mesma regra medida SEM o grafo de oclusão da coreografia (``choreo.verify_independent``)
            if "oculto_independente_px" in choreo_stats:
                hidden_ind = int(choreo_stats.get("oculto_independente_px", 0))
                if hidden_ind:
                    pares = ", ".join(f"{d['peca']} antes de {d['coberta_por']}"
                                      for d in choreo_stats.get("oculto_independente_pares", [])[:3])
                    self._fail(f"regra 5b: {hidden_ind} pixels de corte à vista pela verificação independente"
                               + (f" ({pares})" if pares else ""))
        H, W = self.shape
        return {"verificacao_modo_b": {
            "ok": not self.failures, "quadros": self.n,
            "quadro0_igual_placa": not any(s.startswith("regra 1") for s in self.failures),
            "ultimo_igual_foto": not any(s.startswith("regra 2") for s in self.failures),
            "placa_fora_mascara_px": int(self.plate_outside_px),
            "fora_mascara_max_niveis": int(self.max_outside),
            "morph_px": int(self.morph_px), "oculto_visivel_px": hidden,
            "oculto_independente_px": hidden_ind,
            "mascara_remocao_pct": round(100.0 * float(self.mask.sum()) / (H * W), 3),
            "caixa_movel_max_pct": round(self.max_moving_pct, 3),
            "falhas": list(self.failures[:20])}}
