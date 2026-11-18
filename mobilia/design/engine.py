"""Gerenciador do pipeline de difusão do redesenho (um único UNet para todas as funções).

Um só ``StableDiffusionXLControlNetInpaintPipeline`` (RealVisXL V5.0 Lightning + ControlNet depth small + VAE
fp16-fix, pesos fixados com SHA-256 em ``analysis/weights.py``) cobre as três formas de geração:

* **img2img** ("Mudar estilo", "Reforma", "Redesign simples", "Fachada", "Evento"): máscara toda branca (ou tudo
  menos o miolo das aberturas travadas) e ``strength`` < 1 — dentro da máscara é exatamente o img2img do SDXL; fora
  dela os latentes da foto são repostos a cada passo, então o gerador "vê" as janelas fixas e desenha em volta;
* **inpaint** ("Editar área", "Mobiliar"): máscara do usuário / do volume piso-paredes, ``strength`` ~1;
* **txt2img** ("Texto para ambiente"): máscara toda branca, ``strength`` = 1 (parte de ruído puro) e controle 0.

Os codificadores de texto ficam fora da GPU: os embeddings são calculados na CPU (com prompts longos em blocos de
75 tokens, concatenados — o CLIP sozinho cortaria tudo depois de 77) e guardados em cache no disco
(``<dados>/models/sdxl_prompts``). O pipeline carregado fica em cache no processo (``get_engine``): várias
variações e funções seguidas não recarregam os 6 GB de pesos.

GPU de 8 GB (RTX 5070 Laptop): UNet + ControlNet + VAE em fp16 cabem com folga sem os codificadores de texto
(~5,7 GB + ativações a 1024 px com SDPA); abaixo de ``OFFLOAD_BELOW_GB`` de VRAM usa ``enable_model_cpu_offload``.
"""
from __future__ import annotations

import gc
import time
from typing import Callable, Protocol

import numpy as np

from .presets import DesignPrompt

OFFLOAD_BELOW_GB = 7.0
MAX_PROMPT_CHUNKS = 3


class Generator(Protocol):
    """O que as funções de ``api.py`` precisam de um gerador (o real ou um falso nos testes)."""

    device: str

    def generate(self, image: np.ndarray, mask: np.ndarray, control: np.ndarray | None, prompt: DesignPrompt,
                 seed: int, strength: float, steps: int, guidance: float, control_scale: float,
                 control_end: float, callback: Callable[[int, int], None] | None = None) -> np.ndarray: ...


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - sem torch: só os testes com gerador falso
        return "cpu"


def _vram_gb() -> float:
    try:
        import torch

        return torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    except Exception:  # noqa: BLE001
        return 0.0


class DesignEngine:
    """Pipeline SDXL real. Carrega os pesos na primeira geração (``load``) e mantém tudo na memória."""

    def __init__(self, device: str = "auto", offload: bool | str = "auto", offline: bool = True, dtype=None):
        self.device = pick_device(device)
        self.offline = offline
        self._dtype = dtype
        if offload == "auto":
            offload = self.device.startswith("cuda") and _vram_gb() < OFFLOAD_BELOW_GB
        self.offload = bool(offload and self.device.startswith("cuda"))
        self.pipe = None
        self._embeds: dict[str, dict] = {}
        self.load_s = 0.0
        self.calls: list[dict] = []

    # ------------------------------------------------------------------ carga
    def load(self) -> None:
        if self.pipe is not None:
            return
        import torch
        from diffusers import (AutoencoderKL, ControlNetModel, EulerDiscreteScheduler,
                               StableDiffusionXLControlNetInpaintPipeline, UNet2DConditionModel)

        from ..analysis import weights as W

        t0 = time.perf_counter()
        self.dtype = self._dtype or (torch.float16 if self.device.startswith("cuda") else torch.bfloat16)
        dirs = W.sdxl_dirs(offline=self.offline)
        gen = dirs["gerador"]
        unet = UNet2DConditionModel.from_pretrained(gen, subfolder="unet", variant="fp16", torch_dtype=self.dtype)
        ctl = ControlNetModel.from_pretrained(dirs["controlnet"], variant="fp16", torch_dtype=self.dtype)
        vae = AutoencoderKL.from_pretrained(dirs["vae"], torch_dtype=self.dtype)
        sch = EulerDiscreteScheduler.from_pretrained(gen, subfolder="scheduler", timestep_spacing="trailing")
        pipe = StableDiffusionXLControlNetInpaintPipeline(
            vae=vae, text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None, unet=unet,
            controlnet=ctl, scheduler=sch, add_watermarker=False)
        pipe.set_progress_bar_config(disable=True)
        if self.device.startswith("cuda"):
            try:
                pipe.enable_vae_tiling()     # o pico da decodificação a 1024 px não soma ao do UNet
            except Exception:  # noqa: BLE001
                pass
        if self.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        self.pipe = pipe
        self.load_s = time.perf_counter() - t0

    def release(self) -> None:
        self.pipe = None
        self._embeds.clear()
        gc.collect()
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ prompts
    def _cache_path(self, prompt: DesignPrompt):
        from ..analysis.weights import models_dir

        return models_dir() / "sdxl_prompts" / f"d{MAX_PROMPT_CHUNKS}_{prompt.key()}.pt"

    def embeddings(self, prompts: list[DesignPrompt]) -> list[dict]:
        """Embeddings de vários prompts (cache em memória e em disco; os codificadores de texto só são carregados
        se algum ainda não tiver cache, uma vez para todos, e liberados em seguida)."""
        import torch

        out: dict[str, dict] = {}
        missing: list[DesignPrompt] = []
        for p in prompts:
            k = p.key()
            if k in self._embeds:
                out[k] = self._embeds[k]
                continue
            path = self._cache_path(p)
            if path.exists():
                try:
                    out[k] = torch.load(path, map_location="cpu", weights_only=True)
                    continue
                except Exception:  # noqa: BLE001 - cache corrompido: recalcula
                    pass
            if all(m.key() != k for m in missing):
                missing.append(p)
        if missing:
            enc = encode_prompts(missing, offline=self.offline)
            for p, e in zip(missing, enc):
                out[p.key()] = e
                try:
                    path = self._cache_path(p)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(e, path)
                except OSError:
                    pass
        self._embeds.update(out)
        return [out[p.key()] for p in prompts]

    # ------------------------------------------------------------------ geração
    def generate(self, image: np.ndarray, mask: np.ndarray, control: np.ndarray | None, prompt: DesignPrompt,
                 seed: int, strength: float, steps: int, guidance: float, control_scale: float,
                 control_end: float, callback: Callable[[int, int], None] | None = None) -> np.ndarray:
        import torch
        from PIL import Image

        self.load()
        e = self.embeddings([prompt])[0]
        dev = "cuda" if self.offload else self.device
        dt = self.dtype
        h, w = image.shape[:2]
        steps, strength, n_eff = plan_steps(steps, strength)
        if control is None or control_scale <= 0:
            control = np.zeros((h, w), np.float32)
            control_scale, control_end = 0.0, 1.0
        ctrl = np.repeat((np.clip(control, 0, 1) * 255).round().astype(np.uint8)[..., None], 3, axis=2)
        m8, soft = mask_u8(mask)

        def _cb(pipe, i, t, kw):
            if callback is not None:
                callback(i + 1, n_eff)
            return kw

        t0 = time.perf_counter()
        hard_proc = self.pipe.mask_processor
        if soft:
            # máscara suave: o diffusers binariza por padrão; sem isso a mistura dos latentes a cada passo vira um
            # degradê (a área nova "funde" com a foto em vez de terminar num contorno visível)
            from diffusers.image_processor import VaeImageProcessor

            self.pipe.mask_processor = VaeImageProcessor(vae_scale_factor=self.pipe.vae_scale_factor,
                                                         do_normalize=False, do_binarize=False,
                                                         do_convert_grayscale=True)
        try:
            with torch.inference_mode():
                res = self.pipe(
                    prompt_embeds=e["prompt"].to(dev, dt), negative_prompt_embeds=e["negative"].to(dev, dt),
                    pooled_prompt_embeds=e["pooled"].to(dev, dt),
                    negative_pooled_prompt_embeds=e["negative_pooled"].to(dev, dt),
                    image=Image.fromarray(np.ascontiguousarray(image)), mask_image=Image.fromarray(m8),
                    control_image=Image.fromarray(ctrl), height=h, width=w, strength=min(1.0, strength + 1e-4),
                    num_inference_steps=steps, guidance_scale=float(guidance),
                    controlnet_conditioning_scale=float(control_scale), control_guidance_start=0.0,
                    control_guidance_end=float(max(control_end, 0.05)),
                    generator=torch.Generator(device="cpu").manual_seed(int(seed)), output_type="np",
                    callback_on_step_end=_cb)
        finally:
            self.pipe.mask_processor = hard_proc
        out = res.images[0]
        dt_s = time.perf_counter() - t0
        self.calls.append({"tamanho": [w, h], "passos_efetivos": n_eff, "s": round(dt_s, 2),
                           "s_por_passo": round(dt_s / n_eff, 3)})
        return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


def mask_u8(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    """Máscara para o pipeline (uint8 0–255) e se ela é suave: ``bool`` = binária (o padrão); ``float`` em [0, 1] =
    suave (peso da geração por pixel; a mistura com os latentes da foto a cada passo segue esse peso)."""
    m = np.asarray(mask)
    if m.dtype == bool:
        return m.astype(np.uint8) * 255, False
    m = np.clip(m.astype(np.float32), 0, 1)
    soft = bool(((m > 0.02) & (m < 0.98)).any())
    return (m * 255 + 0.5).astype(np.uint8), soft


def plan_steps(steps: int, strength: float) -> tuple[int, float, int]:
    """Passos do agendador para uma força pedida: ``(total, força efetiva, passos efetivos)``.

    O diffusers roda ``int(total × força)`` passos e começa no timestep correspondente; com poucos passos (Lightning)
    isso quantiza a força: com 6 passos, 0,68, 0,72 e 0,82 começam todos no mesmo ruído (0,67). Aqui o número de
    passos efetivos é ``round(steps × força)`` (ou um a mais) e o total é escolhido para a força efetiva ficar o
    mais perto possível da pedida."""
    s = float(np.clip(strength, 0.05, 1.0))
    if s >= 0.97:          # com poucos passos não dá para separar 0,97 de 1: ruído puro
        n = max(1, int(steps))
        return n, 1.0, n
    best = None
    n0 = max(1, int(round(steps * s)))
    for n in (n0, n0 + 1):
        for tot in {max(n + 1, int(np.floor(n / s))), max(n + 1, int(np.ceil(n / s)))}:
            err = abs(n / tot - s) + (0.02 if n > n0 else 0.0)
            if best is None or err < best[0]:
                best = (err, tot, n)
    _, tot, n = best
    return tot, n / tot, n


# ====================================================================================== prompts longos
def _chunk_ids(tokenizer, text: str, max_chunks: int = MAX_PROMPT_CHUNKS) -> list[list[int]]:
    """Ids em blocos de 77 (BOS + até 75 tokens + EOS + preenchimento), no máximo ``max_chunks`` blocos."""
    ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    body = 75
    chunks = [ids[i:i + body] for i in range(0, max(len(ids), 1), body)][:max_chunks] or [[]]
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    out = []
    for c in chunks:
        row = [bos] + list(c) + [eos]
        out.append(row + [pad] * (77 - len(row)))
    return out


def _encode_text(te1, te2, tk1, tk2, text: str, n_chunks: int | None = None):
    import torch

    c1, c2 = _chunk_ids(tk1, text), _chunk_ids(tk2, text)
    n = max(len(c1), len(c2)) if n_chunks is None else n_chunks
    empty1, empty2 = _chunk_ids(tk1, "")[0], _chunk_ids(tk2, "")[0]
    c1 = (c1 + [empty1] * n)[:n]
    c2 = (c2 + [empty2] * n)[:n]
    embs, pooled = [], None
    for j in range(n):
        i1 = torch.tensor([c1[j]])
        i2 = torch.tensor([c2[j]])
        o1 = te1(i1, output_hidden_states=True)
        o2 = te2(i2, output_hidden_states=True)
        embs.append(torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1))
        if j == 0:
            pooled = o2[0]
    return torch.cat(embs, dim=1), pooled, n


def encode_prompts(prompts: list[DesignPrompt], offline: bool = True) -> list[dict]:
    """Calcula os embeddings (CPU, fp32) de uma lista de prompts com os codificadores de texto do SDXL, carregados
    uma vez e liberados no fim."""
    import torch
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    from ..analysis import weights as W

    tdir = W.sdxl_text_dir(offline=offline)
    te1 = CLIPTextModel.from_pretrained(tdir, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float32)
    te2 = CLIPTextModelWithProjection.from_pretrained(tdir, subfolder="text_encoder_2", variant="fp16",
                                                      torch_dtype=torch.float32)
    tk1 = CLIPTokenizer.from_pretrained(tdir, subfolder="tokenizer")
    tk2 = CLIPTokenizer.from_pretrained(tdir, subfolder="tokenizer_2")
    out = []
    try:
        with torch.inference_mode():
            for p in prompts:
                pe, pp, n = _encode_text(te1, te2, tk1, tk2, p.positive)
                ne, npool, _ = _encode_text(te1, te2, tk1, tk2, p.negative, n_chunks=None)
                m = max(pe.shape[1], ne.shape[1]) // 77
                if pe.shape[1] != ne.shape[1]:      # mesmo número de blocos nos dois (o CFG concatena os lotes)
                    pe, pp, _ = _encode_text(te1, te2, tk1, tk2, p.positive, n_chunks=m)
                    ne, npool, _ = _encode_text(te1, te2, tk1, tk2, p.negative, n_chunks=m)
                out.append({k: v.detach().to("cpu", torch.float32).contiguous() for k, v in
                            {"prompt": pe, "negative": ne, "pooled": pp, "negative_pooled": npool}.items()})
    finally:
        del te1, te2
        gc.collect()
    return out


# ====================================================================================== cache do processo
_ENGINES: dict[tuple, DesignEngine] = {}


def get_engine(device: str = "auto", offload: bool | str = "auto", offline: bool = True) -> DesignEngine:
    """Motor em cache no processo (uma carga de pesos por processo de trabalho)."""
    key = (pick_device(device), str(offload), bool(offline))
    eng = _ENGINES.get(key)
    if eng is None:
        eng = _ENGINES[key] = DesignEngine(key[0], offload, offline)
    return eng


def release_engines() -> None:
    for e in _ENGINES.values():
        e.release()
    _ENGINES.clear()


__all__ = ["Generator", "DesignEngine", "get_engine", "release_engines", "encode_prompts", "pick_device",
           "OFFLOAD_BELOW_GB", "plan_steps", "mask_u8"]
