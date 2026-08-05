"""Krea 2 pipeline — single-stream MMDiT architecture (Krea AI).

Krea 2 uses a 12.9B-parameter Diffusion Transformer with a Qwen3-VL text
encoder and Qwen-Image VAE. The Turbo (distilled) variant generates in 8
steps with guidance disabled (guidance_scale=0.0).

Supports two loading modes:
  - Diffusers directory format (from_pretrained)
  - Single-file .safetensors (transformer only; text encoder + VAE are
    auto-downloaded and cached on first use)

Negative prompts and Compel prompt weighting ([token:weight]) are NOT
supported — they will be silently ignored.

Implements the same interface as ImageGenerator in pipeline.py so app.py
can swap between architectures transparently.
"""

import gc
import warnings

import torch
from diffusers import (
    Krea2Pipeline,
    Krea2Transformer2DModel,
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
)

import config

SCHEDULERS = {
    "Euler": (FlowMatchEulerDiscreteScheduler, {}),
}

SCHEDULER_NAMES = list(SCHEDULERS.keys())

_MODEL_DIR = config.ARCH_MODEL_DIRS["Krea 2"]
_LORA_DIR = config.ARCH_LORA_DIRS["Krea 2"]
_ENCODERS_DIR = _MODEL_DIR / "_encoders"  # cached text encoder + VAE (auto-downloaded)

# HuggingFace repos for components (used by single-file loading path)
_TEXT_ENCODER_REPO = "Qwen/Qwen3-VL-4B-Instruct"
_VAE_REPO = "Qwen/Qwen-Image"

# Layer indices from the 36-layer Qwen3-VL model to tap for text conditioning
_TEXT_ENCODER_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


# ── Single-file component loaders ────────────────────────────────────

def _load_text_encoder(progress_callback=None):
    """Load the Qwen3-VL text encoder, downloading from HuggingFace on first use."""
    from transformers import Qwen3VLForConditionalGeneration

    cache_dir = _ENCODERS_DIR / "text_encoder"
    if cache_dir.exists() and any(cache_dir.iterdir()):
        if progress_callback:
            progress_callback("Loading Qwen3-VL text encoder (cached)...")
        return Qwen3VLForConditionalGeneration.from_pretrained(
            str(cache_dir), torch_dtype=torch.bfloat16, local_files_only=True,
        )

    if progress_callback:
        progress_callback(
            f"Downloading Qwen3-VL text encoder from {_TEXT_ENCODER_REPO} "
            "(first time only, ~9 GB)..."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        _TEXT_ENCODER_REPO, torch_dtype=torch.bfloat16,
    )
    model.save_pretrained(str(cache_dir))
    return model


def _load_tokenizer(progress_callback=None):
    """Load the Qwen3-VL tokenizer, downloading from HuggingFace on first use."""
    from transformers import AutoTokenizer

    cache_dir = _ENCODERS_DIR / "tokenizer"
    if cache_dir.exists() and any(cache_dir.iterdir()):
        return AutoTokenizer.from_pretrained(str(cache_dir), local_files_only=True)

    if progress_callback:
        progress_callback("Downloading Qwen3-VL tokenizer (first time only)...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(_TEXT_ENCODER_REPO)
    tokenizer.save_pretrained(str(cache_dir))
    return tokenizer


def _load_vae(progress_callback=None):
    """Load the Qwen-Image VAE, downloading from HuggingFace on first use."""
    cache_dir = _ENCODERS_DIR / "vae"
    if cache_dir.exists() and any(cache_dir.iterdir()):
        if progress_callback:
            progress_callback("Loading Qwen-Image VAE (cached)...")
        return AutoencoderKLQwenImage.from_pretrained(
            str(cache_dir), torch_dtype=torch.bfloat16, local_files_only=True,
        )

    if progress_callback:
        progress_callback(
            f"Downloading Qwen-Image VAE from {_VAE_REPO} "
            "(first time only, ~254 MB)..."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLQwenImage.from_pretrained(
        _VAE_REPO, subfolder="vae", torch_dtype=torch.bfloat16,
    )
    vae.save_pretrained(str(cache_dir))
    return vae


def _convert_comfyui_keys(state_dict):
    """Convert ComfyUI/Comfy-Org key names to diffusers format.

    ComfyUI checkpoints use the original model key names while diffusers
    uses its own naming convention. This handles both the main transformer
    blocks and the text fusion sub-network.
    """
    # Check if conversion is needed — if any key already matches diffusers
    # format (e.g. starts with "transformer_blocks."), skip conversion.
    if any(k.startswith("transformer_blocks.") for k in state_dict):
        return state_dict

    # Check if this looks like a ComfyUI checkpoint
    if not any(k.startswith("blocks.") for k in state_dict):
        return state_dict

    # Per-layer key suffix renames (shared by main blocks, txtfusion, refiners)
    _BLOCK_RENAMES = {
        "attn.gate.weight": "attn.to_gate.weight",
        "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
        "attn.qknorm.knorm.scale": "attn.norm_k.weight",
        "attn.wq.weight": "attn.to_q.weight",
        "attn.wk.weight": "attn.to_k.weight",
        "attn.wv.weight": "attn.to_v.weight",
        "attn.wo.weight": "attn.to_out.0.weight",
        "mlp.gate.weight": "ff.gate.weight",
        "mlp.up.weight": "ff.up.weight",
        "mlp.down.weight": "ff.down.weight",
        "prenorm.scale": "norm1.weight",
        "postnorm.scale": "norm2.weight",
    }

    # Top-level (non-block) renames
    _TOP_RENAMES = {
        "first.weight": "img_in.weight",
        "first.bias": "img_in.bias",
        "last.linear.weight": "final_layer.linear.weight",
        "last.linear.bias": "final_layer.linear.bias",
        "last.norm.scale": "final_layer.norm.weight",
        "last.modulation.lin": "final_layer.scale_shift_table",
        "tmlp.0.weight": "time_embed.linear_1.weight",
        "tmlp.0.bias": "time_embed.linear_1.bias",
        "tmlp.2.weight": "time_embed.linear_2.weight",
        "tmlp.2.bias": "time_embed.linear_2.bias",
        "tproj.1.weight": "time_mod_proj.weight",
        "tproj.1.bias": "time_mod_proj.bias",
        "txtmlp.0.scale": "txt_in.norm.weight",
        "txtmlp.1.weight": "txt_in.linear_1.weight",
        "txtmlp.1.bias": "txt_in.linear_1.bias",
        "txtmlp.3.weight": "txt_in.linear_2.weight",
        "txtmlp.3.bias": "txt_in.linear_2.bias",
        "txtfusion.projector.weight": "text_fusion.projector.weight",
    }

    converted = {}
    for key, value in state_dict.items():
        new_key = None

        # Top-level renames
        if key in _TOP_RENAMES:
            new_key = _TOP_RENAMES[key]

        # Main transformer blocks: blocks.N.suffix → transformer_blocks.N.suffix
        elif key.startswith("blocks."):
            parts = key.split(".", 2)  # ["blocks", "N", "suffix"]
            suffix = parts[2]
            if suffix == "mod.lin":
                new_key = f"transformer_blocks.{parts[1]}.scale_shift_table"
                # Stored flat [36864] in ComfyUI, model expects [6, 6144]
                if value.dim() == 1:
                    value = value.view(6, -1)
            elif suffix in _BLOCK_RENAMES:
                new_key = f"transformer_blocks.{parts[1]}.{_BLOCK_RENAMES[suffix]}"

        # Text fusion layerwise blocks
        elif key.startswith("txtfusion.layerwise_blocks."):
            parts = key.split(".", 3)  # ["txtfusion", "layerwise_blocks", "N", "suffix"]
            suffix = parts[3]
            if suffix in _BLOCK_RENAMES:
                new_key = f"text_fusion.layerwise_blocks.{parts[2]}.{_BLOCK_RENAMES[suffix]}"

        # Text fusion refiner blocks
        elif key.startswith("txtfusion.refiner_blocks."):
            parts = key.split(".", 3)  # ["txtfusion", "refiner_blocks", "N", "suffix"]
            suffix = parts[3]
            if suffix in _BLOCK_RENAMES:
                new_key = f"text_fusion.refiner_blocks.{parts[2]}.{_BLOCK_RENAMES[suffix]}"

        if new_key is not None:
            converted[new_key] = value
        else:
            # Keep unknown keys as-is (will be ignored by strict=False)
            converted[key] = value

    return converted


def _load_single_file_transformer(checkpoint_path, progress_callback=None):
    """Load a Krea 2 transformer from a single safetensors file.

    Supports multiple checkpoint formats:
    - **bf16 / fp32**: Official krea/Krea-2-Turbo ``turbo.safetensors`` or
      Comfy-Org ``krea2_turbo_bf16.safetensors``. Loaded directly.
    - **fp16**: Community re-quantised checkpoints. Cast to bf16 on load.
    - **fp8 (no scales)**: e.g. AlperKTS/Krea2_FP8. Plain float8_e4m3fn
      weights with no ``weight_scale`` companions. Cast to bf16.
    - **fp8 scaled**: e.g. Comfy-Org ``krea2_turbo_fp8_scaled.safetensors``.
      Each float8_e4m3fn weight has a ``<name>_scale`` float32 scalar.
      Dequantised as ``bf16 = fp8.to(bf16) * scale``.
    - **int8 / other quantised**: Not supported (require specialised
      ComfyUI nodes). Raises a clear error.
    """
    from safetensors.torch import load_file

    if progress_callback:
        progress_callback(f"Loading transformer from {checkpoint_path.name}...")

    state_dict = load_file(str(checkpoint_path))

    # ── Detect checkpoint format ──────────────────────────────────────
    dtypes_present = {v.dtype for v in state_dict.values()}
    has_fp8 = torch.float8_e4m3fn in dtypes_present
    has_int8 = torch.int8 in dtypes_present
    scale_keys = [k for k in state_dict if k.endswith("_scale")]
    has_scales = len(scale_keys) > 0

    if has_int8:
        raise gr.Error(
            "INT8-quantised checkpoints are not supported by the diffusers "
            "pipeline. Use a bf16 or fp8 checkpoint instead."
        )

    if has_fp8 and has_scales:
        raise gr.Error(
            "FP8-scaled checkpoints (e.g. Comfy-Org fp8_scaled) are designed "
            "for ComfyUI's native FP8 inference and cannot be accurately "
            "dequantised for diffusers. Use a bf16 checkpoint or a "
            "non-scaled FP8 checkpoint (e.g. AlperKTS/Krea2_FP8) instead."
        )

    # ── Convert ComfyUI key names to diffusers format if needed ─────
    state_dict = _convert_comfyui_keys(state_dict)

    # ── Dequantise / cast to bf16 ─────────────────────────────────────
    if has_fp8 and has_scales:
        # FP8 scaled (Comfy-Org style): dequantise with per-tensor scales
        if progress_callback:
            progress_callback("Dequantizing FP8 scaled weights to bfloat16...")
        for sk in scale_keys:
            weight_key = sk.removesuffix("_scale")
            if weight_key in state_dict:
                state_dict[weight_key] = (
                    state_dict[weight_key].to(torch.bfloat16)
                    * state_dict[sk].to(torch.bfloat16)
                )
            del state_dict[sk]

    # Cast remaining tensors to bf16, but keep norm layers in fp32 for
    # numerical stability (avoids the diffusers warning about norm modules).
    _NORM_PARTS = {"norm", "norm1", "norm2", "norm_q", "norm_k",
                   "qnorm", "knorm"}
    needs_cast = False
    for k in list(state_dict.keys()):
        t = state_dict[k]
        if not t.is_floating_point():
            continue
        key_parts = k.split(".")
        if any(part in _NORM_PARTS for part in key_parts):
            if t.dtype != torch.float32:
                state_dict[k] = t.to(torch.float32)
            continue
        if t.dtype != torch.bfloat16:
            needs_cast = True
            state_dict[k] = t.to(torch.bfloat16)

    if needs_cast and progress_callback and not (has_fp8 and has_scales):
        progress_callback("Converting weights to bfloat16...")

    # ── Build transformer and load weights ────────────────────────────
    transformer = Krea2Transformer2DModel.from_config({
        "hidden_size": 6144,
        "num_attention_heads": 48,
        "num_kv_heads": 12,
        "num_transformer_blocks": 28,
        "in_channels": 64,
        "num_text_layers": 12,
        "patch_size": 2,
        "is_distilled": True,
    })

    # Cast non-norm modules to bf16 individually to avoid the blanket
    # .to(dtype) warning about norm layers needing float32.
    for name, param in transformer.named_parameters():
        parts = name.split(".")
        if any(part in _NORM_PARTS for part in parts):
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.bfloat16)
    transformer.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()

    return transformer


class Krea2Generator:
    def __init__(self):
        self.pipe = None
        self.img2img_pipe = None   # kept for interface compat
        self.inpaint_pipe = None   # kept for interface compat
        self._active_loras = []
        self._model_type = "krea2"
        self._model_name = None
        self._is_single_file = False
        self._interrupt = False
        self._cached_embeds = None
        self._vae_name = None

    # ── Config compatibility ────────────────────────────────────────

    @staticmethod
    def _patch_text_encoder_config(model_path):
        """Ensure text_config has rope_scaling for older transformers versions.

        Models saved with transformers 5.x store rope info under
        ``rope_parameters``, but transformers 4.x reads ``rope_scaling``.
        If rope_scaling is missing or None in text_config, copy from
        rope_parameters so Qwen3VLTextRotaryEmbedding doesn't crash.
        """
        import json

        cfg_path = model_path / "text_encoder" / "config.json"
        if not cfg_path.exists():
            return

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        text_cfg = cfg.get("text_config", {})
        rope_params = text_cfg.get("rope_parameters")
        rope_scaling = text_cfg.get("rope_scaling")

        if rope_params and not rope_scaling:
            text_cfg["rope_scaling"] = rope_params
            cfg["text_config"] = text_cfg
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

    # ── Model discovery ──────────────────────────────────────────────

    def get_available_models(self):
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        models = []
        for item in _MODEL_DIR.iterdir():
            if item.name.startswith("_"):
                continue  # skip internal dirs like _encoders
            if item.is_dir() and (item / "model_index.json").exists():
                models.append(item.name)
            elif item.is_file() and item.suffix in (".safetensors", ".ckpt"):
                models.append(item.name)
        return sorted(models)

    def get_available_loras(self):
        _LORA_DIR.mkdir(parents=True, exist_ok=True)
        loras = []
        for f in _LORA_DIR.iterdir():
            if f.suffix == ".safetensors":
                loras.append(f.name)
        return sorted(loras)

    def get_available_vaes(self):
        """Krea 2 uses its own Qwen-Image VAE; custom VAE swapping is not supported."""
        return ["Default"]

    def load_vae(self, vae_name, progress_callback=None):
        """No-op: Krea 2 VAE cannot be swapped."""
        self._vae_name = None

    def needs_encoder_download(self, model_name):
        """Check if loading this model will require downloading components.

        Returns a description string if downloads are needed, or None if
        everything is cached locally.
        """
        local_path = _MODEL_DIR / model_name
        if not local_path.is_file():
            return None  # diffusers directory — fully self-contained

        te_cached = (_ENCODERS_DIR / "text_encoder").exists() and any((_ENCODERS_DIR / "text_encoder").iterdir())
        vae_cached = (_ENCODERS_DIR / "vae").exists() and any((_ENCODERS_DIR / "vae").iterdir())
        if te_cached and vae_cached:
            return None  # already cached

        parts = []
        if not te_cached:
            parts.append("Qwen3-VL text encoder (~9 GB)")
        if not vae_cached:
            parts.append("Qwen-Image VAE (~254 MB)")
        return (
            f"This single-file checkpoint requires downloading: "
            f"{', '.join(parts)}. Files will be cached in "
            f"models/krea2/_encoders/ for offline use afterward. "
            f"Select the model again to confirm and begin download."
        )

    # ── Model loading / unloading ────────────────────────────────────

    def load_model(self, model_name=None, progress_callback=None):
        if self.pipe is not None:
            self.unload_model()

        if not model_name:
            models = self.get_available_models()
            if not models:
                raise FileNotFoundError(
                    "No Krea 2 models found in models/krea2/. "
                    "Download a model (e.g. a .safetensors from CivitAI, or a "
                    "diffusers directory from HuggingFace) and place it in "
                    "the models/krea2/ directory."
                )
            model_name = models[0]

        local_path = _MODEL_DIR / model_name
        if not local_path.exists():
            raise FileNotFoundError(f"Model not found: {local_path}")

        self._model_name = local_path.name
        self._is_single_file = local_path.is_file()

        if progress_callback:
            progress_callback(f"Loading {self._model_name} (krea2)...")

        if self._is_single_file:
            self._load_from_single_file(local_path, progress_callback)
        else:
            self._load_from_directory(local_path, progress_callback)

        # Enable sequential CPU offload to fit within 24GB VRAM.
        # The bf16 transformer alone is ~25GB, too large for model-level
        # offload (which moves entire submodels at once). Sequential offload
        # moves individual layers to GPU on demand instead.
        if config.DEVICE == "cuda":
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.vae.enable_tiling()

        if progress_callback:
            progress_callback(f"Ready — {self._model_name}")

    def _load_from_directory(self, local_path, progress_callback=None):
        """Load from a diffusers-format directory (model_index.json + subdirs)."""
        self._patch_text_encoder_config(local_path)

        self.pipe = Krea2Pipeline.from_pretrained(
            str(local_path),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

    def _load_from_single_file(self, checkpoint_path, progress_callback=None):
        """Load from a single .safetensors file (transformer only).

        Downloads and caches the text encoder (Qwen3-VL-4B) and VAE
        (Qwen-Image) on first use.
        """
        _ENCODERS_DIR.mkdir(parents=True, exist_ok=True)

        # Load components
        text_encoder = _load_text_encoder(progress_callback)
        tokenizer = _load_tokenizer(progress_callback)
        vae = _load_vae(progress_callback)
        transformer = _load_single_file_transformer(checkpoint_path, progress_callback)

        # Build scheduler with Krea 2 settings
        scheduler = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=6400,
        )

        if progress_callback:
            progress_callback("Assembling pipeline...")

        self.pipe = Krea2Pipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            text_encoder_select_layers=_TEXT_ENCODER_SELECT_LAYERS,
            is_distilled=True,
            patch_size=2,
        )

    def unload_model(self):
        self._active_loras = []
        self._cached_embeds = None
        self.pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        self._model_name = None
        self._is_single_file = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Scheduler ────────────────────────────────────────────────────

    def set_scheduler(self, name: str):
        # Krea 2 only uses FlowMatchEulerDiscreteScheduler
        pass

    # ── VRAM management ──────────────────────────────────────────────

    def flush_vram(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Interrupt ────────────────────────────────────────────────────

    def interrupt(self):
        self._interrupt = True

    @property
    def was_interrupted(self):
        return self._interrupt

    def _step_callback(self, pipeline, i, t, callback_kwargs):
        if self._interrupt:
            pipeline._interrupt = True
        return callback_kwargs

    # ── LoRA management ──────────────────────────────────────────────

    def load_loras(self, lora_list):
        from pathlib import Path
        if self._active_loras:
            self.unload_loras()
        if not lora_list:
            return

        adapter_names = []
        adapter_weights = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Already found a")
            for i, (lora_path, weight) in enumerate(lora_list):
                p = Path(lora_path)
                name = f"lora_{i}"
                self.pipe.load_lora_weights(
                    str(p.parent), weight_name=p.name, adapter_name=name,
                )
                adapter_names.append(name)
                adapter_weights.append(weight)

        self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        self.pipe.fuse_lora(adapter_names=adapter_names, safe_fusing=False)
        self._active_loras = list(lora_list)

    def unload_loras(self):
        if self._active_loras:
            self.pipe.unfuse_lora()
            self.pipe.unload_lora_weights()
            self._active_loras = []

    # ── Text-to-image ────────────────────────────────────────────────

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str = "",
        steps: int = 8,
        guidance_scale: float = 0.0,
        width: int = config.DEFAULT_WIDTH,
        height: int = config.DEFAULT_HEIGHT,
        seed: int = config.DEFAULT_SEED,
        scheduler_name: str = "Euler",
        offload_encoders: bool = False,
        keep_encoders_offloaded: bool = False,
        batch_size: int = 1,
    ):
        """Generate an image from a text prompt.

        Note: negative_prompt is accepted for interface compatibility but
        Krea 2 Turbo does not use classifier-free guidance — it will be ignored
        unless guidance_scale > 0 (Base/Raw checkpoint).
        """
        self._interrupt = False
        self._cached_embeds = None
        self.flush_vram()

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=config.DEVICE).manual_seed(seed)

        kwargs = dict(
            prompt=positive_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback_on_step_end=self._step_callback,
        )

        # Pass negative prompt only when guidance is enabled (Base checkpoint)
        if guidance_scale > 0 and negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        if batch_size > 1:
            kwargs["num_images_per_prompt"] = batch_size

        result = self.pipe(**kwargs)
        images = result.images

        self.flush_vram()

        return images if batch_size > 1 else images[0]

    # ── Image-to-image ───────────────────────────────────────────────

    def img2img(
        self,
        source_image,
        positive_prompt: str,
        negative_prompt: str = "",
        strength: float = 0.7,
        steps: int = 8,
        guidance_scale: float = 0.0,
        seed: int = config.DEFAULT_SEED,
        scheduler_name: str = "Euler",
        offload_encoders: bool = False,
        use_cached_embeds: bool = False,
    ):
        """Image-to-image is not currently supported for Krea 2 models."""
        import gradio as gr
        raise gr.Error("Image-to-image is not currently supported for Krea 2 models.")

    # ── Inpainting ───────────────────────────────────────────────────

    def inpaint(
        self,
        source_image,
        mask_image,
        positive_prompt: str,
        negative_prompt: str = "",
        strength: float = 0.7,
        steps: int = 8,
        guidance_scale: float = 0.0,
        seed: int = config.DEFAULT_SEED,
        scheduler_name: str = "Euler",
    ):
        """Inpainting is not currently supported for Krea 2 models."""
        import gradio as gr
        raise gr.Error("Inpainting is not currently supported for Krea 2 models.")
