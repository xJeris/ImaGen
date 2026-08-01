"""Flux pipeline — transformer-based architecture (Black Forest Labs).

Flux uses a flow-matching transformer instead of a UNet, T5 + CLIP text
encoders, and does NOT support negative prompts or Compel prompt weighting.
The [token:weight] syntax will be silently ignored.

Implements the same interface as ImageGenerator in pipeline.py so app.py
can swap between architectures transparently.
"""

import gc
import json
import warnings

import torch
import torch.nn as nn
from diffusers import (
    FluxPipeline,
    FluxImg2ImgPipeline,
    FluxTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)

import config

SCHEDULERS = {
    "Euler": (FlowMatchEulerDiscreteScheduler, {}),
}

SCHEDULER_NAMES = list(SCHEDULERS.keys())

_MODEL_DIR = config.ARCH_MODEL_DIRS["Flux"]
_LORA_DIR = config.ARCH_LORA_DIRS["Flux"]
_CONFIG_DIR = _MODEL_DIR / "_config"  # bundled Flux config files for offline single-file loading
_ENCODERS_DIR = _MODEL_DIR / "_encoders"  # cached text encoders (auto-downloaded on first use)

# Non-gated HuggingFace repos for the text encoders
_CLIP_REPO = "openai/clip-vit-large-patch14"
_T5_REPO = "comfyanonymous/flux_text_encoders"
_T5_FILENAME = "t5xxl_fp8_e4m3fn.safetensors"  # kept as fallback reference
_T5_PRETRAINED_REPO = "google/t5-v1_1-xxl"  # for NF4 quantized loading


def _load_clip_encoder(progress_callback=None):
    """Load the CLIP text encoder, downloading from HuggingFace on first use."""
    from transformers import CLIPTextModel

    clip_dir = _ENCODERS_DIR / "clip"
    if clip_dir.exists() and any(clip_dir.iterdir()):
        # Already cached locally
        if progress_callback:
            progress_callback("Loading CLIP text encoder (cached)...")
        return CLIPTextModel.from_pretrained(
            str(clip_dir), torch_dtype=config.DTYPE, local_files_only=True,
        )

    # Download from non-gated repo and cache locally
    if progress_callback:
        progress_callback(f"Downloading CLIP text encoder from {_CLIP_REPO} (first time only)...")
    clip_dir.mkdir(parents=True, exist_ok=True)
    model = CLIPTextModel.from_pretrained(_CLIP_REPO, torch_dtype=config.DTYPE)
    model.save_pretrained(str(clip_dir))
    return model


def _load_clip_tokenizer(progress_callback=None):
    """Load the CLIP tokenizer, downloading from HuggingFace on first use."""
    from transformers import CLIPTokenizer

    tok_dir = _ENCODERS_DIR / "clip_tokenizer"
    if tok_dir.exists() and any(tok_dir.iterdir()):
        return CLIPTokenizer.from_pretrained(str(tok_dir), local_files_only=True)

    if progress_callback:
        progress_callback("Downloading CLIP tokenizer (first time only)...")
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = CLIPTokenizer.from_pretrained(_CLIP_REPO)
    tokenizer.save_pretrained(str(tok_dir))
    return tokenizer


def _load_t5_tokenizer(progress_callback=None):
    """Load the T5 tokenizer, downloading from HuggingFace on first use."""
    from transformers import AutoTokenizer

    tok_dir = _ENCODERS_DIR / "t5_tokenizer"
    if tok_dir.exists() and any(tok_dir.iterdir()):
        return AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)

    if progress_callback:
        progress_callback("Downloading T5 tokenizer (first time only)...")
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl")
    tokenizer.save_pretrained(str(tok_dir))
    return tokenizer


def _load_t5_encoder(progress_callback=None):
    """Load the T5-XXL text encoder from the fp8 single file (~4.9 GB download).

    Downloads the fp8 version from comfyanonymous/flux_text_encoders on first
    use. Weights are loaded as fp8 then cast to bfloat16 (wider exponent
    range than fp16 prevents overflow/NaN from fp8 conversion).
    """
    from transformers import T5EncoderModel, T5Config
    from safetensors.torch import load_file

    t5_dir = _ENCODERS_DIR / "t5"
    t5_dir.mkdir(parents=True, exist_ok=True)
    raw_file = t5_dir / _T5_FILENAME

    # Download if not present
    if not raw_file.exists():
        if progress_callback:
            progress_callback("Downloading T5-XXL text encoder fp8 (~4.9 GB, first time only)...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=_T5_REPO,
            filename=_T5_FILENAME,
            local_dir=str(t5_dir),
        )
        # Clean up .cache dir left by hf_hub_download
        _cache_dir = t5_dir / ".cache"
        if _cache_dir.exists():
            import shutil
            shutil.rmtree(_cache_dir, ignore_errors=True)

    # Build model skeleton and load fp8 weights, cast to bf16
    if progress_callback:
        progress_callback("Loading T5-XXL text encoder...")
    t5_cfg = T5Config.from_pretrained(str(_CONFIG_DIR / "text_encoder_2"))
    model = T5EncoderModel(t5_cfg)

    state_dict = load_file(str(raw_file))
    # Cast fp8 weights to bf16 before loading to avoid NaN from fp8→fp32 conversion
    for k in state_dict:
        if state_dict[k].is_floating_point():
            state_dict[k] = state_dict[k].to(torch.bfloat16)
    model = model.to(dtype=torch.bfloat16)
    model.load_state_dict(state_dict, strict=False)

    return model


def _load_single_file_components(checkpoint_path, progress_callback=None):
    """Load transformer (NF4) and VAE from a single-file Flux checkpoint.

    Loads the checkpoint once, splits it into transformer and VAE keys,
    builds the NF4-quantized transformer and bf16 VAE, then returns both.
    This avoids from_single_file re-reading the 12 GB file.
    """
    from safetensors.torch import load_file
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import (
        convert_flux_transformer_checkpoint_to_diffusers,
        convert_ldm_vae_checkpoint,
    )
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from bitsandbytes.nn import Linear4bit

    # 1. Load the full checkpoint once
    if progress_callback:
        progress_callback("Loading checkpoint...")
    raw_sd = load_file(str(checkpoint_path))

    # Split into transformer and VAE keys
    vae_keys = {k: v for k, v in raw_sd.items() if k.startswith("vae.")}
    transformer_keys = {k: v for k, v in raw_sd.items() if not k.startswith("vae.")}
    del raw_sd
    gc.collect()

    # --- Build NF4 transformer ---
    if progress_callback:
        progress_callback("Building NF4 transformer...")

    with init_empty_weights():
        transformer = FluxTransformer2DModel.from_config(
            str(_CONFIG_DIR / "transformer")
        )

    converted = convert_flux_transformer_checkpoint_to_diffusers(transformer_keys)
    del transformer_keys
    gc.collect()

    # Replace Linear layers with NF4, skipping precision-sensitive top-level
    # modules (embedding projections, conditioning MLPs, final output).
    _SKIP_TOP_LEVEL = {
        "proj_out", "x_embedder", "context_embedder",
        "time_text_embed", "norm_out",
    }

    def _replace_linear_with_nf4(module, depth=0):
        for name, child in module.named_children():
            if depth == 0 and name in _SKIP_TOP_LEVEL:
                continue
            if isinstance(child, nn.Linear):
                setattr(module, name, Linear4bit(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
                ))
            else:
                _replace_linear_with_nf4(child, depth + 1)

    _replace_linear_with_nf4(transformer)

    # Cast weights to bf16 and load into quantized model on GPU
    for k in converted:
        if converted[k].is_floating_point():
            converted[k] = converted[k].to(torch.bfloat16)

    if progress_callback:
        progress_callback("Quantizing transformer to NF4...")
    for key, val in converted.items():
        set_module_tensor_to_device(transformer, key, "cuda", value=val)
    del converted
    gc.collect()

    # --- Build VAE ---
    if progress_callback:
        progress_callback("Loading VAE...")

    vae = AutoencoderKL.from_config(str(_CONFIG_DIR / "vae"))
    vae_converted = convert_ldm_vae_checkpoint(vae_keys, vae.config)
    del vae_keys
    gc.collect()

    vae.load_state_dict(vae_converted)
    del vae_converted
    gc.collect()

    return transformer, vae


class FluxGenerator:
    def __init__(self):
        self.pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None  # kept for interface compat
        self._active_loras = []
        self._model_type = "flux"
        self._model_name = None
        self._is_single_file = False
        self._interrupt = False
        self._cached_embeds = None
        self._vae_name = None

    def get_available_vaes(self):
        """Flux uses its own VAE architecture; custom VAE swapping is not supported."""
        return ["Default"]

    def load_vae(self, vae_name, progress_callback=None):
        """No-op: Flux VAE cannot be swapped."""
        self._vae_name = None

    def get_available_models(self):
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        models = []
        for item in _MODEL_DIR.iterdir():
            if item.name.startswith("_"):
                continue  # skip internal dirs like _config, _encoders
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

    def unload_model(self):
        self._active_loras = []
        self.pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        self._model_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, model_name=None, progress_callback=None):
        if self.pipe is not None:
            self.unload_model()

        if not model_name:
            models = self.get_available_models()
            if not models:
                raise FileNotFoundError("No Flux models found in models/flux/")
            model_name = models[0]

        local_path = _MODEL_DIR / model_name
        if not local_path.exists():
            raise FileNotFoundError(f"Model not found: {local_path}")

        self._is_single_file = local_path.is_file()
        self._model_name = local_path.name

        if progress_callback:
            progress_callback(f"Loading {self._model_name} (flux)...")

        if self._is_single_file:
            # Single-file Flux checkpoints typically don't include the text
            # encoders or tokenizers. Load them separately from non-gated
            # HuggingFace repos (auto-downloaded and cached on first use).
            text_encoder = _load_clip_encoder(progress_callback)
            text_encoder_2 = _load_t5_encoder(progress_callback)
            tokenizer = _load_clip_tokenizer(progress_callback)
            tokenizer_2 = _load_t5_tokenizer(progress_callback)

            if progress_callback:
                progress_callback(f"Loading {self._model_name} (NF4 transformer + VAE)...")

            # Load checkpoint once, build NF4 transformer + VAE together
            transformer, vae = _load_single_file_components(
                local_path, progress_callback
            )

            if progress_callback:
                progress_callback("Assembling pipeline...")

            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(_CONFIG_DIR / "scheduler")
            )

            self.pipe = FluxPipeline(
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                scheduler=scheduler,
            )
        else:
            self.pipe = FluxPipeline.from_pretrained(
                str(local_path), torch_dtype=config.DTYPE, local_files_only=True,
            )

        if self._is_single_file and config.DEVICE == "cuda":
            # Transformer is already on CUDA (NF4 quantized). Move VAE and
            # CLIP to GPU. Keep T5 on CPU — it's ~10 GB in bf16 and would
            # leave no headroom for inference. It gets moved to GPU
            # temporarily during encode_prompt in generate().
            self.pipe.vae.to(config.DEVICE, dtype=torch.bfloat16)
            self.pipe.text_encoder.to(config.DEVICE)  # CLIP ~0.5 GB
            # text_encoder_2 (T5) stays on CPU
        else:
            self.pipe.to(config.DEVICE)

        if config.DEVICE == "cuda":
            self.pipe.vae.enable_tiling()

        # Build img2img pipeline sharing the same components
        self._build_img2img()

        if progress_callback:
            progress_callback(f"Ready — {self._model_name}")

    def _build_img2img(self):
        try:
            self.img2img_pipe = FluxImg2ImgPipeline(
                transformer=self.pipe.transformer,
                scheduler=self.pipe.scheduler,
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer=self.pipe.tokenizer,
                tokenizer_2=self.pipe.tokenizer_2,
            )
        except Exception:
            # FluxImg2ImgPipeline may not be available in older diffusers
            self.img2img_pipe = None

    def set_scheduler(self, name: str):
        # Flux only supports FlowMatchEulerDiscreteScheduler
        pass

    def flush_vram(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _decode_latents(self, latents):
        """Decode latents to a PIL image using the VAE (outside autocast)."""
        from diffusers.image_processor import VaeImageProcessor
        latents = self.pipe._unpack_latents(
            latents, self._last_height, self._last_width, self.pipe.vae_scale_factor
        )
        latents = (
            latents / self.pipe.vae.config.scaling_factor
        ) + self.pipe.vae.config.shift_factor
        with torch.no_grad():
            decoded = self.pipe.vae.decode(latents, return_dict=False)[0]
        del latents
        image_processor = VaeImageProcessor(vae_scale_factor=self.pipe.vae_scale_factor)
        image = image_processor.postprocess(decoded, output_type="pil")[0]
        del decoded
        return image

    def interrupt(self):
        self._interrupt = True

    @property
    def was_interrupted(self):
        return self._interrupt

    def _step_callback(self, pipeline, i, t, callback_kwargs):
        if self._interrupt:
            pipeline._interrupt = True
        return callback_kwargs

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
        self.pipe.fuse_lora(adapter_names=adapter_names, safe_fusing=True)
        self._active_loras = list(lora_list)

    def unload_loras(self):
        if self._active_loras:
            self.pipe.unfuse_lora()
            self.pipe.unload_lora_weights()
            self._active_loras = []

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str = "",
        steps: int = 20,
        guidance_scale: float = 3.5,
        width: int = config.DEFAULT_WIDTH,
        height: int = config.DEFAULT_HEIGHT,
        seed: int = config.DEFAULT_SEED,
        scheduler_name: str = "Euler",
        offload_encoders: bool = False,
        keep_encoders_offloaded: bool = False,
    ):
        """Generate an image from a text prompt.

        Note: negative_prompt is accepted for interface compatibility but
        Flux does not support negative prompts — it will be ignored.
        """
        self._interrupt = False
        # negative_prompt is intentionally ignored for Flux

        # Clear leftover VRAM from any previous generation before loading encoders
        self._cached_embeds = None
        self.flush_vram()

        # Move both text encoders to GPU for encoding, then back to CPU to free VRAM
        if config.DEVICE == "cuda":
            self.pipe.text_encoder.to(config.DEVICE)
            self.pipe.text_encoder_2.to(config.DEVICE)

        prompt_embeds, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=positive_prompt,
            prompt_2=None,
            device=torch.device(config.DEVICE),
            max_sequence_length=512,
        )
        # Match transformer compute dtype (NF4 layers operate in bf16)
        prompt_embeds = prompt_embeds.to(torch.bfloat16)
        pooled_prompt_embeds = pooled_prompt_embeds.to(torch.bfloat16)

        # Cache embeddings for potential hires img2img reuse
        self._cached_embeds = (prompt_embeds, pooled_prompt_embeds)

        # Offload both text encoders to CPU — embeddings are already computed.
        # Patch _execution_device so diffusers doesn't infer CPU from the
        # offloaded text encoders.
        _orig_exec = None
        if config.DEVICE == "cuda":
            self.pipe.text_encoder.to("cpu")
            self.pipe.text_encoder_2.to("cpu")
            self.flush_vram()

            _orig_exec = type(self.pipe)._execution_device.fget
            type(self.pipe)._execution_device = property(
                lambda self_pipe: torch.device("cuda")
            )

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=config.DEVICE).manual_seed(seed)

        # Store for _decode_latents (needs to unpack latents to spatial dims)
        self._last_height = height
        self._last_width = width

        kwargs = dict(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback_on_step_end=self._step_callback,
        )

        # Run diffusion steps under autocast (NF4 transformer needs bf16),
        # but get raw latents so VAE decodes outside autocast.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = self.pipe(**kwargs, output_type="latent")
            latents = result.images
            del result

        # Restore execution device property (encoders stay on CPU)
        if config.DEVICE == "cuda" and _orig_exec is not None:
            type(self.pipe)._execution_device = property(_orig_exec)

        # Decode latents with VAE outside autocast (VAE uses force_upcast
        # to float32 internally, which conflicts with bf16 autocast)
        image = self._decode_latents(latents)

        # Free leftover VRAM from this generation so the next one starts clean
        del latents, prompt_embeds, pooled_prompt_embeds
        self.flush_vram()

        return image

    def img2img(
        self,
        source_image,
        positive_prompt: str,
        negative_prompt: str = "",
        strength: float = 0.7,
        steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int = config.DEFAULT_SEED,
        scheduler_name: str = "Euler",
        offload_encoders: bool = False,
        use_cached_embeds: bool = False,
    ):
        """Generate from a source image + text prompt.

        Note: negative_prompt is ignored for Flux.
        """
        import gradio as gr

        if self.img2img_pipe is None:
            raise gr.Error(
                "Flux img2img is not available. Your version of diffusers "
                "may not include FluxImg2ImgPipeline."
            )

        self._interrupt = False
        source_image = source_image.convert("RGB")
        self._last_width, self._last_height = source_image.size

        # Reuse cached embeddings from txt2img if available, otherwise encode
        if use_cached_embeds and self._cached_embeds is not None:
            prompt_embeds, pooled_prompt_embeds = self._cached_embeds
        else:
            # Move T5 to GPU for encoding, then back to CPU
            if config.DEVICE == "cuda":
                self.pipe.text_encoder_2.to(config.DEVICE)
            prompt_embeds, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=positive_prompt,
                prompt_2=None,
                max_sequence_length=512,
            )
            # Match transformer compute dtype (NF4 layers operate in bf16)
            prompt_embeds = prompt_embeds.to(torch.bfloat16)
            pooled_prompt_embeds = pooled_prompt_embeds.to(torch.bfloat16)

        # Offload text encoders to CPU — embeddings are already computed.
        # Patch _execution_device so diffusers doesn't infer CPU.
        _orig_exec = None
        if config.DEVICE == "cuda":
            self.pipe.text_encoder.to("cpu")
            self.pipe.text_encoder_2.to("cpu")
            self.flush_vram()

            _orig_exec = type(self.img2img_pipe)._execution_device.fget
            type(self.img2img_pipe)._execution_device = property(
                lambda self_pipe: torch.device("cuda")
            )

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=config.DEVICE).manual_seed(seed)

        kwargs = dict(
            image=source_image,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=self._step_callback,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = self.img2img_pipe(**kwargs, output_type="latent")
            latents = result.images
            del result

        # Restore execution device property (encoders stay on CPU)
        if config.DEVICE == "cuda" and _orig_exec is not None:
            type(self.img2img_pipe)._execution_device = property(_orig_exec)

        image = self._decode_latents(latents)

        del latents, prompt_embeds, pooled_prompt_embeds
        self._cached_embeds = None
        self.flush_vram()

        return image

    def inpaint(
        self,
        source_image,
        mask_image,
        positive_prompt: str,
        negative_prompt: str = "",
        strength: float = 0.7,
        steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int = config.DEFAULT_SEED,
        scheduler_name: str = "Euler",
    ):
        """Inpainting is not currently supported for Flux models."""
        import gradio as gr
        raise gr.Error("Inpainting is not currently supported for Flux models.")
