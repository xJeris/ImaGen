import json
import gc
import warnings

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
)
from compel import Compel, ReturnedEmbeddingsType
from safetensors import safe_open

import config
from prompt_parser import parse_weighted_prompt

# Scheduler name -> (class, extra kwargs)
SCHEDULERS = {
    "Euler": (EulerDiscreteScheduler, {}),
    "Euler Ancestral": (EulerAncestralDiscreteScheduler, {}),
    "DPM++ 2M Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "final_sigmas_type": "sigma_min"}),
    "DPM++ SDE Karras": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True, "final_sigmas_type": "sigma_min"}),
    "DDIM": (DDIMScheduler, {}),
    "UniPC": (UniPCMultistepScheduler, {}),
}

SCHEDULER_NAMES = list(SCHEDULERS.keys())


def detect_model_type(model_path):
    """Determine if a model is SDXL or SD 1.5.

    For diffusers folders: reads model_index.json.
    For single .safetensors files: uses file size heuristic (SDXL > 5GB).
    """
    if model_path.is_dir():
        index_file = model_path / "model_index.json"
        if index_file.exists():
            data = json.loads(index_file.read_text(encoding="utf-8"))
            class_name = data.get("_class_name", "")
            if "XL" in class_name:
                return "sdxl"
        return "sd15"
    else:
        # Single file — SDXL checkpoints are typically > 5GB
        size_gb = model_path.stat().st_size / (1024 ** 3)
        return "sdxl" if size_gb > 5.0 else "sd15"


class ImageGenerator:
    def __init__(self):
        self.pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        self.compel_proc = None
        self._active_loras = []
        self._model_type = None
        self._model_name = None
        self._interrupt = False

    def get_available_models(self):
        """List models in models/ — both diffusers folders and single .safetensors files."""
        config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        models = []
        for item in config.MODEL_CACHE_DIR.iterdir():
            if item.is_dir() and (item / "model_index.json").exists():
                models.append(item.name)
            elif item.is_file() and item.suffix == ".safetensors":
                models.append(item.name)
        return sorted(models)

    def unload_model(self):
        """Free VRAM by unloading the current model."""
        self._active_loras = []
        self.pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        self.compel_proc = None
        self._model_type = None
        self._model_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, model_name=None, progress_callback=None):
        """Load a model by name from models/ directory.

        If model_name is None, downloads the default SDXL model on first run
        or loads the first available model.
        """
        if self.pipe is not None:
            self.unload_model()

        local_path = None

        if model_name:
            local_path = config.MODEL_CACHE_DIR / model_name
            if not local_path.exists():
                raise FileNotFoundError(f"Model not found: {local_path}")
            self._is_single_file = local_path.is_file()
        else:
            self._is_single_file = False
            # First run or no model specified — try default, then download
            default_path = config.MODEL_CACHE_DIR / config.DEFAULT_MODEL_NAME
            if default_path.exists():
                local_path = default_path
            else:
                # Download default model
                if progress_callback:
                    progress_callback("Downloading model (first run, ~6.5GB)...")
                config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    config.DEFAULT_MODEL_ID,
                    torch_dtype=config.DTYPE,
                )
                if progress_callback:
                    progress_callback("Saving model to local cache...")
                pipe.save_pretrained(str(default_path))
                del pipe
                gc.collect()
                local_path = default_path

        # Detect model type
        self._model_type = detect_model_type(local_path)
        self._model_name = local_path.name

        if progress_callback:
            progress_callback(f"Loading {self._model_name} ({self._model_type})...")

        # Load appropriate pipeline
        if self._is_single_file:
            self._load_single_file(local_path)
        elif self._model_type == "sdxl":
            self._load_sdxl(local_path)
        else:
            self._load_sd15(local_path)

        # Set scheduler
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Ensure VAE runs in float32 for numerical stability during decode
        self.pipe.vae.to(torch.float32)

        # Move to device and optimize
        self.pipe.to(config.DEVICE)
        if config.DEVICE == "cuda":
            # VAE tiling: decode/encode large images in overlapping tiles
            # instead of all at once — critical for hires fix at 1536+ px
            self.pipe.enable_vae_tiling()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                self.pipe.enable_attention_slicing()

        # Build img2img and inpaint pipelines sharing components
        self._build_img2img()
        self._build_inpaint()

        # Initialize compel for prompt weighting
        self._init_compel()

        if progress_callback:
            progress_callback(f"Ready — {self._model_name}")

    def _load_sdxl(self, path):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            str(path),
            torch_dtype=config.DTYPE,
            local_files_only=True,
        )

    def _load_sd15(self, path):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            str(path),
            torch_dtype=config.DTYPE,
            local_files_only=True,
        )

    def _load_single_file(self, path):
        """Load a single .safetensors checkpoint file."""
        if self._model_type == "sdxl":
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                str(path),
                torch_dtype=config.DTYPE,
            )
        else:
            self.pipe = StableDiffusionPipeline.from_single_file(
                str(path),
                torch_dtype=config.DTYPE,
            )

    def _build_img2img(self):
        if self._model_type == "sdxl":
            self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer=self.pipe.tokenizer,
                tokenizer_2=self.pipe.tokenizer_2,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
            )
        else:
            self.img2img_pipe = StableDiffusionImg2ImgPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
            )

    def _build_inpaint(self):
        if self._model_type == "sdxl":
            self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer=self.pipe.tokenizer,
                tokenizer_2=self.pipe.tokenizer_2,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
            )
        else:
            self.inpaint_pipe = StableDiffusionInpaintPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
            )

    def _init_compel(self):
        if self._model_type == "sdxl":
            self.compel_proc = Compel(
                tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
            )
        else:
            self.compel_proc = Compel(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
            )

    def set_scheduler(self, name: str):
        """Swap the scheduler on all pipelines by name."""
        if name not in SCHEDULERS:
            return
        cls, kwargs = SCHEDULERS[name]
        self.pipe.scheduler = cls.from_config(self.pipe.scheduler.config, **kwargs)
        if self.img2img_pipe is not None:
            self.img2img_pipe.scheduler = self.pipe.scheduler
        if self.inpaint_pipe is not None:
            self.inpaint_pipe.scheduler = self.pipe.scheduler

    def flush_vram(self):
        """Free cached VRAM between heavy phases (e.g. between hires upscale and img2img)."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def interrupt(self):
        """Signal the pipeline to stop after the current step."""
        self._interrupt = True

    @property
    def was_interrupted(self):
        """Check whether the last generation was interrupted."""
        return self._interrupt
    
    def _check_lora_compatible(self, lora_path: str) -> bool:
        """Check if a LoRA file is compatible with the currently loaded model.

        Uses a simple heuristic: SDXL LoRAs have 'text_encoder_2' or
        'lora_te2' keys (for the second text encoder), SD 1.5 LoRAs don't.
        This avoids the fragile UNet key-matching approach.
        """
        if self._model_type is None:
            return True  # no model loaded yet, show all

        try:
            with safe_open(lora_path, framework="pt") as f:
                lora_keys = set(f.keys())
        except Exception:
            return True  # if we can't read it, don't hide it

        if not lora_keys:
            return True

        # Detect if the LoRA targets SDXL (has second text encoder keys)
        has_te2 = any(
            "text_encoder_2" in k or "lora_te2_" in k
            for k in lora_keys
        )

        if self._model_type == "sdxl":
            # SDXL model: accept LoRAs that have TE2 keys (SDXL LoRAs),
            # or LoRAs that only have UNet keys (style LoRAs work on both)
            has_te1_only = any(
                ("text_encoder." in k or "lora_te_" in k or "lora_te1_" in k)
                and "text_encoder_2" not in k and "lora_te2_" not in k
                for k in lora_keys
            )
            # Reject if it has SD1.5-only text encoder keys and no TE2 keys
            if has_te1_only and not has_te2:
                return False
            return True
        else:
            # SD 1.5 model: reject LoRAs that have TE2 keys (SDXL-only)
            return not has_te2

    def _build_embeddings(self, prompt_text):
        """Build prompt embeddings. Returns (embeds, pooled) for SDXL, (embeds, None) for SD 1.5."""
        if self._model_type == "sdxl":
            conditioning, pooled = self.compel_proc(prompt_text)
            return conditioning, pooled
        else:
            conditioning = self.compel_proc(prompt_text)
            return conditioning, None

    def load_loras(self, lora_list):
        """Load and fuse one or more LoRAs.

        Args:
            lora_list: list of (path, weight) tuples, e.g.
                       [("/path/to/lora.safetensors", 0.8)]
        """
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
        self.pipe.fuse_lora(adapter_names=adapter_names)
        self._active_loras = list(lora_list)

    def unload_loras(self):
        """Remove all active LoRAs."""
        if self._active_loras:
            self.pipe.unfuse_lora()
            self.pipe.unload_lora_weights()
            self._active_loras = []

    def _step_callback(self, pipeline, i, t, callback_kwargs):
        """Check interrupt flag at each diffusion step."""
        if self._interrupt:
            pipeline._interrupt = True
        return callback_kwargs

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str = "",
        steps: int = config.DEFAULT_STEPS,
        guidance_scale: float = config.DEFAULT_GUIDANCE_SCALE,
        width: int = config.DEFAULT_WIDTH,
        height: int = config.DEFAULT_HEIGHT,
        seed: int = config.DEFAULT_SEED,
        scheduler_name: str = "Euler",
    ):
        """Generate an image from text prompts. Returns a PIL Image."""
        self._interrupt = False
        self.set_scheduler(scheduler_name)

        parsed_pos = parse_weighted_prompt(positive_prompt)
        parsed_neg = parse_weighted_prompt(negative_prompt) if negative_prompt else ""

        pos_embeds, pos_pooled = self._build_embeddings(parsed_pos)
        neg_embeds, neg_pooled = self._build_embeddings(parsed_neg if parsed_neg else "")

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=config.DEVICE).manual_seed(seed)

        kwargs = dict(
            prompt_embeds=pos_embeds,
            negative_prompt_embeds=neg_embeds,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback_on_step_end=self._step_callback,
        )

        if self._model_type == "sdxl":
            kwargs["pooled_prompt_embeds"] = pos_pooled
            kwargs["negative_pooled_prompt_embeds"] = neg_pooled

        image = self.pipe(**kwargs).images[0]
        return image

    def img2img(
        self,
        source_image,
        positive_prompt: str,
        negative_prompt: str = "",
        strength: float = 0.7,
        steps: int = config.DEFAULT_STEPS,
        guidance_scale: float = config.DEFAULT_GUIDANCE_SCALE,
        seed: int = config.DEFAULT_SEED,
        scheduler_name: str = "Euler",
        offload_encoders: bool = False,
    ):
        """Generate a new image from a source image + text prompts. Returns a PIL Image.

        Args:
            offload_encoders: If True, move text encoders to CPU before the
                diffusion pass and restore them after.  Frees VRAM for the UNet
                when processing large images (used by hires fix).
        """
        self._interrupt = False
        self.set_scheduler(scheduler_name)

        source_image = source_image.convert("RGB")

        parsed_pos = parse_weighted_prompt(positive_prompt)
        parsed_neg = parse_weighted_prompt(negative_prompt) if negative_prompt else ""

        pos_embeds, pos_pooled = self._build_embeddings(parsed_pos)
        neg_embeds, neg_pooled = self._build_embeddings(parsed_neg if parsed_neg else "")

        # Offload text encoders to free VRAM for the UNet at high resolution.
        # Embeddings are already computed above so the encoders aren't needed.
        # We must also patch _execution_device on the img2img pipe because
        # diffusers infers the device from the first nn.Module component — if
        # that happens to be a text encoder now on CPU, the whole pipeline
        # would incorrectly run on CPU and hit a device mismatch.
        if offload_encoders and config.DEVICE == "cuda":
            self.pipe.text_encoder.to("cpu")
            if hasattr(self.pipe, "text_encoder_2") and self.pipe.text_encoder_2 is not None:
                self.pipe.text_encoder_2.to("cpu")
            self.flush_vram()

            # Force the img2img pipe to use CUDA despite text encoders on CPU
            _orig_exec = type(self.img2img_pipe)._execution_device.fget
            type(self.img2img_pipe)._execution_device = property(
                lambda self_pipe: torch.device("cuda")
            )

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=config.DEVICE).manual_seed(seed)

        kwargs = dict(
            image=source_image,
            prompt_embeds=pos_embeds,
            negative_prompt_embeds=neg_embeds,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=self._step_callback,
        )

        if self._model_type == "sdxl":
            kwargs["pooled_prompt_embeds"] = pos_pooled
            kwargs["negative_pooled_prompt_embeds"] = neg_pooled

        image = self.img2img_pipe(**kwargs).images[0]

        # Restore text encoders and execution device property
        if offload_encoders and config.DEVICE == "cuda":
            type(self.img2img_pipe)._execution_device = property(_orig_exec)
            self.pipe.text_encoder.to(config.DEVICE)
            if hasattr(self.pipe, "text_encoder_2") and self.pipe.text_encoder_2 is not None:
                self.pipe.text_encoder_2.to(config.DEVICE)

        return image

    def inpaint(
        self,
        source_image,
        mask_image,
        positive_prompt: str,
        negative_prompt: str = "",
        strength: float = 0.7,
        steps: int = config.DEFAULT_STEPS,
        guidance_scale: float = config.DEFAULT_GUIDANCE_SCALE,
        seed: int = config.DEFAULT_SEED,
        scheduler_name: str = "Euler",
    ):
        """Inpaint masked regions of an image. Returns a PIL Image."""
        self._interrupt = False
        self.set_scheduler(scheduler_name)

        source_image = source_image.convert("RGB")
        mask_image = mask_image.convert("RGB")

        parsed_pos = parse_weighted_prompt(positive_prompt)
        parsed_neg = parse_weighted_prompt(negative_prompt) if negative_prompt else ""

        pos_embeds, pos_pooled = self._build_embeddings(parsed_pos)
        neg_embeds, neg_pooled = self._build_embeddings(parsed_neg if parsed_neg else "")

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=config.DEVICE).manual_seed(seed)

        kwargs = dict(
            image=source_image,
            mask_image=mask_image,
            prompt_embeds=pos_embeds,
            negative_prompt_embeds=neg_embeds,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=self._step_callback,
        )

        if self._model_type == "sdxl":
            kwargs["pooled_prompt_embeds"] = pos_pooled
            kwargs["negative_pooled_prompt_embeds"] = neg_pooled

        image = self.inpaint_pipe(**kwargs).images[0]
        return image

    def get_available_loras(self):
        """List compatible LoRA files available in the loras/ directory."""
        config.LORA_DIR.mkdir(parents=True, exist_ok=True)
        loras = []
        for f in config.LORA_DIR.iterdir():
            if f.suffix == ".safetensors":
                if self._check_lora_compatible(str(f)):
                    loras.append(f.name)
        return sorted(loras)
