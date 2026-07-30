"""Illustrious / NoobAI pipeline — SDXL-based architecture for anime/illustration.

Implements the same interface as ImageGenerator in pipeline.py so app.py
can swap between architectures transparently.
"""

import gc
import json
import warnings

import torch
from diffusers import (
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

SCHEDULERS = {
    "Euler": (EulerDiscreteScheduler, {}),
    "Euler Ancestral": (EulerAncestralDiscreteScheduler, {}),
    "DPM++ 2M Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "final_sigmas_type": "sigma_min"}),
    "DPM++ SDE Karras": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True, "final_sigmas_type": "sigma_min"}),
    "DDIM": (DDIMScheduler, {}),
    "UniPC": (UniPCMultistepScheduler, {}),
}

SCHEDULER_NAMES = list(SCHEDULERS.keys())

_MODEL_DIR = config.ARCH_MODEL_DIRS["Illustrious"]
_LORA_DIR = config.ARCH_LORA_DIRS["Illustrious"]


class IllustriousGenerator:
    def __init__(self):
        self.pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        self.compel_proc = None
        self._active_loras = []
        self._model_type = "illustrious"
        self._model_name = None
        self._is_single_file = False
        self._interrupt = False
        self._cached_embeds = None

    def get_available_models(self):
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        models = []
        for item in _MODEL_DIR.iterdir():
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
        self.compel_proc = None
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
                raise FileNotFoundError("No Illustrious models found in models/illustrious/")
            model_name = models[0]

        local_path = _MODEL_DIR / model_name
        if not local_path.exists():
            raise FileNotFoundError(f"Model not found: {local_path}")

        self._is_single_file = local_path.is_file()
        self._model_name = local_path.name

        if progress_callback:
            progress_callback(f"Loading {self._model_name} (illustrious)...")

        if self._is_single_file:
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                str(local_path), torch_dtype=config.DTYPE,
            )
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                str(local_path), torch_dtype=config.DTYPE, local_files_only=True,
            )

        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe.to(config.DEVICE)
        if config.DEVICE == "cuda":
            self.pipe.vae.enable_tiling()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                from diffusers.models.attention_processor import AttnProcessor2_0
                self.pipe.unet.set_attn_processor(AttnProcessor2_0())

        self._build_img2img()
        self._build_inpaint()
        self._init_compel()

        if progress_callback:
            progress_callback(f"Ready — {self._model_name}")

    def _build_img2img(self):
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            text_encoder_2=self.pipe.text_encoder_2,
            tokenizer=self.pipe.tokenizer,
            tokenizer_2=self.pipe.tokenizer_2,
            unet=self.pipe.unet,
            scheduler=self.pipe.scheduler,
        )

    def _build_inpaint(self):
        self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            text_encoder_2=self.pipe.text_encoder_2,
            tokenizer=self.pipe.tokenizer,
            tokenizer_2=self.pipe.tokenizer_2,
            unet=self.pipe.unet,
            scheduler=self.pipe.scheduler,
        )

    def _init_compel(self):
        self.compel_proc = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

    def set_scheduler(self, name: str):
        if name not in SCHEDULERS:
            return
        cls, kwargs = SCHEDULERS[name]
        self.pipe.scheduler = cls.from_config(self.pipe.scheduler.config, **kwargs)
        if self.img2img_pipe is not None:
            self.img2img_pipe.scheduler = self.pipe.scheduler
        if self.inpaint_pipe is not None:
            self.inpaint_pipe.scheduler = self.pipe.scheduler

    def flush_vram(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def interrupt(self):
        self._interrupt = True

    @property
    def was_interrupted(self):
        return self._interrupt

    def _build_embeddings(self, prompt_text):
        with torch.inference_mode():
            conditioning, pooled = self.compel_proc(prompt_text)
            return conditioning, pooled

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
        self.pipe.fuse_lora(adapter_names=adapter_names)
        self.pipe.to(dtype=config.DTYPE)
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
        steps: int = 28,
        guidance_scale: float = 5.0,
        width: int = config.DEFAULT_WIDTH,
        height: int = config.DEFAULT_HEIGHT,
        seed: int = config.DEFAULT_SEED,
        scheduler_name: str = "Euler Ancestral",
        offload_encoders: bool = False,
        keep_encoders_offloaded: bool = False,
    ):
        self._interrupt = False
        self.set_scheduler(scheduler_name)

        parsed_pos = parse_weighted_prompt(positive_prompt)
        parsed_neg = parse_weighted_prompt(negative_prompt) if negative_prompt else ""

        pos_embeds, pos_pooled = self._build_embeddings(parsed_pos)
        neg_embeds, neg_pooled = self._build_embeddings(parsed_neg if parsed_neg else "")

        self._cached_embeds = (pos_embeds, pos_pooled, neg_embeds, neg_pooled)

        if offload_encoders and config.DEVICE == "cuda":
            self.pipe.text_encoder.to("cpu")
            if self.pipe.text_encoder_2 is not None:
                self.pipe.text_encoder_2.to("cpu")
            self.flush_vram()
            _orig_exec = type(self.pipe)._execution_device.fget
            type(self.pipe)._execution_device = property(
                lambda self_pipe: torch.device("cuda")
            )

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=config.DEVICE).manual_seed(seed)

        kwargs = dict(
            prompt_embeds=pos_embeds,
            negative_prompt_embeds=neg_embeds,
            pooled_prompt_embeds=pos_pooled,
            negative_pooled_prompt_embeds=neg_pooled,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback_on_step_end=self._step_callback,
        )

        image = self.pipe(**kwargs).images[0]

        if offload_encoders and config.DEVICE == "cuda":
            type(self.pipe)._execution_device = property(_orig_exec)
            if not keep_encoders_offloaded:
                self.pipe.text_encoder.to(config.DEVICE)
                if self.pipe.text_encoder_2 is not None:
                    self.pipe.text_encoder_2.to(config.DEVICE)

        return image

    def img2img(
        self,
        source_image,
        positive_prompt: str,
        negative_prompt: str = "",
        strength: float = 0.7,
        steps: int = 28,
        guidance_scale: float = 5.0,
        seed: int = config.DEFAULT_SEED,
        scheduler_name: str = "Euler Ancestral",
        offload_encoders: bool = False,
        use_cached_embeds: bool = False,
    ):
        self._interrupt = False
        self.set_scheduler(scheduler_name)
        source_image = source_image.convert("RGB")

        if use_cached_embeds and self._cached_embeds is not None:
            pos_embeds, pos_pooled, neg_embeds, neg_pooled = self._cached_embeds
        else:
            parsed_pos = parse_weighted_prompt(positive_prompt)
            parsed_neg = parse_weighted_prompt(negative_prompt) if negative_prompt else ""
            pos_embeds, pos_pooled = self._build_embeddings(parsed_pos)
            neg_embeds, neg_pooled = self._build_embeddings(parsed_neg if parsed_neg else "")

        encoders_already_offloaded = (
            config.DEVICE == "cuda"
            and next(self.pipe.text_encoder.parameters()).device.type == "cpu"
        )

        if offload_encoders and config.DEVICE == "cuda" and not encoders_already_offloaded:
            self.pipe.text_encoder.to("cpu")
            if self.pipe.text_encoder_2 is not None:
                self.pipe.text_encoder_2.to("cpu")
            self.flush_vram()

        needs_device_patch = offload_encoders or encoders_already_offloaded
        if needs_device_patch and config.DEVICE == "cuda":
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
            pooled_prompt_embeds=pos_pooled,
            negative_pooled_prompt_embeds=neg_pooled,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=self._step_callback,
        )

        image = self.img2img_pipe(**kwargs).images[0]

        if needs_device_patch and config.DEVICE == "cuda":
            type(self.img2img_pipe)._execution_device = property(_orig_exec)
            self.pipe.text_encoder.to(config.DEVICE)
            if self.pipe.text_encoder_2 is not None:
                self.pipe.text_encoder_2.to(config.DEVICE)

        return image

    def inpaint(
        self,
        source_image,
        mask_image,
        positive_prompt: str,
        negative_prompt: str = "",
        strength: float = 0.7,
        steps: int = 28,
        guidance_scale: float = 5.0,
        seed: int = config.DEFAULT_SEED,
        scheduler_name: str = "Euler Ancestral",
    ):
        self._interrupt = False
        self.set_scheduler(scheduler_name)

        source_image = source_image.convert("RGB")
        mask_image = mask_image.convert("L")

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
            pooled_prompt_embeds=pos_pooled,
            negative_pooled_prompt_embeds=neg_pooled,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=self._step_callback,
        )

        image = self.inpaint_pipe(**kwargs).images[0]
        return image
