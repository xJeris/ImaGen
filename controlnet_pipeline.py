"""ControlNet pipeline — wraps an existing base model with ControlNet conditioning.

Supports:
- SDXL / SD 1.5 / Pony / Illustrious via StableDiffusionXLControlNetPipeline
  or StableDiffusionControlNetPipeline
- Flux via FluxControlNetPipeline + FluxControlNetModel
- Krea 2 is NOT supported (no ControlNet pipeline exists in diffusers)

This is NOT a standalone generator. It borrows components (UNet/transformer,
VAE, text encoders) from the loaded base generator to avoid duplicate VRAM
usage. Only the ControlNet model itself is loaded separately.
"""

import gc
import json
from pathlib import Path

import torch
from PIL import Image

import config

CONTROLNET_DIR = config.MODEL_CACHE_DIR / "controlnet"
CONTROLNET_DIR.mkdir(parents=True, exist_ok=True)

# Preprocessor display names → internal keys.
# "None (raw image)" means the user supplies a pre-processed control image.
PREPROCESSORS = {
    "None (raw image)": None,
    "Canny": "canny",
    "Depth (MiDaS)": "depth_midas",
    "Normal Map": "normal_bae",
    "OpenPose": "openpose",
    "Lineart": "lineart",
    "SoftEdge (HED)": "softedge_hed",
    "Scribble": "scribble",
}

PREPROCESSOR_NAMES = list(PREPROCESSORS.keys())

# Cached preprocessor instances (lazy-loaded on first use)
_preprocessor_cache = {}


class ControlNetRunner:
    """Loads a ControlNet model and builds a ControlNet pipeline from an
    existing base generator's components."""

    def __init__(self):
        self.controlnet = None
        self.pipe = None
        self._controlnet_name = None
        self._model_type = None   # "sdxl", "sd15", or "flux"
        self._interrupt = False
        self._progress_callback = None
        self._num_steps = 0

    # ── Model discovery ───────────────────────────────────────

    def get_available_controlnets(self):
        """List ControlNet models in models/controlnet/."""
        CONTROLNET_DIR.mkdir(parents=True, exist_ok=True)
        models = []
        for item in CONTROLNET_DIR.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                models.append(item.name)
            elif item.is_file() and item.suffix == ".safetensors":
                models.append(item.name)
        return sorted(models)

    # ── ControlNet loading ────────────────────────────────────

    # Config repos used by from_single_file when it needs to fetch model
    # architecture info.  These are public HuggingFace repos.
    _SINGLE_FILE_CONFIGS = {
        "sdxl": "diffusers/controlnet-canny-sdxl-1.0",
        "sd15": "lllyasviel/control_v11p_sd15_canny",
    }

    def load_controlnet(self, controlnet_name, model_type="sdxl",
                        progress_callback=None):
        """Load a ControlNet model from models/controlnet/."""
        self.unload()

        cn_path = CONTROLNET_DIR / controlnet_name
        self._model_type = model_type

        if progress_callback:
            progress_callback(f"Loading ControlNet: {controlnet_name}...")

        # For single-file models, validate format before attempting load
        if cn_path.is_file() and cn_path.suffix == ".safetensors":
            self._validate_single_file(cn_path)

        if model_type == "flux":
            from diffusers import FluxControlNetModel
            if cn_path.is_file():
                self.controlnet = FluxControlNetModel.from_single_file(
                    str(cn_path), torch_dtype=torch.bfloat16,
                )
            else:
                self.controlnet = FluxControlNetModel.from_pretrained(
                    str(cn_path), torch_dtype=torch.bfloat16,
                    local_files_only=True,
                )
        else:
            from diffusers import ControlNetModel
            if cn_path.is_file():
                # Try local config cache first, then download config if needed
                cfg = self._SINGLE_FILE_CONFIGS.get(model_type, self._SINGLE_FILE_CONFIGS["sdxl"])
                try:
                    self.controlnet = ControlNetModel.from_single_file(
                        str(cn_path), torch_dtype=config.DTYPE,
                        config=cfg, local_files_only=True,
                    )
                except OSError:
                    # Config not cached locally — download it
                    if progress_callback:
                        progress_callback(f"Downloading ControlNet config from {cfg}...")
                    try:
                        self.controlnet = ControlNetModel.from_single_file(
                            str(cn_path), torch_dtype=config.DTYPE,
                            config=cfg,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to load ControlNet from single file. The config "
                            f"could not be fetched from '{cfg}'.\n\n"
                            f"You can fix this by downloading the model in diffusers "
                            f"format (a folder with config.json) instead of a single "
                            f".safetensors file, and placing it in:\n"
                            f"  {CONTROLNET_DIR}\n\n"
                            f"Original error: {e}"
                        ) from e
            else:
                self.controlnet = ControlNetModel.from_pretrained(
                    str(cn_path), torch_dtype=config.DTYPE,
                    local_files_only=True,
                )

        self.controlnet.to(config.DEVICE)
        self._controlnet_name = controlnet_name

        if progress_callback:
            progress_callback(f"ControlNet ready: {controlnet_name}")

    @staticmethod
    def _validate_single_file(path):
        """Check a .safetensors file for unsupported ControlNet formats."""
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            sample_keys = list(f.keys())[:5]
        if any(k.startswith("lllite_") for k in sample_keys):
            raise RuntimeError(
                f"'{path.name}' is a ControlNet-LLLite model, which is not "
                f"supported by diffusers. LLLite is a lightweight ControlNet "
                f"variant used by ComfyUI.\n\n"
                f"Please download a standard ControlNet model instead "
                f"(diffusers-format folder or standard .safetensors)."
            )

    # ── Pipeline construction ─────────────────────────────────

    def build_pipeline(self, base_generator):
        """Build the ControlNet pipeline using shared components from the
        base generator. This avoids loading the base model twice.

        Must be called after load_controlnet() and after the base generator
        has a model loaded.
        """
        if self.controlnet is None:
            raise RuntimeError("Load a ControlNet model first.")

        base_pipe = base_generator.pipe
        if base_pipe is None:
            raise RuntimeError("Load a base model first.")

        model_type = getattr(base_generator, '_model_type', 'sdxl')
        self._model_type = model_type

        if model_type == "flux":
            from diffusers import FluxControlNetPipeline
            self.pipe = FluxControlNetPipeline(
                transformer=base_pipe.transformer,
                vae=base_pipe.vae,
                text_encoder=base_pipe.text_encoder,
                text_encoder_2=base_pipe.text_encoder_2,
                tokenizer=base_pipe.tokenizer,
                tokenizer_2=base_pipe.tokenizer_2,
                scheduler=base_pipe.scheduler,
                controlnet=self.controlnet,
            )
        elif model_type == "sdxl":
            from diffusers import StableDiffusionXLControlNetPipeline
            self.pipe = StableDiffusionXLControlNetPipeline(
                vae=base_pipe.vae,
                text_encoder=base_pipe.text_encoder,
                text_encoder_2=base_pipe.text_encoder_2,
                tokenizer=base_pipe.tokenizer,
                tokenizer_2=base_pipe.tokenizer_2,
                unet=base_pipe.unet,
                scheduler=base_pipe.scheduler,
                controlnet=self.controlnet,
            )
        else:
            # SD 1.5
            from diffusers import StableDiffusionControlNetPipeline
            self.pipe = StableDiffusionControlNetPipeline(
                vae=base_pipe.vae,
                text_encoder=base_pipe.text_encoder,
                tokenizer=base_pipe.tokenizer,
                unet=base_pipe.unet,
                scheduler=base_pipe.scheduler,
                controlnet=self.controlnet,
                safety_checker=None,
                feature_extractor=None,
            )

    # ── Preprocessing ─────────────────────────────────────────

    def preprocess_image(self, image, preprocessor_key):
        """Apply a ControlNet preprocessor to a source image.

        Args:
            image: PIL Image
            preprocessor_key: internal key from PREPROCESSORS dict, or None

        Returns:
            Processed PIL Image (or the original if preprocessor_key is None)
        """
        if preprocessor_key is None:
            return image

        global _preprocessor_cache

        if preprocessor_key not in _preprocessor_cache:
            _preprocessor_cache[preprocessor_key] = _load_preprocessor(
                preprocessor_key
            )

        processor = _preprocessor_cache[preprocessor_key]
        if processor is None:
            return image

        result = processor(image)
        # Some processors return numpy arrays; convert to PIL
        if not isinstance(result, Image.Image):
            import numpy as np
            if hasattr(result, 'numpy'):
                result = result.numpy()
            result = Image.fromarray(result)
        return result

    # ── Generation ────────────────────────────────────────────

    def generate(
        self,
        control_image,
        positive_prompt,
        negative_prompt="",
        steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024,
        seed=-1,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        guess_mode=False,
    ):
        """Generate an image conditioned on a control image.

        Args:
            control_image: PIL Image (preprocessed control image)
            positive_prompt: text prompt
            negative_prompt: negative prompt text
            steps: diffusion steps
            guidance_scale: CFG scale
            width, height: output resolution
            seed: random seed (-1 for random)
            controlnet_conditioning_scale: ControlNet strength (0-2)
            control_guidance_start: when ControlNet starts (0-1)
            control_guidance_end: when ControlNet ends (0-1)
            guess_mode: if True, ControlNet encoder tries to recognise
                        the content of the input image without the prompt

        Returns:
            PIL Image result
        """
        if self.pipe is None:
            raise RuntimeError("Build the pipeline first (call build_pipeline).")

        self._interrupt = False
        self._num_steps = steps

        # Resize control image to target dimensions
        control_image = control_image.resize((width, height), Image.LANCZOS)

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=config.DEVICE).manual_seed(seed)

        kwargs = dict(
            prompt=positive_prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=control_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            guess_mode=guess_mode,
            callback_on_step_end=self._step_callback,
        )

        result = self.pipe(**kwargs).images[0]
        return result

    # ── Lifecycle ─────────────────────────────────────────────

    def _step_callback(self, pipeline, i, t, callback_kwargs):
        if self._interrupt:
            pipeline._interrupt = True
        if self._progress_callback and self._num_steps > 0:
            self._progress_callback(i + 1, self._num_steps)
        return callback_kwargs

    def interrupt(self):
        self._interrupt = True

    @property
    def was_interrupted(self):
        return self._interrupt

    def unload(self):
        """Free ControlNet VRAM. Does NOT unload the base model."""
        self.pipe = None
        self.controlnet = None
        self._controlnet_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Preprocessor loading ──────────────────────────────────────

def _load_preprocessor(key):
    """Lazy-load a controlnet_aux preprocessor by key.

    Preprocessor models are downloaded from HuggingFace on first use
    (~100-300 MB each). Subsequent calls use the cached model.
    """
    try:
        import controlnet_aux
    except ImportError:
        print("[ControlNet] controlnet_aux not installed. "
              "Run: pip install controlnet_aux opencv-python")
        return None

    loaders = {
        "canny": lambda: controlnet_aux.CannyDetector(),
        "depth_midas": lambda: controlnet_aux.MidasDetector.from_pretrained(
            "lllyasviel/Annotators"),
        "normal_bae": lambda: controlnet_aux.NormalBaeDetector.from_pretrained(
            "lllyasviel/Annotators"),
        "openpose": lambda: controlnet_aux.OpenposeDetector.from_pretrained(
            "lllyasviel/Annotators"),
        "lineart": lambda: controlnet_aux.LineartDetector.from_pretrained(
            "lllyasviel/Annotators"),
        "softedge_hed": lambda: controlnet_aux.HEDdetector.from_pretrained(
            "lllyasviel/Annotators"),
        "scribble": lambda: controlnet_aux.HEDdetector.from_pretrained(
            "lllyasviel/Annotators"),
    }

    loader = loaders.get(key)
    if loader is None:
        return None

    try:
        return loader()
    except Exception as e:
        print(f"[ControlNet] Failed to load preprocessor '{key}': {e}")
        return None
