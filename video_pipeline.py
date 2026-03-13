import gc
import json
import re

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
from diffusers import BitsAndBytesConfig
from diffusers import (
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
from safetensors import safe_open

import config

# Schedulers compatible with WAN's flow-matching prediction type.
# Each must be instantiated with the original scheduler's config to inherit
# flow_shift, prediction_type, etc.
VIDEO_SCHEDULER_MAP = {
    "UniPC": UniPCMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
}
VIDEO_SCHEDULER_NAMES = list(VIDEO_SCHEDULER_MAP.keys())

# WAN 2.1 generates at 16 fps.  Frame counts must be 4k+1.
WAN_FPS = 16
DURATION_TO_FRAMES = {
    1: 17,
    2: 33,
    3: 49,
    4: 65,
    5: 81,
}


def _is_wan_model(model_path):
    """Check if a model directory contains a WAN video pipeline."""
    index_file = model_path / "model_index.json"
    if not index_file.exists():
        return False
    try:
        data = json.loads(index_file.read_text(encoding="utf-8"))
        class_name = data.get("_class_name", "")
        return "Wan" in class_name
    except Exception:
        return False


class VideoGenerator:
    def __init__(self):
        self.pipe = None
        self._model_name = None
        self._interrupt = False
        self._transformer_keys = None  # cached for LoRA compat checks
        self._active_lora = None

    def get_available_video_models(self):
        """List WAN video models in models/ directory."""
        config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        models = []
        for item in config.MODEL_CACHE_DIR.iterdir():
            if item.is_dir() and _is_wan_model(item):
                models.append(item.name)
        return sorted(models)

    def unload_model(self):
        """Free VRAM by unloading the current video model."""
        self.pipe = None
        self._model_name = None
        self._transformer_keys = None
        self._active_lora = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, model_name, progress_callback=None):
        """Stable load for 4090: Simple Lite path and 4-bit Full path."""
        if self.pipe is not None:
            self.unload_model()

        local_path = config.MODEL_CACHE_DIR / model_name
        is_lite = "1.3B" in model_name

        if progress_callback:
            mode_text = "Standard BF16" if is_lite else "4-bit Optimized"
            progress_callback(f"Loading {model_name} ({mode_text})...")

        if is_lite:
            # --- 1.3B LITE PATH ---
            # Small enough to fit entirely on GPU in bfloat16 (~5GB VRAM).
            # bfloat16 avoids float16 precision issues with WAN's 3D convolutions on Windows.
            self.pipe = WanPipeline.from_pretrained(
                str(local_path),
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            self.pipe.to("cuda")

        else:
            # --- 14B FULL PATH (4-bit NF4 quantization) ---
            # Quantize the transformer from ~28GB to ~7GB so it fits on a 4090.
            # Use enable_model_cpu_offload() so diffusers automatically moves each
            # component to GPU only when needed — the transformer gets offloaded
            # before VAE decode runs, freeing VRAM for decoding.
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            transformer = WanTransformer3DModel.from_pretrained(
                str(local_path),
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )

            vae = AutoencoderKLWan.from_pretrained(
                str(local_path),
                subfolder="vae",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )

            self.pipe = WanPipeline.from_pretrained(
                str(local_path),
                transformer=transformer,
                vae=vae,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            # Sequential CPU offload: each component moves to GPU only when active.
            # With 4-bit quantization, diffusion steps are still fast (~7GB on GPU),
            # and the transformer is offloaded before VAE decode so decode doesn't OOM.
            self.pipe.enable_model_cpu_offload()

        # Enable VAE slicing if supported (reduces peak VRAM during decode).
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()

        # Cache transformer key names for LoRA compatibility checks.
        self._transformer_keys = set(
            n for n, _ in self.pipe.transformer.named_parameters()
        )

        self._model_name = model_name
        if progress_callback:
            progress_callback(f"Ready — {model_name}")

    def _check_lora_compatible(self, lora_path: str) -> bool:
        """Check if a LoRA file is compatible with the loaded video model."""
        if self._transformer_keys is None:
            return True

        try:
            with safe_open(lora_path, framework="pt") as f:
                lora_keys = list(f.keys())
        except Exception:
            return True

        if not lora_keys:
            return True

        transformer_lora_keys = [
            k for k in lora_keys
            if "transformer" in k.lower() or "lora_unet_" in k
        ]
        if not transformer_lora_keys:
            return True

        base_names = set()
        for k in transformer_lora_keys:
            base = re.sub(r'\.(lora_A|lora_B|lora_down|lora_up|alpha)\b.*', '', k)
            if base.startswith("transformer."):
                base = base[len("transformer."):]
            base_names.add(base + ".weight")

        if not base_names:
            return True

        matches = sum(1 for name in base_names if name in self._transformer_keys)
        ratio = matches / len(base_names)
        return ratio > 0.3

    def get_available_loras(self):
        """List compatible LoRA files for the loaded video model."""
        config.LORA_DIR.mkdir(parents=True, exist_ok=True)
        loras = []
        for f in config.LORA_DIR.iterdir():
            if f.suffix == ".safetensors":
                if self._check_lora_compatible(str(f)):
                    loras.append(f.name)
        return sorted(loras)

    def load_lora(self, lora_path: str, weight: float = 1.0):
        """Load and fuse a LoRA."""
        from pathlib import Path
        if self._active_lora:
            self.unload_lora()
        p = Path(lora_path)
        self.pipe.load_lora_weights(str(p.parent), weight_name=p.name)
        self.pipe.fuse_lora(lora_scale=weight)
        self._active_lora = lora_path

    def unload_lora(self):
        """Remove currently active LoRA."""
        if self._active_lora:
            self.pipe.unfuse_lora()
            self.pipe.unload_lora_weights()
            self._active_lora = None

    def set_scheduler(self, scheduler_name: str):
        """Switch the pipeline's scheduler by name."""
        if self.pipe is None:
            return
        cls = VIDEO_SCHEDULER_MAP.get(scheduler_name)
        if cls is None:
            return
        # from_config inherits flow_shift, prediction_type, etc. from the original
        self.pipe.scheduler = cls.from_config(self.pipe.scheduler.config)

    def interrupt(self):
        """Signal the pipeline to stop after the current step."""
        self._interrupt = True

    def _step_callback(self, pipeline, i, t, callback_kwargs):
        """Check interrupt flag at each diffusion step."""
        if self._interrupt:
            pipeline._interrupt = True
        return callback_kwargs

    def generate_video(
        self,
        positive_prompt: str,
        negative_prompt: str = "",
        num_frames: int = 49,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = -1,
        scheduler_name: str = "UniPC",
    ):
        """Generate a video from text prompts. Returns a list of PIL frames."""
        self._interrupt = False
        self.set_scheduler(scheduler_name)

        generator = None
        if seed >= 0:
            # Note: keeping "cpu" for generator is fine/stable for seed consistency
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # --- RESOLUTION ---
        # Use 480p for both models on a 4090 to keep VRAM usage safe.
        # 14B *can* do 720p but the VAE decode often OOMs at 24GB.
        width, height = 832, 480

        kwargs = dict(
            prompt=positive_prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,    # Added automatic width
            height=height,  # Added automatic height
            callback_on_step_end=self._step_callback,
        )

        # This calls the underlying Diffusers pipeline
        output = self.pipe(**kwargs)
        
        # output.frames is usually a list of lists: [[PIL, PIL, PIL]]
        frames = output.frames[0]
        return frames

    @staticmethod
    def export_video(frames, output_path: str, fps: int = WAN_FPS):
        """Write a list of PIL frames to an MP4 file."""
        import imageio

        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
        for frame in frames:
            arr = np.asarray(frame)
            # Convert float [0,1] to uint8 [0,255] if needed
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            writer.append_data(arr)
        writer.close()
        return output_path