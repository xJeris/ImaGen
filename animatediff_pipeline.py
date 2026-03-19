import gc
import json
import re
import warnings

import numpy as np
import torch
from diffusers import AnimateDiffSparseControlNetPipeline
from diffusers.models import MotionAdapter, SparseControlNetModel
from diffusers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
)
from safetensors import safe_open

import config


# Schedulers compatible with AnimateDiff (SD 1.5 based).
ANIMATEDIFF_SCHEDULER_MAP = {
    "Euler": (EulerDiscreteScheduler, {}),
    "Euler Ancestral": (EulerAncestralDiscreteScheduler, {}),
    "DPM++ 2M Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "final_sigmas_type": "sigma_min"}),
    "DDIM": (DDIMScheduler, {}),
    "UniPC": (UniPCMultistepScheduler, {}),
}
ANIMATEDIFF_SCHEDULER_NAMES = list(ANIMATEDIFF_SCHEDULER_MAP.keys())

# AnimateDiff FPS defaults.
ANIMATEDIFF_FPS = 12
ANIMATEDIFF_MIN_FPS = 6

# AnimateDiff was trained on 16-frame contexts. Going beyond 16 causes noise
# because the positional embeddings at positions 16-31 were never trained.
# Other tools (ComfyUI, A1111) enforce 16-frame context and use sliding window
# overlap for longer videos.
ANIMATEDIFF_MAX_CONTEXT = 16


def estimate_animatediff_vram_gb(num_frames: int, width: int = 512, height: int = 512) -> float:
    """Estimate peak VRAM usage in GB for AnimateDiff generation.

    AnimateDiff uses an SD 1.5 UNet (~3.4 GB fp16) plus a motion adapter
    (~0.4 GB) and SparseControlNet (~0.5 GB), all loaded on GPU.
    The latent tensor scales with frame count, and the UNet processes all
    frames in a batch.

    Empirical baselines measured on an RTX 4090 at 512×512:
      - Base (model weights): ~4.5 GB
      - Per-frame overhead: ~0.12 GB (latents + UNet activations per frame)
    """
    base_gb = 4.5
    per_frame_gb = 0.12
    peak_gb = base_gb + num_frames * per_frame_gb
    return round(peak_gb, 1)


class AnimateDiffGenerator:
    """AnimateDiff image-to-video pipeline using SparseCtrl.

    Directory layout expected under ``config.ANIMATEDIFF_DIR``::

        models/animatediff/
        ├── base_model/          # SD 1.5 diffusers model folder
        ├── motion_adapter/      # MotionAdapter folder (e.g. v1-5-3)
        └── sparsectrl/          # SparseControlNetModel folder
    """

    def __init__(self):
        self.pipe = None
        self._model_name = None
        self._interrupt = False
        self._unet_keys = None
        self._active_loras = []

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _is_sd15_model(path):
        """Check if a directory is an SD 1.5 (non-SDXL) base model."""
        index = path / "model_index.json"
        if not index.exists():
            return False
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            class_name = data.get("_class_name", "")
            # Must be a StableDiffusion pipeline but NOT XL and NOT Wan
            return (
                "StableDiffusion" in class_name
                and "XL" not in class_name
                and "Wan" not in class_name
            )
        except Exception:
            return False

    def get_available_base_models(self):
        """List SD 1.5 base models from both models/animatediff/ and models/.

        Scans the dedicated animatediff folder first, then the main models
        folder so users can reuse existing SD 1.5 models without copying them.
        Models from the main folder are shown with a ``(shared)`` suffix.
        """
        seen = set()
        models = []

        # 1. Dedicated animatediff folder
        ad_dir = config.ANIMATEDIFF_DIR
        ad_dir.mkdir(parents=True, exist_ok=True)
        for item in ad_dir.iterdir():
            if item.is_dir() and self._is_sd15_model(item):
                models.append(item.name)
                seen.add(item.name)

        # 2. Main models folder (skip animatediff subfolder itself)
        for item in config.MODEL_CACHE_DIR.iterdir():
            if item.name == "animatediff":
                continue
            if item.is_dir() and item.name not in seen and self._is_sd15_model(item):
                models.append(item.name)

        return sorted(models)

    def get_available_motion_adapters(self):
        """List motion adapter folders inside the animatediff directory."""
        ad_dir = config.ANIMATEDIFF_DIR
        if not ad_dir.exists():
            return []
        adapters = []
        for item in ad_dir.iterdir():
            if not item.is_dir():
                continue
            # Motion adapters have a diffusion_pytorch_model file but no model_index.json
            # OR they have a model_index.json with MotionAdapter class
            index = item / "model_index.json"
            if index.exists():
                try:
                    data = json.loads(index.read_text(encoding="utf-8"))
                    if "MotionAdapter" in data.get("_class_name", ""):
                        adapters.append(item.name)
                        continue
                except Exception:
                    pass
            # Fallback: folder contains a config.json with MotionAdapter class
            # but no model_index.json — common for HuggingFace MotionAdapter downloads
            config_file = item / "config.json"
            if config_file.exists() and not index.exists():
                try:
                    data = json.loads(config_file.read_text(encoding="utf-8"))
                    if data.get("_class_name", "") == "MotionAdapter":
                        adapters.append(item.name)
                except Exception:
                    pass
        return sorted(adapters)

    def get_available_sparsectrls(self):
        """List SparseControlNet model folders inside the animatediff directory."""
        ad_dir = config.ANIMATEDIFF_DIR
        if not ad_dir.exists():
            return []
        ctrls = []
        for item in ad_dir.iterdir():
            if not item.is_dir():
                continue
            config_file = item / "config.json"
            if not config_file.exists():
                continue
            try:
                data = json.loads(config_file.read_text(encoding="utf-8"))
                class_name = data.get("_class_name", "")
                if "SparseControlNet" in class_name:
                    ctrls.append(item.name)
            except Exception:
                pass
        return sorted(ctrls)

    # ------------------------------------------------------------------
    # Model loading / unloading
    # ------------------------------------------------------------------

    def unload_model(self):
        """Free VRAM by unloading the current pipeline."""
        self.pipe = None
        self._model_name = None
        self._unet_keys = None
        self._active_loras = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(
        self,
        base_model_name,
        motion_adapter_name,
        sparsectrl_name,
        progress_callback=None,
    ):
        """Load the AnimateDiff SparseCtrl pipeline.

        The base model is resolved from ``models/animatediff/`` first, then
        falls back to the main ``models/`` folder so users can reuse existing
        SD 1.5 models without copying.  Motion adapter and SparseControlNet
        are always loaded from ``models/animatediff/``.
        """
        if self.pipe is not None:
            self.unload_model()

        ad_dir = config.ANIMATEDIFF_DIR

        # Resolve base model: check animatediff folder first, then main models
        base_path = ad_dir / base_model_name
        if not base_path.exists():
            base_path = config.MODEL_CACHE_DIR / base_model_name
        if not base_path.exists():
            raise FileNotFoundError(f"Base model not found: {base_model_name}")

        adapter_path = ad_dir / motion_adapter_name
        ctrl_path = ad_dir / sparsectrl_name

        if progress_callback:
            progress_callback(f"Loading motion adapter: {motion_adapter_name}...")

        motion_adapter = MotionAdapter.from_pretrained(
            str(adapter_path),
            torch_dtype=torch.float16,
            local_files_only=True,
        )

        if progress_callback:
            progress_callback(f"Loading SparseControlNet: {sparsectrl_name}...")

        controlnet = SparseControlNetModel.from_pretrained(
            str(ctrl_path),
            torch_dtype=torch.float16,
            local_files_only=True,
        )

        if progress_callback:
            progress_callback(f"Loading base model: {base_model_name}...")

        self.pipe = AnimateDiffSparseControlNetPipeline.from_pretrained(
            str(base_path),
            motion_adapter=motion_adapter,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            local_files_only=True,
        )

        self.pipe.to(config.DEVICE)

        # Enable VAE slicing to reduce peak VRAM during decode
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

        if config.DEVICE == "cuda":
            # xformers flash attention is incompatible with AnimateDiff's
            # temporal attention patterns — use SDPA (PyTorch native) instead.
            from diffusers.models.attention_processor import AttnProcessor2_0
            self.pipe.unet.set_attn_processor(AttnProcessor2_0())

        # Cache UNet keys for LoRA compatibility checks
        self._unet_keys = set(self.pipe.unet.state_dict().keys())
        self._model_name = base_model_name

        if progress_callback:
            progress_callback(f"Ready — AnimateDiff ({base_model_name})")

    # ------------------------------------------------------------------
    # LoRA support
    # ------------------------------------------------------------------

    def _check_lora_compatible(self, lora_path: str) -> bool:
        """Check if a LoRA is compatible with the loaded SD 1.5 UNet."""
        if self._unet_keys is None:
            return False

        try:
            with safe_open(lora_path, framework="pt") as f:
                lora_keys = list(f.keys())
        except Exception:
            return True

        if not lora_keys:
            return True

        unet_lora_keys = [
            k for k in lora_keys
            if k.startswith("unet.") or k.startswith("lora_unet_")
        ]
        if not unet_lora_keys:
            return True

        base_names = set()
        for k in unet_lora_keys:
            base = re.sub(r'\.(lora_A|lora_B|lora_down|lora_up|alpha)\b.*', '', k)
            if base.startswith("unet."):
                base = base[5:]
            elif base.startswith("lora_unet_"):
                base = base[10:].replace("_", ".")
            base_names.add(base + ".weight")

        if not base_names:
            return True

        matches = sum(1 for name in base_names if name in self._unet_keys)
        ratio = matches / len(base_names)
        return ratio > 0.3

    def get_available_loras(self):
        """List compatible LoRA files for the loaded model."""
        config.LORA_DIR.mkdir(parents=True, exist_ok=True)
        loras = []
        for f in config.LORA_DIR.iterdir():
            if f.suffix == ".safetensors":
                if self._check_lora_compatible(str(f)):
                    loras.append(f.name)
        return sorted(loras)

    def load_loras(self, lora_list):
        """Load and fuse one or more LoRAs.

        Args:
            lora_list: list of (path, weight) tuples.
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
        # Fusing LoRAs can upcast weights to float32; cast everything back
        self.pipe.to(dtype=torch.float16)
        self._active_loras = list(lora_list)

    def unload_loras(self):
        """Remove all active LoRAs."""
        if self._active_loras:
            self.pipe.unfuse_lora()
            self.pipe.unload_lora_weights()
            self._active_loras = []

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def set_scheduler(self, scheduler_name: str):
        """Switch the pipeline's scheduler by name."""
        if self.pipe is None:
            return
        entry = ANIMATEDIFF_SCHEDULER_MAP.get(scheduler_name)
        if entry is None:
            return
        cls, kwargs = entry
        # AnimateDiff requires linear beta schedule — the base SD 1.5 model
        # ships with scaled_linear which causes noise/artifacts in animations.
        self.pipe.scheduler = cls.from_config(
            self.pipe.scheduler.config,
            beta_schedule="linear",
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Interrupt support
    # ------------------------------------------------------------------

    def interrupt(self):
        """Signal the pipeline to stop after the current step."""
        self._interrupt = True

    @property
    def was_interrupted(self):
        """Check whether the last generation was interrupted."""
        return self._interrupt

    class _Interrupted(Exception):
        """Raised inside the callback to immediately abort generation."""
        pass

    def _step_callback(self, pipeline, i, t, callback_kwargs):
        """Check interrupt flag at each diffusion step."""
        if self._interrupt:
            raise self._Interrupted()
        return callback_kwargs

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_latents(
        self,
        source_image,
        positive_prompt: str,
        negative_prompt: str = "",
        num_frames: int = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: int = -1,
        scheduler_name: str = "DPM++ 2M Karras",
    ):
        """Run diffusion steps and return raw latents (no VAE decode).

        Returns latents tensor on success, or None if interrupted.
        """
        self._interrupt = False
        self.set_scheduler(scheduler_name)

        # Ensure source image is the right size for SD 1.5
        source_image = source_image.convert("RGB")
        w, h = source_image.size
        # Round down to nearest multiple of 8
        w = (w // 8) * 8
        h = (h // 8) * 8
        # Clamp to reasonable range for SD 1.5
        # SD 1.5 was trained at 512x512 — larger sizes cause massive slowdowns
        # (768x768 is 2.25x more pixels = 2.25x slower per step)
        w = max(256, min(w, 512))
        h = max(256, min(h, 512))
        source_image = source_image.resize((w, h))

        generator = None
        if seed >= 0:
            gen_device = config.DEVICE if config.DEVICE == "cuda" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        try:
            output = self.pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                conditioning_frames=source_image,
                controlnet_frame_indices=[0],
                controlnet_conditioning_scale=float(controlnet_conditioning_scale),
                generator=generator,
                width=w,
                height=h,
                callback_on_step_end=self._step_callback,
                output_type="latent",
            )
        except self._Interrupted:
            self._flush_vram()
            return None

        return output.frames

    def decode_latents(self, latents):
        """Decode latents through the VAE and return a list of PIL frames."""
        video_tensor = self.pipe.decode_latents(latents)
        frames = self.pipe.video_processor.postprocess_video(video=video_tensor, output_type="pil")
        if frames and isinstance(frames[0], list):
            frames = frames[0]
        return frames

    def _flush_vram(self):
        """Free cached VRAM after interruption."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def export_video(frames, output_path: str, fps: int = ANIMATEDIFF_FPS):
        """Write a list of PIL frames to an MP4 file."""
        import imageio

        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
        for frame in frames:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            writer.append_data(arr)
        writer.close()
        return output_path
