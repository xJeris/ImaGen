import gc
import json
import re
import warnings

import numpy as np
import torch
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from safetensors import safe_open

import config

# Schedulers compatible with CogVideoX.
COGVIDEOX_SCHEDULER_MAP = {
    "DDIM": CogVideoXDDIMScheduler,
    "DPM++": CogVideoXDPMScheduler,
}
COGVIDEOX_SCHEDULER_NAMES = list(COGVIDEOX_SCHEDULER_MAP.keys())

# Default FPS for CogVideoX video export.
COGVIDEOX_FPS = 8


def _is_cogvideox_model(model_path):
    """Check if a model directory contains a CogVideoX video pipeline."""
    index_file = model_path / "model_index.json"
    if not index_file.exists():
        return False
    try:
        data = json.loads(index_file.read_text(encoding="utf-8"))
        class_name = data.get("_class_name", "")
        return "CogVideoX" in class_name
    except Exception:
        return False


def estimate_cogvideox_vram_gb(num_frames: int, is_2b: bool = True) -> float:
    """Estimate peak VRAM usage in GB for a CogVideoX video generation.

    CogVideoX-2b runs entirely on GPU (~5 GB model weights in fp16).
    CogVideoX-5b uses CPU offload (~10 GB transformer swapped in/out).

    Estimates are empirical baselines for an RTX 4090.
    """
    if is_2b:
        # Model is already on GPU (~5 GB).  Peak = model + diffusion
        # activations + latent tensor.  VAE decode reuses the same VRAM
        # since the model stays loaded throughout.
        base_gb = 5.0
        per_frame_gb = 0.04
        # VAE decode peak (model already loaded, just decode intermediates)
        vae_overhead_gb = 2.0 + num_frames * 0.03
        diffusion_gb = base_gb + num_frames * per_frame_gb
        peak_gb = max(diffusion_gb, base_gb + vae_overhead_gb)
    else:
        # 5B with CPU offload: only one component on GPU at a time.
        base_gb = 11.0
        per_frame_gb = 0.03
        diffusion_gb = base_gb + num_frames * per_frame_gb
        vae_decode_gb = 4.0 + num_frames * 0.04
        peak_gb = max(diffusion_gb, vae_decode_gb + 1.0)

    return round(peak_gb, 1)


class CogVideoXGenerator:
    def __init__(self):
        self.pipe = None
        self._model_name = None
        self._interrupt = False
        self._transformer_keys = None
        self._active_loras = []

    def get_available_video_models(self):
        """List CogVideoX video models in the cogvideox directory."""
        config.COGVIDEOX_DIR.mkdir(parents=True, exist_ok=True)
        models = []
        for item in config.COGVIDEOX_DIR.iterdir():
            if item.is_dir() and _is_cogvideox_model(item):
                models.append(item.name)
        return sorted(models)

    def unload_model(self):
        """Free VRAM by unloading the current video model."""
        self.pipe = None
        self._model_name = None
        self._transformer_keys = None
        self._active_loras = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, model_name, progress_callback=None):
        """Load a CogVideoX model with CPU offload and VAE tiling."""
        if self.pipe is not None:
            self.unload_model()

        local_path = config.COGVIDEOX_DIR / model_name

        # CogVideoX-2b was trained in float16; 5b models use bfloat16.
        is_2b = "2b" in model_name.lower()
        dtype = torch.float16 if is_2b else torch.bfloat16
        dtype_label = "FP16" if is_2b else "BF16"

        if progress_callback:
            progress_callback(f"Loading {model_name} (CogVideoX {dtype_label})...")

        self.pipe = CogVideoXPipeline.from_pretrained(
            str(local_path),
            torch_dtype=dtype,
            local_files_only=True,
        )

        # CPU offload for diffusion (transformer + text encoder).
        self.pipe.enable_sequential_cpu_offload()

        self.pipe.vae.enable_slicing()

        # Cache transformer key names for LoRA compatibility checks.
        self._transformer_keys = set(
            n for n, _ in self.pipe.transformer.named_parameters()
        )

        self._model_name = model_name
        if progress_callback:
            progress_callback(f"Ready — {model_name}")

    def _check_lora_compatible(self, lora_path: str) -> bool:
        """Check if a LoRA file is compatible with the loaded CogVideoX model."""
        if self._transformer_keys is None:
            return False

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
        """List compatible LoRA files for the loaded CogVideoX model."""
        config.COGVIDEOX_LORA_DIR.mkdir(parents=True, exist_ok=True)
        loras = []
        for f in config.COGVIDEOX_LORA_DIR.iterdir():
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
        self.pipe.to(dtype=self.pipe.dtype)
        self._active_loras = list(lora_list)

    def unload_loras(self):
        """Remove all active LoRAs."""
        if self._active_loras:
            self.pipe.unfuse_lora()
            self.pipe.unload_lora_weights()
            self._active_loras = []

    def set_scheduler(self, scheduler_name: str):
        """Switch the pipeline's scheduler by name."""
        if self.pipe is None:
            return
        cls = COGVIDEOX_SCHEDULER_MAP.get(scheduler_name)
        if cls is None:
            return
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not expected.*")
            self.pipe.scheduler = cls.from_config(self.pipe.scheduler.config)

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

    def generate_latents(
        self,
        positive_prompt: str,
        negative_prompt: str = "",
        num_frames: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        seed: int = -1,
        scheduler_name: str = "DDIM",
        progress_callback=None,
    ):
        """Run diffusion + manual VAE decode and return PIL frames.

        The pipeline runs diffusion in fp16 with sequential CPU offload,
        then decodes latents through the VAE manually in float32 on GPU
        with spatial tiling.  This two-phase approach works around:
          1. Accelerate hook dtype mismatches (hooks don't cast inputs)
          2. fp16 precision loss in 3D convolutions (dark/red frames)
          3. cuDNN conv3d hangs on Windows with large temporal tensors

        Returns list of PIL frames on success, or None if interrupted.
        """
        self._interrupt = False
        self.set_scheduler(scheduler_name)

        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        width, height = 720, 480

        if progress_callback:
            progress_callback("Running diffusion...")

        try:
            output = self.pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                width=width,
                height=height,
                callback_on_step_end=self._step_callback,
                output_type="latent",
            )
        except self._Interrupted:
            self._flush_vram()
            return None
        except Exception as e:
            import traceback
            print(f"\n[CogVideoX] Pipeline error: {e}")
            traceback.print_exc()
            self._flush_vram()
            raise

        latents = output.frames

        if progress_callback:
            progress_callback("Decoding VAE...")

        # --- Manual VAE decode in float32 on GPU with tiling ---
        from accelerate.hooks import remove_hook_from_module

        vae = self.pipe.vae
        remove_hook_from_module(vae, recurse=True)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Casting directly with.*")
            vae.to(device="cuda", dtype=torch.float32)

        vae.enable_tiling()

        # Permute [B,T,C,H,W] -> [B,C,T,H,W], apply scaling, cast to fp32
        latents = latents.permute(0, 2, 1, 3, 4)
        latents = (1 / self.pipe.vae_scaling_factor_image) * latents
        latents = latents.to(device="cuda", dtype=torch.float32)

        with torch.inference_mode():
            video = vae.decode(latents, return_dict=False)[0]

        # Move VAE back to CPU and free VRAM
        vae.to(device="cpu")
        del latents
        self._flush_vram()

        frames = self.pipe.video_processor.postprocess_video(video=video, output_type="pil")
        if frames and isinstance(frames[0], list):
            frames = frames[0]

        del video
        self._flush_vram()

        return frames

    def _flush_vram(self):
        """Free cached VRAM after interruption."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def export_video(frames, output_path: str, fps: int = COGVIDEOX_FPS):
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
