import gc
import json
import re
import warnings

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
from diffusers import BitsAndBytesConfig
from diffusers import (
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
)
from safetensors import safe_open

import config

# Schedulers compatible with WAN's flow-matching prediction type.
# Each must be instantiated with the original scheduler's config to inherit
# flow_shift, prediction_type, etc.
VIDEO_SCHEDULER_MAP = {
    "UniPC": UniPCMultistepScheduler,
    "Euler": FlowMatchEulerDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
}
VIDEO_SCHEDULER_NAMES = list(VIDEO_SCHEDULER_MAP.keys())

# Default FPS for WAN video export.
WAN_FPS = 24
MIN_FPS = 6
MAX_FPS = 30


def estimate_video_vram_gb(num_frames: int, width: int = 832, height: int = 480, is_lite: bool = True) -> float:
    """Estimate peak VRAM usage in GB for a WAN video generation.

    WAN's VAE compresses 8× spatially and 4× temporally. The dominant VRAM
    consumers during diffusion are the latent tensor and the transformer
    activations.  During VAE decode, the full-resolution frame tensor is the
    bottleneck.

    The estimates below are empirical baselines measured on an RTX 4090 with
    a linear per-frame overhead added.  They are *approximations* — actual
    usage can vary with scheduler, guidance scale, and CUDA allocator
    fragmentation.
    """
    # --- Diffusion pass (transformer) ---
    # Base VRAM: model weights sitting on GPU.
    #   1.3B bf16 ≈ 5.0 GB,  14B 4-bit + CPU offload ≈ 7.0 GB
    # Per-frame overhead: latent channels × spatial dims × bytes, plus
    #   intermediate activations.  Empirically ~0.06 GB/frame for 1.3B
    #   and ~0.05 GB/frame for 14B (offload keeps transformer on CPU between
    #   steps, so only latents + one active component sit on GPU).
    if is_lite:
        base_gb = 5.0
        per_frame_gb = 0.06
    else:
        base_gb = 7.0
        per_frame_gb = 0.05

    diffusion_gb = base_gb + num_frames * per_frame_gb

    # --- VAE decode pass ---
    # The VAE decodes the full latent volume into pixel frames.
    # With VAE slicing enabled the peak is lower, but it still scales
    # roughly with the number of frames.
    # Empirical: ~2.5 GB base + ~0.04 GB/frame (with slicing).
    vae_decode_gb = 2.5 + num_frames * 0.04

    # For 1.3B everything is on GPU so diffusion + VAE overlap a bit;
    # peak ≈ max(diffusion, model_base + vae_decode).
    # For 14B with CPU offload the transformer is offloaded before VAE
    # runs, so peak ≈ max(diffusion, vae_decode + ~1 GB scheduler state).
    if is_lite:
        peak_gb = max(diffusion_gb, base_gb + vae_decode_gb)
    else:
        peak_gb = max(diffusion_gb, vae_decode_gb + 1.0)

    return round(peak_gb, 1)


def get_available_vram_gb() -> float | None:
    """Return free VRAM in GB, or None if no CUDA GPU."""
    if not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info()
    return round(free / (1024 ** 3), 1)


def get_total_vram_gb() -> float | None:
    """Return total VRAM in GB, or None if no CUDA GPU."""
    if not torch.cuda.is_available():
        return None
    _, total = torch.cuda.mem_get_info()
    return round(total / (1024 ** 3), 1)


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
        self._active_loras = []

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
        self._active_loras = []
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
            self.pipe.to(config.DEVICE)

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
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

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
        """List compatible LoRA files for the loaded video model."""
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
        # Fusing LoRAs can upcast weights to float32; cast back to bfloat16
        # (WAN uses bfloat16 — float16 causes precision issues with 3D convolutions)
        self.pipe.to(dtype=torch.bfloat16)
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
        cls = VIDEO_SCHEDULER_MAP.get(scheduler_name)
        if cls is None:
            return
        # from_config inherits flow_shift, prediction_type, etc. from the original
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
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = -1,
        scheduler_name: str = "UniPC",
    ):
        """Run diffusion steps and return raw latents (no VAE decode).

        Returns latents tensor on success, or None if interrupted.
        """
        self._interrupt = False
        self.set_scheduler(scheduler_name)

        generator = None
        if seed >= 0:
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
            width=width,
            height=height,
            callback_on_step_end=self._step_callback,
            output_type="latent",
        )

        try:
            output = self.pipe(**kwargs)
        except self._Interrupted:
            self._flush_vram()
            return None

        # output.frames contains raw latents when output_type="latent"
        return output.frames

    def decode_latents(self, latents):
        """Decode latents through the VAE and return a list of PIL frames."""
        latents = latents.to(self.pipe.vae.dtype)

        # Denormalize latents using VAE config (required by WAN pipeline)
        latents_mean = (
            torch.tensor(self.pipe.vae.config.latents_mean)
            .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0 / torch.tensor(self.pipe.vae.config.latents_std)
            .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean

        video = self.pipe.vae.decode(latents, return_dict=False)[0]
        frames = self.pipe.video_processor.postprocess_video(video, output_type="pil")
        # postprocess returns list of lists: [[PIL, PIL, ...]]
        if frames and isinstance(frames[0], list):
            frames = frames[0]
        return frames

    def _flush_vram(self):
        """Free cached VRAM after interruption."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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