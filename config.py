import os
from pathlib import Path

import torch

# Reduce CUDA memory fragmentation (must be set before any CUDA calls)
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "garbage_collection_threshold:0.9,max_split_size_mb:512",
)

# Paths
PROJECT_ROOT = Path(__file__).parent
MODEL_CACHE_DIR = PROJECT_ROOT / "models"
LORA_DIR = PROJECT_ROOT / "loras"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPSCALER_DIR = PROJECT_ROOT / "upscalers"
ANIMATEDIFF_DIR = PROJECT_ROOT / "models" / "animatediff"

# Default model (used on first run to download)
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_MODEL_NAME = "sdxl-base"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# CUDA performance: enable TF32 tensor cores and cuDNN autotuner.
# TF32 uses the 4090's tensor cores for ~3-5x faster matmul/conv at fp32
# with negligible precision loss. cuDNN benchmark auto-selects the fastest
# kernel for each conv shape (small one-time cost on first run).
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Inference defaults
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_SEED = -1

# Prompt defaults (loaded from text files)
_pos_file = PROJECT_ROOT / "default_positive.txt"
_neg_file = PROJECT_ROOT / "default_negative.txt"
DEFAULT_POSITIVE = " ".join(_pos_file.read_text(encoding="utf-8").split()) if _pos_file.exists() else ""
DEFAULT_NEGATIVE = " ".join(_neg_file.read_text(encoding="utf-8").split()) if _neg_file.exists() else ""

# Training defaults
LORA_RANK = 4
TRAINING_STEPS = 500
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 1

# ── Multi-architecture support ──
ARCHITECTURES = ["SDXL / SD 1.5", "Pony", "Illustrious", "Flux"]

ARCH_MODEL_DIRS = {
    "SDXL / SD 1.5": MODEL_CACHE_DIR,
    "Pony": MODEL_CACHE_DIR / "pony",
    "Illustrious": MODEL_CACHE_DIR / "illustrious",
    "Flux": MODEL_CACHE_DIR / "flux",
}

ARCH_LORA_DIRS = {
    "SDXL / SD 1.5": LORA_DIR,
    "Pony": LORA_DIR / "pony",
    "Illustrious": LORA_DIR / "illustrious",
    "Flux": LORA_DIR / "flux",
}

ARCH_DEFAULTS = {
    "SDXL / SD 1.5": {
        "steps": 30,
        "guidance_scale": 7.5,
        "width": 1024,
        "height": 1024,
        "scheduler": "Euler",
    },
    "Pony": {
        "steps": 25,
        "guidance_scale": 3.0,
        "width": 1024,
        "height": 1024,
        "scheduler": "Euler Ancestral",
    },
    "Illustrious": {
        "steps": 28,
        "guidance_scale": 5.0,
        "width": 1024,
        "height": 1024,
        "scheduler": "Euler Ancestral",
    },
    "Flux": {
        "steps": 20,
        "guidance_scale": 3.5,
        "width": 512,
        "height": 512,
        "scheduler": "Euler",
    },
}

# Auto-create architecture subdirectories
for _d in list(ARCH_MODEL_DIRS.values()) + list(ARCH_LORA_DIRS.values()):
    _d.mkdir(parents=True, exist_ok=True)
