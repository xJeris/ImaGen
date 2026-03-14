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

# Inference defaults
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_SEED = -1

# Prompt defaults (loaded from text files)
_pos_file = PROJECT_ROOT / "default_positive.txt"
_neg_file = PROJECT_ROOT / "default_negative.txt"
DEFAULT_POSITIVE = _pos_file.read_text(encoding="utf-8").strip() if _pos_file.exists() else ""
DEFAULT_NEGATIVE = _neg_file.read_text(encoding="utf-8").strip() if _neg_file.exists() else ""

# Training defaults
LORA_RANK = 4
TRAINING_STEPS = 500
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 1
