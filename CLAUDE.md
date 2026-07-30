# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Description

ImaGen is a fully self-contained, offline-capable AI image and video generation application. It provides text-to-image, image-to-image, inpainting, text-to-video, and image animation through a Gradio web UI. It does not access the internet or external sources after initial model download.

## End User Needs

- Train the model on existing images (LoRA fine-tuning)
- Interface for positive/negative prompts and text descriptions
- Preview and gallery of created images
- Download/save images as PNG files
- Weighted prompt syntax: `[green curtains:1.5]`
- Upscalers, Hires Fix (two-pass generation), and LoRA adapters

## Development Commands

```bash
# Setup (Python 3.12 required, NVIDIA GPU recommended)
py -3.12 -m venv venv
source venv/Scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Run the application (launches Gradio on http://127.0.0.1:7860)
python app.py

# Windows quick-start
start.bat
```

There is no test suite, linter configuration, or build step. The application is run directly with `python app.py`.

## Architecture

The application is ~5,000 lines of Python organized as follows:

### Entry Point & UI
- **app.py** (2,440 lines) — Gradio web UI with tabs: Text-to-Image, Image-to-Image, Image Animation, Text-to-Video, Model Browser, Preview Files, LoRA Training. All generation functions are wired here.

### Generation Pipelines
- **pipeline.py** — `ImageGenerator` class. Loads SDXL or SD 1.5 models via HuggingFace Diffusers. Handles txt2img, img2img, inpainting. Uses Compel for weighted prompt parsing. Supports hot-swap model loading with VRAM cleanup.
- **video_pipeline.py** — `VideoGenerator` class for WAN 2.1 models (1.3B lite, 14B full with 4-bit quantization). Single-pass diffusion for temporal coherence.
- **cogvideox_pipeline.py** — `CogVideoXGenerator` class for CogVideoX models (2b, 5b). Two-phase generation: diffusion in fp16 with sequential CPU offload, then manual VAE decode in float32 on GPU with spatial tiling. Works around cuDNN conv3d hangs on Windows.
- **video_chunker.py** — Chunked VAE decoding for video frames to stay within VRAM limits. Routes CogVideoX through single-pass pipeline path.
- **animatediff_pipeline.py** — AnimateDiff + SparseCtrl for SD 1.5 image animation (max 16 frames).

### Utilities
- **upscaler.py** — AI upscaling via Spandrel (Real-ESRGAN, SwinIR, ESRGAN model formats).
- **training.py** — LoRA fine-tuning for SDXL using PEFT library.
- **civitai_browser.py** — CivitAI model search and download integration.
- **preview_files.py** — Output gallery and file management.
- **prompt_parser.py** — Weighted prompt token parsing.
- **config.py** — Central settings: device detection, dtype, default parameters, directory paths.

### Key Directories
- `models/` — Pre-loaded model checkpoints (SDXL, SD 1.5, WAN, CogVideoX, AnimateDiff)
- `loras/` — LoRA adapter files (auto-created)
- `upscalers/` — Upscaler model files (auto-created)
- `outputs/` — Generated images and videos (auto-created)
- `profiles/` — Saved prompt presets (auto-created)

## Key Technical Details

- **GPU Memory Management**: Models are offloaded/unloaded between pipelines. VAE tiling enabled for large images. 4-bit quantization (bitsandbytes) for 14B video model. CogVideoX uses sequential CPU offload for diffusion and manual float32 VAE decode with tiling to avoid cuDNN conv3d hangs on Windows.
- **Model Formats**: Supports both diffusers-format directories and single-file safetensors/ckpt checkpoints.
- **Prompt Weighting**: Uses `[token:weight]` syntax parsed by `prompt_parser.py`, then processed by Compel for attention scaling.
- **Hires Fix**: Two-pass generation — first at lower resolution, then img2img upscale pass at target resolution.
- **Config Defaults** (in `config.py`): 30 steps, 7.5 CFG, 1024×1024, float16 on CUDA, float32 on CPU.

## Constraints

- Windows 10/11 only (start.bat, path conventions)
- Requires NVIDIA GPU with 8GB+ VRAM (24GB recommended for video)
- Python 3.12 specifically (PyTorch/xformers compatibility)
- Fully offline after initial model downloads — no runtime internet access
