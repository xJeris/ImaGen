# ImaGen

**Offline text-to-image, image-to-image & text-to-video generation.**

A fully self-contained AI image and video generator that runs entirely on your local machine — no internet connection required after initial setup. Built with Stable Diffusion (SDXL / SD 1.5) for images and WAN 2.1 for video, wrapped in a clean Gradio web UI.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

- **Text to Image** — Generate images from text prompts using Stable Diffusion XL or SD 1.5 models
- **Image to Image** — Upload an image and transform it with text-guided diffusion
- **Inpainting** — Paint a mask over part of an image and regenerate just that area
- **Text to Video** — Generate short video clips (1–5 seconds) using WAN 2.1 models
- **Image Animation** — Animate a still image using AnimateDiff + SparseCtrl (SD 1.5)
- **Weighted Prompts** — Fine-tune emphasis with `[green curtains:1.5]` syntax
- **Dual LoRA Support** — Load up to two LoRA adapters simultaneously with independent weight controls
- **LoRA Training** — Train your own LoRA on custom images directly from the UI
- **Hires Fix** — Two-pass generation: base render → AI upscale → img2img refinement for sharper detail
- **AI Upscalers** — Post-process upscaling with Real-ESRGAN, SwinIR, ESRGAN, and other models via [Spandrel](https://github.com/chaiNNer-org/spandrel)
- **Multiple Samplers** — Euler, Euler Ancestral, DPM++ 2M Karras, DPM++ SDE Karras, DDIM, UniPC
- **Hot-Swap Models** — Switch between models from the UI without restarting
- **Fully Offline** — After first-run model download, everything runs locally
- **VRAM Management** — Automatic model offloading, VAE tiling, 4-bit quantization for large video models

## Screenshots

<!-- Add screenshots of your UI here -->
<!-- ![Text to Image](screenshots/txt2img.png) -->
<!-- ![Text to Video](screenshots/txt2vid.png) -->

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 | Windows 11 |
| Python | 3.12 | 3.12 |
| GPU | NVIDIA, 8GB VRAM | NVIDIA RTX 4090 (24GB) |
| Disk | ~10GB free | ~30GB+ (multiple models) |

> CPU-only mode works but is very slow. 24GB VRAM is recommended for video generation with the 14B model.

## Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ImaGen.git
cd ImaGen

# Create a virtual environment
py -3.12 -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# or: venv\Scripts\activate    # Windows (cmd)

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
```

> **Note:** PyTorch is installed separately to ensure you get the CUDA (GPU) build. The `bitsandbytes` package is required for 4-bit quantization of large video models.

## Quick Start

```bash
source venv/Scripts/activate
python app.py
```

Or double-click **`start.bat`** on Windows.

On first launch, the default SDXL model (~6.5GB) downloads from HuggingFace. This only happens once — all future runs are fully offline.

Once loaded, open **http://127.0.0.1:7860** in your browser.

## Usage

### Text to Image

1. Enter a **Positive Prompt** describing what you want
2. Enter a **Negative Prompt** for things to avoid
3. Click **Generate**
4. Click **Save as PNG** to save to the `outputs/` folder

#### Weighted Prompts

Emphasize or de-emphasize words with `[word:weight]` syntax:

```
[green curtains:1.5] in a cozy room with [soft lighting:1.3]
```

Weights above 1.0 increase emphasis, below 1.0 decrease it.

### Image to Image

1. Upload a source image
2. Describe the changes you want
3. Adjust **Strength** (0.0 = no change, 1.0 = fully reimagine)
4. Click **Generate**

#### Inpainting

Enable the **Enable Inpainting** checkbox to switch to inpainting mode. This replaces the image upload with a canvas editor where you can paint a white mask over the area you want to regenerate. Only the masked area is changed — the rest of the image stays intact.

### Animate Image

1. Load an SD 1.5 base model, motion adapter, and SparseControlNet from the `models/animatediff/` folder
2. Upload a source image
3. Describe the desired motion (e.g. "wind blowing through hair, gentle swaying")
4. Click **Animate**

### Text to Video

1. Select a WAN 2.1 video model from the dropdown
2. Enter a prompt describing the scene
3. Set duration (1–5 seconds at 16fps)
4. Click **Generate**

Videos are exported as MP4. The 1.3B Lite model generates in seconds; the 14B Full model takes minutes but produces higher quality.

### Hires Fix

A two-pass approach for high-resolution detail:

1. First pass generates at base resolution (e.g. 1024x1024)
2. AI upscaler enlarges the image (e.g. 2x → 2048x2048)
3. Second pass runs img2img with low denoise to add real diffusion detail

Enable it under the **Hires Fix** accordion in the Text to Image tab.

### Training LoRA

1. Prepare a folder of training images (optionally with `.txt` caption files)
2. Go to the **Train LoRA** tab
3. Set the image directory, LoRA name, and training parameters
4. Click **Start Training**

The trained LoRA is saved to `loras/` and immediately available in the LoRA dropdown.

> LoRA training currently requires an SDXL model to be loaded.

## Adding Models

### Image Models

1. Download a model in diffusers format (from HuggingFace, CivitAI, etc.) or as a single `.safetensors` checkpoint
2. Place it in the `models/` folder
3. Click the **Base Model** dropdown to refresh — the model appears automatically

The app auto-detects SDXL vs SD 1.5 and adjusts accordingly.

### Video Models

1. Download a WAN 2.1 model in diffusers format
2. Place the model folder in `models/`
3. Click the **Video Model** dropdown to refresh

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| WAN 2.1 1.3B (Lite) | ~5GB | Fast | Good for simple scenes |
| WAN 2.1 14B (Full) | ~7GB (4-bit) | Slower | Higher quality, more detail |

### Upscalers

1. Download an upscaler model (`.pth` or `.safetensors`)
2. Place it in the `upscalers/` folder
3. Select it from the **Upscaler** dropdown

Popular upscalers: `RealESRGAN_x4plus.pth`, `RealESRGAN_x2plus.pth`, `4x-UltraSharp.pth`

## Project Structure

```
ImaGen/
├── app.py                  # Gradio web UI
├── pipeline.py             # Image generation pipeline (txt2img, img2img, inpainting)
├── video_pipeline.py       # Video generation pipeline (WAN 2.1)
├── animatediff_pipeline.py # Image animation pipeline (AnimateDiff + SparseCtrl)
├── upscaler.py             # AI upscaler inference (Spandrel)
├── prompt_parser.py        # Weighted prompt syntax parser
├── training.py             # LoRA fine-tuning (SDXL)
├── config.py               # Settings and defaults
├── requirements.txt        # Python dependencies
├── start.bat               # Windows launcher
├── default_positive.txt    # Default positive prompt
├── default_negative.txt    # Default negative prompt
├── models/                 # Base models (image + video)
│   └── animatediff/        # AnimateDiff components (base model, motion adapter, SparseCtrl)
├── upscalers/              # Upscaler model files
├── loras/                  # LoRA adapter files
└── outputs/                # Saved images and videos
```

## Tech Stack

- **[Diffusers](https://github.com/huggingface/diffusers)** — Stable Diffusion & WAN 2.1 pipelines
- **[PyTorch](https://pytorch.org/)** — Deep learning framework with CUDA acceleration
- **[Gradio](https://gradio.app/)** — Web UI
- **[Compel](https://github.com/damian0815/compel)** — Prompt weighting and embedding
- **[Spandrel](https://github.com/chaiNNer-org/spandrel)** — Universal upscaler model loader
- **[PEFT](https://github.com/huggingface/peft)** — LoRA training and loading
- **[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)** — 4-bit quantization for large models

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA not available / very slow | Reinstall PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| Out of memory (images) | Reduce resolution (768x768 or 512x512), reduce steps, close other GPU apps |
| Out of memory (video) | Use the 1.3B Lite model; 14B uses 4-bit quantization automatically |
| Model not in dropdown | Ensure it's in `models/` with a `model_index.json`; click dropdown to refresh |
| Training fails | LoRA training requires an SDXL model — switch models before training |
| First run download fails | Internet is needed only once; delete `models/` and retry if interrupted |

## License

This project is provided as-is for personal and educational use.
