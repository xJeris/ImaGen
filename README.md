# ImaGen

**Offline text-to-image, image-to-image & text-to-video generation.**

A fully self-contained AI image and video generator that runs entirely on your local machine — no internet connection required after initial setup. Built with Stable Diffusion (SDXL / SD 1.5), Pony, Illustrious, Flux, and Krea 2 for images, and WAN 2.1 + CogVideoX for video, with a custom FastAPI backend and vanilla HTML/CSS/JS frontend.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

- **Multi-Architecture Support** — Switch between SDXL / SD 1.5, Pony, Illustrious, Flux, and Krea 2 architectures from the UI with per-architecture defaults
- **Text to Image** — Generate images from text prompts using any supported architecture
- **Image to Image** — Upload an image and transform it with text-guided diffusion
- **Inpainting** — Paint a mask over part of an image and regenerate just that area
- **Text to Video** — Generate short video clips (1–5 seconds) using WAN 2.1 or CogVideoX models
- **Image Animation** — Animate a still image using AnimateDiff + SparseCtrl (SD 1.5)
- **Weighted Prompts** — Fine-tune emphasis with `[green curtains:1.5]` syntax
- **Dual LoRA Support** — Load up to two LoRA adapters simultaneously with independent weight controls; trigger words display automatically when available
- **LoRA Training** — Train your own LoRA on custom images directly from the UI
- **Hires Fix** — Two-pass generation: base render → AI upscale → img2img refinement for sharper detail
- **AI Upscalers** — Post-process upscaling with Real-ESRGAN, SwinIR, ESRGAN, and other models via [Spandrel](https://github.com/chaiNNer-org/spandrel)
- **Preview Files** — Browse, preview, and bulk delete generated images and videos from the outputs folder
- **Model Browser** — Search and download models and LoRAs from CivitAI directly within the UI, with trigger words, recommended settings, and HuggingFace search links
- **Prompt Profiles** — Save and load prompt presets (positive + negative) across all tabs
- **Multiple Samplers** — Euler, Euler Ancestral, DPM++ 2M Karras, DPM++ SDE Karras, DDIM, UniPC
- **Batch Generation** — Generate 1–8 variations at once from the same prompt, review in the output gallery grid, save selected or all
- **Custom VAE** — Swap the model's VAE with a custom one from the `models/vaes/` folder (SDXL, SD 1.5, Pony, Illustrious)
- **Generation History** — Optionally save generation parameters (prompt, seed, model, etc.) as JSON sidecar files and PNG metadata
- **Hot-Swap Models** — Switch between models from the UI without restarting
- **Fully Offline** — After first-run model download, everything runs locally
- **VRAM Management** — Automatic model offloading, VAE tiling, chunked VAE decode, 4-bit quantization for large video models

## Screenshots

![Screenshot](https://github.com/xJeris/ImaGen/blob/main/sample.png)

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
git clone https://github.com/xJeris/ImaGen.git
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
python server.py
```

Or double-click **`start.bat`** on Windows.

On first launch, the default SDXL model (~6.5GB) downloads from HuggingFace. This only happens once — all future runs are fully offline.

The browser opens automatically to **http://127.0.0.1:7860**.

## Usage

### Text to Image

1. Select an **Architecture** (SDXL / SD 1.5, Pony, Illustrious, Flux, or Krea 2)
2. Enter a **Positive Prompt** describing what you want
3. Enter a **Negative Prompt** for things to avoid
4. Click **Generate**
5. Click **Save as PNG** to save to the `outputs/` folder

A **Prompting Guide** below the Generate button shows architecture-specific tips (tag-based vs natural language, recommended settings, etc.) and updates automatically when you switch architectures.

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

Select the **Inpainting** sub-tab in the bottom nav bar. Upload a source image, then paint a white mask over the area you want to regenerate using the brush tools. Only the masked area is changed — the rest of the image stays intact.

### Animate Image

1. Load an SD 1.5 base model, motion adapter, and SparseControlNet from the `models/animatediff/` folder
2. Upload a source image
3. Describe the desired motion (e.g. "wind blowing through hair, gentle swaying")
4. Set **Frames** (2–16) and **Playback FPS** (6–16)
5. Click **Animate**

AnimateDiff generates at 512x512 (source images are automatically resized). The maximum is 16 frames per generation — this is the trained context length of the motion adapter.

### Text to Video

1. Select a video model from the dropdown (WAN 2.1 or CogVideoX)
2. Enter a prompt describing the scene
3. Set duration (1–5 seconds) and FPS (6–30, default 24 for WAN, 8 for CogVideoX)
4. Click **Generate**

Videos are exported as MP4. WAN uses single-pass diffusion with chunked VAE decode. CogVideoX runs diffusion in fp16 with sequential CPU offload, then decodes the VAE in float32 with spatial tiling.

### Preview Files

The **Preview Files** tab lets you browse all saved images and videos in the `outputs/` folder:

1. Click on any thumbnail to view it in the detail panel
2. Use the **Filter** dropdown to show only images or videos
3. Use the **Sort** dropdown to order by date or name
4. Use **Select All** / **Deselect All** for batch operations
5. Check individual files and click **Delete Selected** to remove them
6. Click **Open** to view the full file in a new browser tab
7. Click **Refresh** to reload the gallery

### Model Browser

The **Model Browser** tab lets you search and download models directly from CivitAI:

1. Enter a search query or browse the default results
2. Filter by model type (Checkpoint, LORA), base model (SDXL, SD 1.5, etc.), and content rating
3. Click a tile to select it — the info panel shows file details, trigger words, recommended settings (CFG, steps, sampler), and links to CivitAI and HuggingFace
4. Click **Download** to save the model to the appropriate folder (`models/` or `loras/`)

When a LoRA is downloaded, a metadata sidecar (`.json`) is saved alongside it. This enables automatic trigger word display when the LoRA is selected in the Text to Image tab.

Some models require a CivitAI API key — expand the **API Key** section to enter and save your key.

> **Note:** This is the only feature in ImaGen that requires an internet connection.

### Hires Fix

A two-pass approach for high-resolution detail:

1. First pass generates at base resolution (e.g. 1024x1024)
2. AI upscaler enlarges the image (e.g. 2x → 2048x2048)
3. Second pass runs img2img with low denoise to add real diffusion detail

Enable it under the **Hires Fix** accordion in the Text to Image tab.

### Prompt Profiles

Save and reuse prompt combinations across sessions:

1. Expand the **Prompt Profiles** accordion in the Text to Image sidebar
2. **Save** — click Save, enter a name (letters and numbers, max 30 characters), and confirm
3. **Load** — select a profile from the dropdown and click Load to fill the current tab's prompt fields
4. **Delete** — remove a saved profile (the "default" profile clears its contents instead of being removed)

Profiles are stored as text files in the `profiles/` folder. You can also create profiles manually by placing `{name}_positive.txt` and `{name}_negative.txt` files there.

### Training LoRA

1. Load an SDXL model first (LoRA training requires SDXL)
2. Prepare a folder of training images (optionally with `.txt` caption files — without a caption file, the filename is used as the caption)
3. Go to the **LoRA Training** tab
4. Set the image folder path, output name, and training parameters (steps, learning rate, LoRA rank)
5. Click **Start Training** — progress and loss are shown in real time
6. Click **Stop** to interrupt training early (partial LoRA is still saved)

The trained LoRA is saved to `loras/sdxl/` and immediately available in the LoRA dropdown.

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Start generation (current tab) |
| `Ctrl+S` | Save current output |
| `Escape` | Close lightbox or stop generation |

## Adding Models

### Image Models

1. Download a model in diffusers format (from HuggingFace, CivitAI, etc.) or as a single `.safetensors` checkpoint
2. Place it in the appropriate folder based on architecture:
   - **SDXL / SD 1.5** → `models/sdxl/`
   - **Pony** → `models/pony/`
   - **Illustrious** → `models/illustrious/`
   - **Flux** → `models/flux/`
   - **Krea 2** → `models/krea2/` (diffusers directory or single `.safetensors` file)
3. Select the matching architecture from the **Architecture** dropdown
4. Click the **Base Model** dropdown to refresh — the model appears automatically

LoRA files follow the same pattern (`loras/`, `loras/pony/`, `loras/illustrious/`, `loras/flux/`, `loras/krea2/`).

> **Krea 2 Note:** Single-file `.safetensors` checkpoints contain only the transformer weights. On first load, the text encoder (~9GB) and VAE (~254MB) will be automatically downloaded from HuggingFace and cached in `models/krea2/_encoders/`. This is the only time an internet connection is needed. Diffusers-format directories include all components and work fully offline.
>
> **Supported formats:** bf16, fp16, and non-scaled fp8 checkpoints (e.g. AlperKTS/Krea2_FP8). Both diffusers and ComfyUI key naming conventions are auto-detected. **Not supported:** FP8-scaled checkpoints (e.g. `_fp8_scaled` variants) and INT8 checkpoints — these use quantization formats that are incompatible with diffusers.

### Video Models

**WAN 2.1** — Download in diffusers format and place in `models/wan/`.

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| WAN 2.1 1.3B (Lite) | ~5GB | 1–2 minutes | Good for simple scenes |
| WAN 2.1 14B (Full) | ~7GB (4-bit) | Slower | Higher quality, more detail |

**CogVideoX** — Download in diffusers format and place in `models/cogvideox/`.

| Model | VRAM | Speed | Resolution |
|-------|------|-------|------------|
| CogVideoX-2b | ~5GB | ~2.5s/step | 720x480 |

Click the **Video Model** dropdown to refresh after adding models.

### Upscalers

1. Download an upscaler model (`.pth` or `.safetensors`)
2. Place it in the `upscalers/` folder
3. Select it from the **Upscaler** dropdown

Popular upscalers: `RealESRGAN_x4plus.pth`, `RealESRGAN_x2plus.pth`, `4x-UltraSharp.pth`

### Custom VAEs

1. Download a VAE file (`.safetensors`) or a diffusers-format VAE directory
2. Place it in `models/vaes/`
3. Select it from the **VAE** dropdown on the Text to Image or Image to Image tab

The VAE persists across model swaps. Set to "Default" to use the model's bundled VAE. Flux and Krea 2 use their own VAE architectures and do not support custom VAE swapping.

## Project Structure

```
ImaGen/
├── server.py               # FastAPI backend (REST API + WebSocket)
├── static/                 # Frontend
│   ├── index.html          # Main HTML page
│   ├── css/style.css       # Stylesheet (dark theme)
│   └── js/
│       ├── api.js          # API client + WebSocket
│       └── app.js          # UI logic (tabs, forms, generation)
├── pipeline.py             # Image generation pipeline (txt2img, img2img, inpainting)
├── flux_pipeline.py        # Flux image generation pipeline
├── krea2_pipeline.py       # Krea 2 image generation pipeline (12.9B DiT, Turbo/Raw)
├── video_pipeline.py       # Video generation pipeline (WAN 2.1)
├── cogvideox_pipeline.py   # CogVideoX video pipeline (fp16 diffusion + fp32 VAE decode)
├── video_chunker.py        # VRAM-safe video generation (chunked VAE decode)
├── animatediff_pipeline.py # Image animation pipeline (AnimateDiff + SparseCtrl)
├── civitai_browser.py      # CivitAI model search and download
├── upscaler.py             # AI upscaler inference (Spandrel)
├── prompt_parser.py        # Weighted prompt syntax parser
├── training.py             # LoRA fine-tuning (SDXL)
├── config.py               # Settings and defaults
├── requirements.txt        # Python dependencies
├── start.bat               # Windows launcher
├── default_positive.txt    # Default positive prompt
├── default_negative.txt    # Default negative prompt
├── profiles/               # Saved prompt profiles
├── models/                 # Base models (per-architecture subdirectories)
│   ├── sdxl/               # SDXL / SD 1.5 checkpoints
│   ├── pony/               # Pony architecture models
│   ├── illustrious/        # Illustrious architecture models
│   ├── flux/               # Flux architecture models
│   ├── krea2/              # Krea 2 architecture models
│   │   └── _encoders/      # Auto-cached text encoder + VAE (single-file loading)
│   ├── wan/                # WAN 2.1 video models
│   ├── cogvideox/          # CogVideoX video models
│   ├── animatediff/        # AnimateDiff components (base model, motion adapter, SparseCtrl)
│   └── vaes/               # Custom VAE files (.safetensors or diffusers dirs)
├── upscalers/              # Upscaler model files
├── loras/                  # LoRA adapter files (per-architecture subdirectories)
│   ├── sdxl/               # SDXL / SD 1.5 LoRAs (+ JSON metadata sidecars)
│   ├── pony/               # Pony LoRAs
│   ├── illustrious/        # Illustrious LoRAs
│   ├── flux/               # Flux LoRAs
│   └── krea2/              # Krea 2 LoRAs
└── outputs/                # Saved images and videos (+ JSON sidecar files)
```

## Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** — Backend REST API + WebSocket server
- **[Diffusers](https://github.com/huggingface/diffusers)** — Stable Diffusion & WAN 2.1 pipelines
- **[PyTorch](https://pytorch.org/)** — Deep learning framework with CUDA acceleration
- **[Compel](https://github.com/damian0815/compel)** — Prompt weighting and embedding
- **[Spandrel](https://github.com/chaiNNer-org/spandrel)** — Universal upscaler model loader
- **[PEFT](https://github.com/huggingface/peft)** — LoRA training and loading
- **[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)** — 4-bit quantization for large models

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA not available / very slow | Reinstall PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| Out of memory (images) | Reduce resolution (768x768 or 512x512), reduce batch size, reduce steps, close other GPU apps |
| Out of memory (video) | Use the 1.3B Lite model; 14B uses 4-bit quantization + chunked VAE decode automatically |
| CogVideoX black/red frames | This is handled automatically — the pipeline decodes in float32 to avoid fp16 precision loss |
| Model not in dropdown | Ensure it's in the correct architecture subfolder under `models/`; click the dropdown to refresh |
| Training fails | LoRA training requires an SDXL model — switch to SDXL / SD 1.5 architecture and load a model before training |
| Krea 2 single-file: "text encoder not found" | Internet is needed on first load to download the text encoder + VAE (~9GB). These are cached in `models/krea2/_encoders/` for offline use afterward |
| Krea 2: "FP8-scaled checkpoints..." error | FP8-scaled checkpoints (e.g. `_fp8_scaled` variants) use a quantization format incompatible with diffusers. Use a bf16 or non-scaled fp8 checkpoint instead |
| Krea 2: "INT8-quantised checkpoints..." error | INT8 checkpoints use a quantization format incompatible with diffusers. Use a bf16 or fp8 checkpoint instead |
| First run download fails | Internet is needed only once; delete `models/` and retry if interrupted |

## License

This project is provided as-is for personal and educational use.
