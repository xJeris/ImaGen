# Getting Started with ImaGen

## Requirements

- Windows 10/11
- Python 3.12
- NVIDIA GPU with 8GB+ VRAM (recommended) — CPU works but is very slow
- 24GB VRAM (e.g. RTX 4090) recommended for video generation
- ~10GB free disk space (model download + cache)

## Setup

Open a terminal in the ImaGen folder and run:

```bash
py -3.12 -m venv venv
source venv/Scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

> **Note:** torch is installed separately to ensure you get the CUDA (GPU) version, not CPU-only. The `bitsandbytes` package (in requirements.txt) is required for 4-bit quantization of large video models.

## Running the App

```bash
source venv/Scripts/activate
python app.py
```

On first launch the SDXL model (~6.5GB) will download from HuggingFace. This only happens once — the model is saved locally to `models/` and all future runs are fully offline.

Once loaded, the UI opens at **http://127.0.0.1:7860** in your browser.

## Switching Models

The **Base Model** dropdown at the top of the page lists all image models in the `models/` folder. Selecting a different model hot-swaps it (unloads the old one, loads the new one) — no restart needed.

> **Note:** Only one pipeline (image or video) is loaded at a time. Switching to a video model automatically unloads the image model to free VRAM, and vice versa.

### Adding New Models

1. Download a diffusers-format model (from HuggingFace, CivitAI, etc.)
2. Place the model folder in `models/` — it must contain a `model_index.json` file
3. Click the dropdown to refresh — the model appears automatically

The app auto-detects whether a model is SDXL or SD 1.5 and adjusts its pipeline accordingly.

> **Note:** SD 1.5 models generate at 512x512 natively. SDXL models generate at 1024x1024. Adjust the width/height sliders to match.

## Upscalers

The **Upscaler** dropdown at the top of the page lets you apply AI upscaling after generation. This is a simple post-process enlargement — see **Hires Fix** below for a more advanced two-pass approach.

### Adding Upscalers

1. Download an upscaler `.pth` file (Real-ESRGAN, SwinIR, ESRGAN, etc.)
2. Place it in the `upscalers/` folder
3. Click the dropdown to refresh — the upscaler appears automatically
4. Select it before generating — the output will be upscaled automatically

Popular upscaler models:
- `RealESRGAN_x4plus.pth` — general-purpose 4x upscaler
- `RealESRGAN_x2plus.pth` — 2x upscaler (faster, less enlargement)
- `4x-UltraSharp.pth` — sharp detail enhancement

Set the upscaler to "None" to disable upscaling.

## Text to Image

### Generate Tab

1. **Positive Prompt** — describe what you want in the image
2. **Negative Prompt** — describe what you want to avoid (e.g. `blurry, low quality, deformed, watermark`)
3. **Description** — optional extra scene details, appended to the positive prompt
4. Click **Generate** and wait a few seconds

After generation, the **seed** used is displayed below the image. Copy it into the Seed field to reproduce the same image.

#### Weighted Prompts

Emphasize or de-emphasize specific words using `[word:weight]` syntax:

| Syntax | Effect |
|--------|--------|
| `[green curtains:1.5]` | Stronger emphasis on green curtains |
| `[background:0.5]` | Reduce focus on background |
| `a [castle:1.8] on a [misty:1.3] hill` | Multiple weights in one prompt |

Weights above 1.0 increase emphasis, below 1.0 decrease it.

#### Advanced Settings

Expand the **Advanced Settings** accordion to adjust:

- **Inference Steps** (default 30) — more steps = higher quality but slower. 20–50 is the useful range.
- **Guidance Scale** (default 7.5) — how closely the image follows your prompt. Higher = more literal, lower = more creative. 5–12 is typical.
- **Sampler** — the diffusion scheduler algorithm. Options include Euler, DPM++ 2M, UniPC, and others.
- **Width / Height** (default 1024x1024) — output resolution in multiples of 64.
- **Seed** — set a specific seed to reproduce an image. -1 = random.

#### LoRA

Expand the **LoRA** accordion to apply up to two LoRAs simultaneously:

- **LoRA 1 / LoRA 2** — pick from `.safetensors` files in the `loras/` folder. Set either to "None" to leave that slot unused.
- **LoRA 1 Weight / LoRA 2 Weight** (0.0–1.5) — how strongly each LoRA style is applied

Using two LoRAs at once lets you combine styles — for example, one LoRA for a specific art style and another for a character or subject.

#### Hires Fix

The **Hires Fix** accordion provides a two-pass generation for higher-quality detail at larger resolutions:

1. First pass: generate at the base resolution (e.g. 1024x1024)
2. Upscale using an AI upscaler (e.g. RealESRGAN 2x → 2048x2048)
3. Second pass: run img2img on the upscaled image with low denoise to add real diffusion detail

This is different from the post-process **Upscaler** dropdown, which simply enlarges the image. Hires Fix adds genuine new detail through a second diffusion pass. Both can be used together — Hires Fix runs first, then post-process upscaling.

Settings:
- **Enable Hires Fix** — toggle on/off (default off)
- **Hires Upscaler** — select an upscaler for the intermediate upscale step
- **Denoise Strength** (0.1–0.8, default 0.4) — lower = closer to original, higher = more new detail. 0.3–0.5 is the sweet spot.
- **Hires Steps** (1–100, default 20) — inference steps for the second pass

### Saving Images

Click **Save as PNG** to save the current image to the `outputs/` folder with a timestamped filename.

## Image to Image

The **Image to Image** tab lets you upload an existing image and modify it using text prompts.

1. Upload a source image
2. Describe the changes you want in the **Positive Prompt** (e.g. "make it a watercolor painting" or "add snow to the scene")
3. Use the **Negative Prompt** for things to avoid
4. Adjust **Strength** to control how much the image changes:

| Strength | Effect |
|----------|--------|
| 0.2–0.3 | Subtle tweaks — color shifts, minor adjustments |
| 0.4–0.5 | Moderate changes — style shifts while keeping composition |
| 0.6–0.7 | Significant rework — new details, altered structure |
| 0.8–1.0 | Near-total reimagining — uses the source as a loose guide only |

The output resolution matches the source image dimensions. Dual LoRA and post-process upscaler are also available.

### Inpainting

Inpainting lets you selectively regenerate part of an image while keeping the rest untouched.

1. Check the **Enable Inpainting** checkbox — the source image upload switches to a canvas editor
2. Upload your image into the editor
3. Use the brush tool to **paint white** over the area you want to regenerate (e.g. clothing, a face, an object)
4. Enter a prompt describing what should replace the masked area (e.g. "red leather jacket" or "blue sky with clouds")
5. Adjust **Strength** — lower keeps the masked area closer to the original, higher gives the model more freedom
6. Click **Generate**

Only the white-painted area is regenerated. Use the eraser tool to correct mistakes in your mask.

Common uses:
- Changing clothing or accessories on a person
- Fixing faces or hands
- Removing unwanted objects
- Adding new elements to a scene

## Animate Image

The **Animate Image** tab uses AnimateDiff + SparseCtrl to turn a still image into a short animation. This requires SD 1.5 components.

### Required Models

You need three components in the `models/animatediff/` folder:

1. **SD 1.5 Base Model** — any SD 1.5 model (e.g. Realistic Vision V5.1)
2. **Motion Adapter** — the AnimateDiff motion module (e.g. `animatediff-motion-adapter-v1-5-3`)
3. **SparseControlNet** — the SparseCtrl RGB model (e.g. `animatediff-sparsectrl-rgb`)

### Generating Animations

1. Select and load all three models using the dropdowns at the top
2. Upload a source image
3. Enter a motion prompt (e.g. "wind blowing through hair, gentle swaying")
4. Adjust settings:
   - **Number of Frames** (8–32) — more frames = longer animation
   - **Image Conditioning Scale** (0.0–2.0) — how strongly to follow the source image
   - **Inference Steps** (default 25) — more = higher quality
5. Click **Animate**

Output is saved as MP4 at 8fps. Dual LoRA support is available for SD 1.5-compatible LoRAs.

## Text to Video

The **Text to Video** tab generates short video clips using WAN 2.1 models.

### Video Models

Video models are separate from image models. Two sizes are supported:

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| WAN 2.1 1.3B (Lite) | ~5GB | Fast (seconds) | Good for simple scenes |
| WAN 2.1 14B (Full) | ~7GB (4-bit quantized) | Slower (minutes) | Higher quality |

The 14B model is automatically loaded with 4-bit NF4 quantization and CPU offloading to fit within 24GB VRAM.

### Adding Video Models

1. Download a WAN 2.1 model in diffusers format
2. Place the model folder in `models/` — it must contain a `model_index.json` with a WAN pipeline class
3. Click the Video Model dropdown to refresh — WAN models appear automatically

> **Note:** Video and image models share the `models/` folder but are listed in separate dropdowns. The app auto-detects which are WAN video models.

### Video Settings

- **Duration** (1–5 seconds) — generates at 16fps (e.g. 3s = 49 frames)
- **Inference Steps** (default 30) — more steps = higher quality
- **Guidance Scale** (default 5.0) — prompt adherence
- **Sampler** — UniPC (default), Euler, or DPM++ 2M
- **Seed** — set a specific seed to reproduce a video. -1 = random.
- **LoRA 1 / LoRA 2** — up to two video-compatible LoRAs from the `loras/` folder, with independent weights

> **Note:** Weighted prompts (`[word:weight]`) are not supported for video generation — WAN uses a different text encoder (UMT5) that doesn't support prompt weighting.

### Saving Videos

Click **Save Video** to save the current video as MP4 to the `outputs/` folder.

## Training a LoRA

LoRA (Low-Rank Adaptation) lets you fine-tune the model on your own images to learn a specific style or subject.

> **Important:** LoRA training currently requires an SDXL model to be loaded. SD 1.5 models are not supported for training.

### Preparing Training Data

1. Create a folder with your training images (PNG, JPG, or WebP)
2. Optionally add a `.txt` caption file next to each image with the same name:
   ```
   my_images/
   ├── photo1.png
   ├── photo1.txt    ← "a portrait of a woman in oil painting style"
   ├── photo2.jpg
   └── photo2.txt    ← "an oil painting of a landscape with mountains"
   ```
   If no `.txt` file exists, the filename is used as the caption.

### Running Training

1. Go to the **Train LoRA** tab
2. Enter the path to your training images folder
3. Give your LoRA a name (e.g. `oil-painting-style`)
4. Adjust settings if needed:
   - **Training Steps** (default 500) — more steps = better learning but risk of overfitting. 300–1000 for most cases.
   - **Learning Rate** (default 0.0001) — lower = more stable training
   - **LoRA Rank** (default 4) — higher rank = more capacity but larger file. 4–16 is typical.
5. Click **Start Training** and monitor the log for loss values

Training saves a `.safetensors` file to the `loras/` folder. Decreasing loss values indicate the model is learning.

### Using a Trained LoRA

1. On any generation tab, expand the **LoRA** accordion
2. Select your LoRA from the **LoRA 1** dropdown (and optionally a second from **LoRA 2**)
3. Adjust each LoRA's weight (0.0–1.5) to control how strongly the style is applied
4. Generate as normal

LoRAs are available on all tabs — the app automatically filters to show only LoRAs compatible with the currently loaded model. Both LoRA slots refresh their lists when you switch models.

## Project Structure

```
ImaGen/
├── app.py                  # UI entry point (Gradio)
├── pipeline.py             # Image model loading and inference (txt2img, img2img, inpainting)
├── video_pipeline.py       # Video model loading and inference (WAN 2.1)
├── animatediff_pipeline.py # Image animation pipeline (AnimateDiff + SparseCtrl)
├── upscaler.py             # Upscaler loading and inference (spandrel)
├── prompt_parser.py        # Weighted prompt syntax
├── training.py             # LoRA training (SDXL only)
├── config.py               # Settings and defaults
├── requirements.txt        # Python dependencies
├── default_positive.txt    # Default positive prompt
├── default_negative.txt    # Default negative prompt
├── models/                 # Base models — image and video (auto-created)
│   └── animatediff/        # AnimateDiff components (base model, motion adapter, SparseCtrl)
├── upscalers/              # Upscaler .pth files (auto-created)
├── loras/                  # LoRA .safetensors files (auto-created)
└── outputs/                # Saved images and videos (auto-created)
```

## Troubleshooting

**"CUDA not available" / very slow generation**
- Ensure you installed torch with the CUDA index URL (see Setup)
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

**Out of memory errors (images)**
- Reduce image dimensions (try 768x768 or 512x512)
- Reduce inference steps
- Close other GPU-intensive applications

**Out of memory errors (video)**
- The 14B model uses 4-bit quantization + CPU offloading automatically
- If the 14B model still OOMs, use the 1.3B Lite model instead
- Ensure no image model is loaded when generating video (switching models unloads the other automatically)

**Model not showing in dropdown**
- Ensure the model folder is directly inside `models/` and contains a `model_index.json` file
- Click the dropdown to refresh the list

**Upscaler not showing in dropdown**
- Ensure the `.pth` or `.safetensors` file is in the `upscalers/` folder
- Click the dropdown to refresh the list

**Model download fails**
- Ensure you have internet for the first run only
- If interrupted, delete the `models/` folder and try again

**Video generation hangs or is very slow**
- Ensure `bitsandbytes` is installed (`pip install bitsandbytes>=0.43.0`)
- The 14B model uses CPU offloading and is expected to take several minutes
- The 1.3B model should generate in seconds

**Training fails with "requires an SDXL model"**
- LoRA training only works with SDXL models. Switch to an SDXL model before training.
