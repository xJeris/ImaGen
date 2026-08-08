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
python server.py
```

Or double-click **`start.bat`** on Windows.

On first launch the SDXL model (~6.5GB) will download from HuggingFace. This only happens once — the model is saved locally to `models/sdxl/` and all future runs are fully offline.

The browser opens automatically to **http://127.0.0.1:7860**.

## UI Layout

ImaGen uses a 5-tab layout with a top nav bar, bottom sub-nav bar, center canvas area, and right sidebar:

- **Top Nav** — Image Generator, Video Generator, Model Browser, Preview Files, LoRA Training
- **Bottom Nav** — Sub-tabs for the current mode (e.g. Text to Image / Image to Image / Inpainting / ControlNet)
- **Canvas** — Center area showing generated images, video player, model browser grid, or file gallery
- **Sidebar** — Right panel with settings, controls, and generation parameters
- **Status Bar** — Bottom bar showing current model, seed, and VRAM usage

## Switching Models

### Architecture Selector

The **Architecture** dropdown in the sidebar lets you switch between supported model families:

| Architecture | Default Resolution | Default Steps | Default CFG |
|-------------|-------------------|---------------|-------------|
| SDXL / SD 1.5 | 1024×1024 | 30 | 7.5 |
| Pony | 1024×1024 | 25 | 3.0 |
| Illustrious | 1024×1024 | 28 | 5.0 |
| Flux | 512×512 | 20 | 3.5 |
| Krea 2 | 1024×1024 | 8 | 0.0 |

Switching architecture unloads the current model and updates the model/LoRA dropdowns to show only models in that architecture's folder. Default generation parameters are updated automatically.

A **Prompting Guide** accordion in the sidebar updates when you switch architectures, showing the correct prompting style for each (tag-based vs natural language, score tags for Pony, etc.).

### Base Model Dropdown

The **Checkpoint** dropdown lists all image models in the current architecture's folder. Selecting a different model hot-swaps it (unloads the old one, loads the new one) — no restart needed.

> **Note:** Only one pipeline (image or video) is loaded at a time. Switching to a video model automatically unloads the image model to free VRAM, and vice versa.

### Adding New Models

1. Download a diffusers-format model (from HuggingFace, CivitAI, etc.) or a single `.safetensors` checkpoint
2. Place it in the appropriate directory:
   - **SDXL / SD 1.5** → `models/sdxl/`
   - **Pony** → `models/pony/`
   - **Illustrious** → `models/illustrious/`
   - **Flux** → `models/flux/`
   - **Krea 2** → `models/krea2/`
3. Select the matching architecture from the **Architecture** dropdown
4. Click the **Checkpoint** dropdown to refresh — the model appears automatically

LoRA files follow the same pattern (`loras/sdxl/`, `loras/pony/`, `loras/illustrious/`, `loras/flux/`, `loras/krea2/`).

> **Note:** SD 1.5 models generate at 512x512 natively. SDXL models generate at 1024x1024. Adjust the width/height sliders to match.

### Model Browser

You can also search and download models directly from CivitAI using the **Model Browser** tab:

1. Enter a search query or browse the default results
2. Use the filters to narrow by model type (Checkpoint, LORA), base model (SDXL, SD 1.5, Pony, etc.), and content rating
3. Click a tile to select it — the info panel shows file details, **trigger words**, **recommended settings** (CFG, steps, sampler, clip skip), and links to both **CivitAI** and **HuggingFace** (for alternative downloads)
4. Click **Download** — the file is saved to the appropriate folder for the selected architecture (e.g. `models/pony/` or `loras/flux/`), and the model dropdowns refresh automatically

When a LoRA is downloaded, a `.json` metadata sidecar is saved alongside it. This allows the trigger words to display automatically when you select the LoRA in the generation tabs.

Some restricted models require a CivitAI API key. Expand the **API Key** section at the bottom of the browser sidebar to enter and save your key. The key is stored in `~/.imagen/civitai_key.txt` (outside the project folder).

> **Note:** The Model Browser is the only feature in ImaGen that requires an internet connection. All other functionality works fully offline.

### Model Compatibility

ImaGen supports **SD 1.5**, **SDXL**, **Pony**, **Illustrious**, **Flux**, and **Krea 2** architectures for image generation, and **WAN** + **CogVideoX** for video generation.

**Supported — Image Generation (Text to Image / Image to Image / Inpainting / ControlNet):**

| Base Model | Architecture | Notes |
|------------|-------------|-------|
| SD 1.5 | SDXL / SD 1.5 | Native 512x512 |
| SD 1.5 LCM | SDXL / SD 1.5 | Use low steps (4-8), guidance ~1.0 |
| SD 1.5 Hyper | SDXL / SD 1.5 | Use low steps (1-4) |
| SD 2.0 | SDXL / SD 1.5* | May work — same UNet shape |
| SD 2.1 | SDXL / SD 1.5* | May work — same UNet shape |
| SDXL 1.0 | SDXL / SD 1.5 | Native 1024x1024 |
| SDXL Lightning | SDXL / SD 1.5 | Use low steps (2-8), guidance ~1.0 |
| SDXL Hyper | SDXL / SD 1.5 | Use low steps (1-4) |
| Z Image Turbo | SDXL / SD 1.5 | Use low steps (1-4), guidance ~1.0 |
| Z Image Base | SDXL / SD 1.5 | SDXL fine-tune |
| Pony / Pony V7 | Pony | Place in `models/pony/` |
| Illustrious / NoobAI | Illustrious | Place in `models/illustrious/` |
| Flux .1 D / .1 S / .1 Krea / .1 Kontext | Flux | Place in `models/flux/`. No negative prompts or weighted syntax. |
| Flux .2 D / .2 Klein variants | Flux | Place in `models/flux/` |
| Krea 2 | Krea 2 | Place in `models/krea2/`. No negative prompts. |

**Supported — Video Generation:**

| Base Model | Type | Folder | Notes |
|------------|------|--------|-------|
| Wan Video 1.3B t2v | T2V | `models/wan/` | Lite model, ~5GB VRAM |
| Wan Video 14B t2v | T2V | `models/wan/` | Full model, uses 4-bit quantization |
| Wan Video 14B i2v 480p | I2V | `models/wan_i2v/` | Image-to-video, requires separate model |
| Wan Video 14B i2v 720p | I2V | `models/wan_i2v/` | Image-to-video, higher resolution |
| CogVideoX-2b | T2V + I2V | `models/cogvideox/` | ~5GB VRAM, 720x480, same model for both |

**Not Supported (different architecture):**

| Base Model | Architecture | Reason |
|------------|-------------|--------|
| SD 1.4 | SD 1.x | Older, untested |
| Aura Flow | Flow-matching transformer | Different architecture |
| Chroma | Unknown | Different architecture |
| HiDream | Unknown | Different architecture |
| Hunyuan 1 / Hunyuan Video | DiT (transformer) | Requires HunyuanPipeline |
| Kolors | Different text encoder | Requires KolorsPipeline |
| Lumina | DiT (transformer) | Different architecture |
| Mochi | DiT (transformer) | Different architecture |
| LTXV / LTXV2 | Transformer-based video | Different architecture |

> **Tip:** If a model is a fine-tune of a supported architecture (e.g. downloaded from CivitAI with those base types), it will work — just place it in the correct architecture folder. The key is the underlying architecture, not the model name.

## Upscalers

The **Upscaler** dropdown in the sidebar lets you apply AI upscaling after generation. This is a simple post-process enlargement — see **Hires Fix** below for a more advanced two-pass approach.

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
3. Click **Generate** and wait a few seconds

During generation, a progress bar at the bottom of the canvas fills as each diffusion step completes. After generation, the **seed** used is displayed in the status bar. Copy it into the Seed field to reproduce the same image.

#### Prompt Token Counter

A live token count badge appears next to each prompt label (e.g. `45/77`) showing how many tokens your prompt uses vs. the model's limit. The badge updates as you type with color coding:

- **Normal** (muted) — within limits
- **Yellow** — at 85%+ of the limit (approaching truncation)
- **Red** — over the limit (prompt will be truncated)

Token limits vary by architecture: SDXL/SD1.5/Pony/Illustrious = 77 (CLIP), Flux/Krea 2 = 77 (CLIP) + 512 (T5), WAN = 512 (T5), CogVideoX = 226 (T5). Token counters appear on all prompt textareas across all modes (T2I, I2I, Inpaint, ControlNet, Video, I2V, Animate).

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
- **Batch Size** (1–8, default 1) — generate multiple variations at once. Results appear as a grid in the canvas. Click any image to view it in the lightbox. Use the checkboxes to select which images to save.

#### LoRA

Expand the **LoRA** accordion to apply up to two LoRAs simultaneously:

- **LoRA 1 / LoRA 2** — pick from `.safetensors` files in the architecture's `loras/` folder. Set either to "None" to leave that slot unused.
- **LoRA 1 Weight / LoRA 2 Weight** (0.0–1.5) — how strongly each LoRA style is applied
- **Trigger Words** — if the LoRA has a metadata sidecar (auto-created when downloaded from the Model Browser), its trigger words are displayed below the dropdown. Include these words in your prompt to activate the LoRA's trained effect.

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

### Custom VAE

The **VAE** dropdown in the sidebar lets you swap the model's VAE with a custom one:

1. Download a VAE `.safetensors` file or a diffusers-format VAE directory
2. Place it in `models/vaes/`
3. Select it from the **VAE** dropdown — it takes effect immediately
4. Set to "Default" to revert to the model's bundled VAE

The selected VAE persists across model switches. Flux and Krea 2 do not support custom VAE swapping (they use different VAE architectures).

### Saving Images

Click **Save PNG** to save the current image to the `outputs/` folder with a timestamped filename.

For batch generations, use the checkboxes on each image in the grid to select which ones to save. The save button text updates to show how many are selected (e.g. "Save 3 PNGs").

#### Generation History

Check **Save with history** (next to the Save button) before saving to embed generation metadata:

- A `.json` sidecar file is saved alongside the PNG with all parameters (prompt, seed, model, LoRAs, sampler, dimensions, VAE, etc.)
- The same metadata is embedded in the PNG file's `tEXt` chunks under the `ImaGen:params` key

This is off by default. When off, images are saved as plain PNGs with no metadata.

## Prompt Profiles

Prompt profiles let you save and reuse positive/negative prompt combinations across sessions.

### Saving a Profile

1. Expand the **Prompt Profiles** accordion in the sidebar
2. Click **Save** — a text input appears
3. Enter a profile name (letters and numbers only, max 30 characters)
4. Click **OK** — the current tab's positive and negative prompts are saved to the `profiles/` folder

### Loading a Profile

1. Select a profile from the dropdown in the **Prompt Profiles** accordion
2. Click **Load** — the prompts are applied to the current tab's prompt fields

### Deleting a Profile

Click **Delete** to remove the selected profile. The "default" profile is special — deleting it clears the contents of `default_positive.txt` and `default_negative.txt` rather than removing the files.

### Manual Profiles

You can create profiles by hand — place `{name}_positive.txt` and `{name}_negative.txt` in the `profiles/` folder. They appear in the dropdown automatically.

## Image to Image

Select the **Image to Image** sub-tab in the bottom nav bar.

1. Upload a source image (click or drag-and-drop)
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

Select the **Inpainting** sub-tab in the bottom nav bar.

1. Upload your image into the canvas editor
2. Use the **brush tool** to paint white over the area you want to regenerate (e.g. clothing, a face, an object)
3. Use the **eraser** to correct mistakes in your mask
4. Enter a prompt describing what should replace the masked area (e.g. "red leather jacket" or "blue sky with clouds")
5. Adjust **Strength** — lower keeps the masked area closer to the original, higher gives the model more freedom
6. Click **Generate**

Only the white-painted area is regenerated. Use **Clear** to reset the mask, or **Undo** to step back.

Common uses:
- Changing clothing or accessories on a person
- Fixing faces or hands
- Removing unwanted objects
- Adding new elements to a scene

## ControlNet

Select the **ControlNet** sub-tab in the bottom nav bar (under Image Generator). ControlNet lets you guide image generation using structural control images — edges, depth maps, body poses, line art, etc.

> **Note:** ControlNet is supported for all image architectures except Krea 2. The ControlNet sub-tab is automatically dimmed when Krea 2 is selected.

### How It Works

ControlNet takes a "control image" (like an edge map or depth map) and uses it to dictate the structure of the generated image, while your prompt controls the content and style. For example, you can extract the edges from a photo of a person, then generate "an oil painting of a knight in armor" that follows the same pose and composition.

### Adding ControlNet Models

1. Download a ControlNet model (`.safetensors` file or diffusers-format directory)
2. Place it in `models/controlnet/`
3. The model appears in the **ControlNet Model** dropdown automatically

ControlNet models are architecture-specific — ensure you use one compatible with your loaded base model (e.g., an SDXL ControlNet for an SDXL checkpoint, or a Flux ControlNet for a Flux checkpoint).

### Generating with ControlNet

1. Load a base model (any architecture except Krea 2)
2. Select a **ControlNet Model** from the dropdown
3. Upload a source image (click or drag-and-drop)
4. Select a **Preprocessor** to extract the control signal from your image:

| Preprocessor | What it extracts | Best for |
|-------------|-----------------|----------|
| Canny | Edge detection | General structure, outlines |
| Depth (MiDaS) | Depth map | Spatial layout, 3D positioning |
| OpenPose | Body pose skeleton | Character poses |
| Lineart | Clean line drawing | Illustrations, drawings |
| SoftEdge (HED) | Soft edges | Smooth outlines |
| Normal Map | Surface normals | 3D surface detail |
| Scribble | Rough sketches | Loose compositional guidance |
| None (raw image) | Nothing — uses image as-is | Pre-processed control maps |

5. Click **Preview** to see what the preprocessor produces before generating
6. Enter a prompt describing the desired output
7. Adjust settings:
   - **Conditioning Scale** (0.0–2.0, default 1.0) — how strongly ControlNet influences the output. Lower = more creative freedom, higher = stricter adherence to the control image.
   - **Guidance Start** (0.0–1.0, default 0.0) — at which diffusion step ControlNet starts influencing
   - **Guidance End** (0.0–1.0, default 1.0) — at which diffusion step ControlNet stops influencing
   - **Guess Mode** — ControlNet tries to recognize the content without the prompt
8. Click **Generate**

> **Note:** ControlNet preprocessor detector models (~100-300MB each) are downloaded from HuggingFace on first use. Subsequent uses are instant.

> **Note:** LoRAs applied to the base model also affect ControlNet generation, since the ControlNet pipeline shares the same UNet/transformer. VRAM usage increases by ~1.2GB (SD 1.5) or ~2.5GB (SDXL) on top of the base model.

## Image to Video

Select the **Image to Video** sub-tab under the Video Generator tab. This converts a still image into a short video clip.

### CogVideoX I2V

CogVideoX image-to-video reuses the same model you already loaded for text-to-video — no extra download needed.

1. Load a CogVideoX model in the Text to Video tab
2. Switch to the **Image to Video** sub-tab
3. Upload a source image
4. Enter a prompt describing the desired motion (e.g. "camera slowly panning right, wind blowing through trees")
5. Adjust duration, FPS, steps, and other settings
6. Click **Generate**

### WAN I2V

WAN image-to-video requires a separate model (different transformer architecture + CLIP image encoder) from the T2V models.

1. Download a WAN I2V model in diffusers format and place it in `models/wan_i2v/`
2. Switch to the **Image to Video** sub-tab
3. Select and load a WAN I2V model from the dropdown
4. Upload a source image
5. Enter a prompt and adjust settings
6. Click **Generate**

## Animate Image

Select the **Animate Image** sub-tab under the Video Generator tab. This uses AnimateDiff + SparseCtrl to turn a still image into a short animation using SD 1.5 components.

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
   - **Frames** (2–16) — number of frames to generate. 16 is the maximum (trained context length of the motion adapter).
   - **Playback FPS** (6–16, default 12) — controls playback speed. 12 frames @ 12 FPS = 1 second, 16 frames @ 8 FPS = 2 seconds.
   - **Image Conditioning Scale** (0.0–2.0) — how strongly to follow the source image. Higher = more faithful to the source.
   - **Inference Steps** (default 25) — more = higher quality
   - **VRAM / Duration Estimate** — shows frame count, duration, and VRAM usage in real-time
5. Click **Animate**

Source images are automatically resized to 512x512 (the native resolution for SD 1.5). Output is saved as MP4.

## Text to Video

Select the **Text to Video** sub-tab under the Video Generator tab.

### Video Models

Video models are separate from image models. Place them in `models/wan/` (WAN) or `models/cogvideox/` (CogVideoX).

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| WAN 2.1 1.3B (Lite) | ~5GB | 1–2 minutes | Good for simple scenes |
| WAN 2.1 14B (Full) | ~7GB (4-bit quantized) | Slower (minutes) | Higher quality |
| CogVideoX-2b | ~5GB | ~2.5s/step | 720x480 |

The 14B model is automatically loaded with 4-bit NF4 quantization and CPU offloading to fit within 24GB VRAM.

### VRAM-Safe Video Generation

Video generation uses a single-pass diffusion + chunked VAE decode approach:

1. **Diffusion** runs all frames in one pass to maintain temporal coherence (no jumpiness or subject drift)
2. **VAE decode** splits the latent tensor into small temporal batches to stay within VRAM budget
3. Aggressive memory reclamation happens between stages — text encoders and transformer caches are freed before VAE decode

This allows generating 5-second videos at 30fps (149 frames) on a 24GB GPU with 2 LoRAs loaded.

### Video Settings

- **Duration** (1–5 seconds) — default 24fps (e.g. 3s at 24fps = 73 frames)
- **FPS** (6–30) — 24 = cinematic, 30 = smooth. Higher FPS uses more VRAM.
- **Inference Steps** (default 30) — more steps = higher quality
- **Guidance Scale** (default 5.0) — prompt adherence
- **Sampler** — UniPC (default), Euler, or DPM++ 2M
- **Seed** — set a specific seed to reproduce a video. -1 = random.
- **LoRA 1 / LoRA 2** — up to two video-compatible LoRAs from the `loras/wan/` folder, with independent weights
- **VRAM Estimate** — shown in real-time as you adjust duration and FPS, with available/total VRAM comparison

> **Note:** WAN requires frame counts matching `4k + 1` (5, 9, 13, ..., 149). The app automatically rounds your duration x FPS to the nearest valid count.

> **Note:** Weighted prompts (`[word:weight]`) are not supported for video generation — WAN uses a different text encoder (UMT5) that doesn't support prompt weighting.

### Saving Videos

Click **Save MP4** to save the current video to the `outputs/` folder.

## Preview Files

The **Preview Files** tab lets you browse, preview, and manage all files in the `outputs/` folder.

### Browsing

- The gallery auto-populates when you open the tab
- Click any thumbnail to view it in the detail panel on the right
- Use **Filter** to show only Images or Videos
- Use **Sort** to order by Newest First, Oldest First, or Name A-Z
- Click **Refresh** to reload the gallery
- Click **Open** to view the selected file in a new browser tab
- Video files show first-frame thumbnails (cached in `outputs/.thumbs/`)

### Deleting Files

1. Use **Select All** to check all files, or check individual files
2. The **Delete Selected (N)** button shows the count of checked files
3. Click **Delete Selected** to permanently remove the checked files
4. Use **Deselect All** to clear all checkboxes

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
   If no `.txt` file exists, the filename is used as the caption (underscores and hyphens are replaced with spaces).

### Running Training

1. Switch to **SDXL / SD 1.5** architecture and load an SDXL model
2. Go to the **LoRA Training** tab
3. Enter the path to your training images folder
4. Give your LoRA a name (e.g. `oil-painting-style`)
5. Adjust settings if needed:
   - **Steps** (default 500) — more steps = better learning but risk of overfitting. 300–1000 for most cases.
   - **Learning Rate** (default 1e-4) — lower = more stable training. The slider uses log scale (1e-5 to 1e-3).
   - **LoRA Rank** (default 4) — higher rank = more capacity but larger file. 4–16 is typical.
6. Click **Start Training** — progress bar and loss values update in real time via WebSocket
7. Click **Stop** to interrupt training early — the partial LoRA is still saved

Training saves a `.safetensors` file to the `loras/sdxl/` folder. Decreasing loss values indicate the model is learning.

### Using a Trained LoRA

1. On any generation tab, expand the **LoRA** accordion
2. Select your LoRA from the **LoRA 1** dropdown (and optionally a second from **LoRA 2**)
3. Adjust each LoRA's weight (0.0–1.5) to control how strongly the style is applied
4. Generate as normal

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Start generation (current tab) |
| `Ctrl+S` | Save current output (prevents browser save dialog) |
| `Escape` | Close lightbox or stop generation |

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
├── video_pipeline.py       # Video generation pipeline (WAN 2.1 T2V + I2V)
├── cogvideox_pipeline.py   # CogVideoX video pipeline (T2V + I2V, fp16 diffusion + fp32 VAE decode)
├── video_chunker.py        # VRAM-safe video generation (chunked VAE decode)
├── animatediff_pipeline.py # Image animation pipeline (AnimateDiff + SparseCtrl)
├── controlnet_pipeline.py  # ControlNet pipeline (borrows base model components)
├── civitai_browser.py      # CivitAI model search and download
├── upscaler.py             # AI upscaler inference (Spandrel)
├── prompt_parser.py        # Weighted prompt syntax parser
├── training.py             # LoRA fine-tuning (SDXL)
├── config.py               # Settings and defaults
├── requirements.txt        # Python dependencies
├── start.bat               # Windows launcher
├── default_positive.txt    # Default positive prompt
├── default_negative.txt    # Default negative prompt
├── profiles/               # Saved prompt profiles (auto-created)
├── models/                 # Base models (per-architecture subdirectories, auto-created)
│   ├── sdxl/               # SDXL / SD 1.5 checkpoints
│   ├── pony/               # Pony architecture models
│   ├── illustrious/        # Illustrious architecture models
│   ├── flux/               # Flux architecture models
│   ├── krea2/              # Krea 2 architecture models
│   │   └── _encoders/      # Auto-cached text encoder + VAE (single-file loading)
│   ├── wan/                # WAN 2.1 T2V video models
│   ├── wan_i2v/            # WAN 2.1 I2V video models
│   ├── cogvideox/          # CogVideoX video models (T2V + I2V)
│   ├── animatediff/        # AnimateDiff components (base model, motion adapter, SparseCtrl)
│   ├── controlnet/         # ControlNet models (.safetensors or diffusers dirs)
│   └── vaes/               # Custom VAE files (.safetensors or diffusers dirs)
├── upscalers/              # Upscaler model files (auto-created)
├── loras/                  # LoRA adapter files (per-architecture subdirectories, auto-created)
│   ├── sdxl/               # SDXL / SD 1.5 LoRAs (+ JSON metadata sidecars)
│   ├── pony/               # Pony LoRAs
│   ├── illustrious/        # Illustrious LoRAs
│   ├── flux/               # Flux LoRAs
│   └── krea2/              # Krea 2 LoRAs
└── outputs/                # Saved images and videos (+ JSON sidecar files)
```

## Troubleshooting

**"CUDA not available" / very slow generation**
- Ensure you installed torch with the CUDA index URL (see Setup)
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

**Out of memory errors (images)**
- Reduce batch size (each additional image needs more VRAM)
- Reduce image dimensions (try 768x768 or 512x512)
- Reduce inference steps
- Close other GPU-intensive applications

**Out of memory errors (video)**
- The 14B model uses 4-bit quantization + CPU offloading automatically
- VAE decode is chunked into small batches to reduce peak VRAM
- If the 14B model still OOMs, use the 1.3B Lite model instead
- Ensure no image model is loaded when generating video (switching models unloads the other automatically)
- Check the VRAM Estimate display before generating — it shows estimated vs available VRAM in real-time

**Model not showing in dropdown**
- Ensure the model is in the correct architecture subfolder under `models/`
- Click the dropdown to refresh the list

**Upscaler not showing in dropdown**
- Ensure the `.pth` or `.safetensors` file is in the `upscalers/` folder
- Click the dropdown to refresh the list

**Model download fails**
- Ensure you have internet for the first run only
- If interrupted, delete the `models/sdxl/` folder and try again

**Video generation hangs or is very slow**
- Ensure `bitsandbytes` is installed (`pip install bitsandbytes>=0.43.0`)
- The 14B model uses CPU offloading and is expected to take several minutes
- The 1.3B model typically takes 1–2 minutes

**Training fails with "requires an SDXL model"**
- LoRA training only works with SDXL models. Switch to SDXL / SD 1.5 architecture and load a model before training.

**Krea 2 single-file: "text encoder not found"**
- Internet is needed on first load to download the text encoder + VAE (~9GB). These are cached in `models/krea2/_encoders/` for offline use afterward.

**Krea 2: "FP8-scaled checkpoints..." error**
- FP8-scaled checkpoints (e.g. `_fp8_scaled` variants) use a quantization format incompatible with diffusers. Use a bf16 or non-scaled fp8 checkpoint instead.

**ControlNet tab is dimmed/disabled**
- ControlNet is not supported with Krea 2. Switch to any other architecture (SDXL, SD 1.5, Pony, Illustrious, or Flux).

**ControlNet preprocessor is slow on first use**
- Preprocessor detector models (~100-300MB each) are downloaded from HuggingFace on first use. Subsequent runs use the cached model and are instant.

**WAN I2V model not showing**
- WAN I2V models go in `models/wan_i2v/`, not `models/wan/`. They use a different transformer architecture and cannot be mixed with T2V models.
