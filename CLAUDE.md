# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Description

ImaGen is a fully self-contained, offline-capable AI image and video generation application. It provides text-to-image, image-to-image, inpainting, text-to-video, and image animation. The frontend is a custom HTML/CSS/JS single-page app served by a FastAPI backend. Supports SDXL / SD 1.5, Pony, Illustrious, Flux, and Krea 2 image architectures, plus WAN 2.1 and CogVideoX for video. It does not access the internet or external sources after initial model download.

## UI Rewrite — Phased Plan

The original Gradio UI (`app.py`, ~3,277 lines) is being replaced with a FastAPI backend + vanilla HTML/CSS/JS frontend. The generation pipelines and utility modules remain unchanged. A backup of the pre-rewrite codebase exists at `c:\Users\jfait\Desktop\ImaGen_backup\`.

### Phase 0: Static Mockup — COMPLETE
- `mockup.html` — self-contained HTML mockup for visual review (inline CSS + JS)
- Layout: top purple nav bar, bottom black sub-nav bar, dark canvas area, black right sidebar with dark grey widgets
- 5 top-level tabs: Image Generator, Video Generator, Model Browser, Preview Files, LoRA Training
- Tab switching logic, collapsible accordion sections, responsive layout

### Phase 1: FastAPI Backend + Basic T2I — COMPLETE
- `server.py` — FastAPI backend with all T2I endpoints, WebSocket progress, global state management
- `static/index.html` — main HTML page with full panel structure
- `static/css/style.css` — complete stylesheet with dark theme
- `static/js/api.js` — API client with fetch wrappers and WebSocket auto-reconnect
- `static/js/app.js` — UI logic, tab switching, form handling, generation display
- Text-to-image generation working (single and batch)
- Lightbox for full-size image viewing
- Architecture-specific prompting guide in sidebar
- Auto-opens browser on server start
- Removed all Gradio dependencies (replaced `gr.Error` with `RuntimeError` in pipeline files)

### Phase 2: Image-to-Image + Inpainting — COMPLETE
- `/api/img2img` and `/api/inpaint` endpoints with multipart file upload
- Image upload widget with drag-and-drop, strength slider
- Inpaint canvas with drawing tools (brush, eraser, clear, undo)
- Sub-nav tab switching between T2I / I2I / Inpainting modes
- Source image preview with clear button

### Phase 3: Video Generation — COMPLETE
- Text-to-video (WAN + CogVideoX) with architecture/model switching
- Image animation (AnimateDiff + SparseCtrl) with source image upload
- Video-specific endpoints: `/api/video/*` and `/api/animate/*`
- HTML5 `<video>` player with controls and loop for preview
- Duration/FPS sliders, scheduler dropdowns, VRAM estimates
- Stop button for interrupting generation
- AnimateDiff triple model loading UI (base model + motion adapter + SparseCtrl)
- WAN frame over-request + trim to compensate for VAE temporal compression boundary loss
- WebSocket progress for diffusion steps and VAE decode batches

### Phase 4: Model Browser + Preview Files — COMPLETE
- CivitAI model search with cursor-based pagination, content filter, per-page control
- Model detail sidebar with description, trigger words, recommended settings, CivitAI + HuggingFace links
- Model download with WebSocket progress and LoRA metadata sidecar auto-save
- API key management (stored in `~/.imagen/civitai_key.txt`)
- CivitAI preview handling: `/width=450` static JPEG for images, `<video>` with hover-to-play for MP4 previews
- Preview Files: filter by type (Images/Videos), sort (Newest/Oldest/Name), Refresh button
- Batch file operations: Select All / Deselect All / Delete (N) with count indicator
- Video thumbnails via first-frame extraction (cached in `outputs/.thumbs/`)
- Open button for viewing files in new tab, inline video player in detail panel

### Phase 5: LoRA Training + Polish — PENDING
- Training config form and progress display
- Profile save/load system
- Keyboard shortcuts, final styling polish

## End User Needs

- Train the model on existing images (LoRA fine-tuning)
- Interface for positive/negative prompts and text descriptions
- Preview and gallery of created images
- Download/save images as PNG files
- Weighted prompt syntax: `[green curtains:1.5]`
- Upscalers, Hires Fix (two-pass generation), and LoRA adapters
- Batch generation (1–8 images per run, T2I only) with inline gallery grid for review and selection
- Optional generation history: saves metadata as JSON sidecar + PNG tEXt chunks
- Custom VAE selection from `models/vaes/` directory

## Development Commands

```bash
# Setup (Python 3.12 required, NVIDIA GPU recommended)
py -3.12 -m venv venv
source venv/Scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Run the application (FastAPI on http://127.0.0.1:7860)
python server.py

# Windows quick-start
start.bat
```

There is no test suite, linter configuration, or build step. The application is run directly with `python server.py`.

## Architecture

### Frontend (static/)
- **static/index.html** — Main HTML page. 5 top-level nav tabs (Image Generator, Video Generator, Model Browser, Preview Files, LoRA Training) with context-sensitive bottom sub-nav. Right sidebar with collapsible accordion sections. Lightbox overlay for full-size image viewing.
- **static/css/style.css** — Full stylesheet. CSS custom properties for theming (`--canvas-bg: #1e1e1e`, `--sidebar-width: 340px`, etc.). Dark theme with black-to-purple gradient background.
- **static/js/api.js** — API client. Fetch wrappers (`get`, `post`, `postForm`, `del`), WebSocket connection with auto-reconnect. Methods for all backend endpoints including video/animate.
- **static/js/app.js** — UI logic. Tab/panel switching (`switchMode`, `switchSubMode`), form handling, generation calls, image display, batch gallery, lightbox, slider sync, accordion toggles. Video/animate generation with HTML5 video player, source image upload for AnimateDiff, VRAM estimation. All generation functions auto-dismiss stale messages on start.

### Backend
- **server.py** — FastAPI app. Mounts `static/` directory, serves `index.html` at `/`. WebSocket at `/ws/progress` for real-time updates. All REST endpoints for image generation, video generation (`/api/video/*`), animation (`/api/animate/*`), model/LoRA/VAE management, file operations. Global state with lazy generator instantiation. Video generators are separate from image generators with independent architecture/model state. Auto-opens browser on startup.

### Generation Pipelines (unchanged)
- **pipeline.py** — `ImageGenerator` class. Loads SDXL or SD 1.5 models from `models/sdxl/`. Handles txt2img, img2img, inpainting. Uses Compel for weighted prompt parsing.
- **flux_pipeline.py** — `FluxGenerator` class for Flux architecture. NF4 quantization, lazy text encoder loading/caching.
- **krea2_pipeline.py** — `Krea2Generator` class for Krea 2 (12.9B DiT). Sequential CPU offload, single-file loading with ComfyUI key conversion.
- **video_pipeline.py** — `VideoGenerator` class for WAN 2.1 models from `models/wan/`. 1.3B lite and 14B full with 4-bit quantization.
- **cogvideox_pipeline.py** — `CogVideoXGenerator` class. Two-phase generation with manual float32 VAE decode.
- **video_chunker.py** — Chunked VAE decoding for video frames.
- **animatediff_pipeline.py** — AnimateDiff + SparseCtrl for SD 1.5 image animation (max 16 frames).

### Utilities (unchanged)
- **upscaler.py** — AI upscaling via Spandrel.
- **training.py** — LoRA fine-tuning for SDXL using PEFT library. Saves to `loras/sdxl/`.
- **civitai_browser.py** — CivitAI model search and download. Routes downloads to architecture-specific subdirectories.
- **preview_files.py** — Output gallery and file management.
- **prompt_parser.py** — Weighted prompt token parsing.
- **config.py** — Central settings: device detection, dtype, default parameters, directory paths, architecture mappings.

### Legacy (not used by new UI)
- **app.py** — Original Gradio web UI. Kept for reference only. Will be removed after all phases are complete.
- **mockup.html** — Static HTML mockup used during Phase 0 design.

### Key Directories

Every architecture has its own subfolder under `models/` and `loras/`:

```
models/
  sdxl/          — SDXL & SD 1.5 checkpoints
  pony/          — Pony checkpoints
  illustrious/   — Illustrious checkpoints
  flux/          — Flux checkpoints
  krea2/         — Krea 2 checkpoints (+ _encoders/ for auto-cached text encoder/VAE)
  wan/           — WAN 2.1 video models
  cogvideox/     — CogVideoX video models
  animatediff/   — AnimateDiff motion adapters + SparseCtrl
  vaes/          — Custom VAE files

loras/
  sdxl/          — SDXL & SD 1.5 LoRAs (+ JSON metadata sidecars)
  pony/          — Pony LoRAs
  illustrious/   — Illustrious LoRAs
  flux/          — Flux LoRAs
  krea2/         — Krea 2 LoRAs
  wan/           — WAN video LoRAs
  cogvideox/     — CogVideoX video LoRAs

upscalers/       — Upscaler model files (auto-created)
outputs/         — Generated images and videos (+ JSON sidecar files for history)
profiles/        — Saved prompt presets (auto-created)
```

## Key Technical Details

- **GPU Memory Management**: Models are offloaded/unloaded between pipelines. VAE tiling enabled for large images. 4-bit quantization (bitsandbytes) for 14B video model. CogVideoX uses sequential CPU offload for diffusion and manual float32 VAE decode with tiling to avoid cuDNN conv3d hangs on Windows.
- **Model Formats**: Supports both diffusers-format directories and single-file safetensors/ckpt checkpoints. Krea 2 single-file loading auto-converts ComfyUI-format key names and handles dtype conversion. FP8-scaled and INT8 checkpoints are not supported.
- **Prompt Weighting**: Uses `[token:weight]` syntax parsed by `prompt_parser.py`, then processed by Compel for attention scaling.
- **Hires Fix**: Two-pass generation — first at lower resolution, then img2img upscale pass at target resolution.
- **VAE Selection**: Custom VAEs can be placed in `models/vaes/`. VAE selection is disabled for Flux and Krea 2 (different VAE architectures).
- **Batch Generation**: T2I supports batch sizes 1–8 via `num_images_per_prompt`. Batch results display in the output gallery as a grid. Hires fix and upscaler are applied to each image in the batch.
- **Generation History**: Opt-in via "Save with history" checkbox. Writes JSON sidecar and embeds params in PNG `tEXt` chunks under the `ImaGen:params` key.
- **LoRA Trigger Words**: When a LoRA with a metadata sidecar (`.json` next to the `.safetensors`) is selected, trigger words display in the sidebar. Sidecars are auto-created on download from the Model Browser.
- **Prompting Guide**: Architecture-specific prompting tips displayed in a collapsible sidebar section. Content is in the `PROMPTING_GUIDES` dict in `server.py`.
- **WAN Frame Compensation**: WAN's 3D VAE temporal decompression produces ~3 fewer video frames than the `num_frames` passed to the pipeline (e.g., 37 requested → 34 decoded). The video generate endpoint over-requests by 4 frames (one extra latent temporal step) and trims the decoded output to the exact target frame count. This ensures the exported video matches the user's requested duration.
- **Config Defaults** (in `config.py`): 30 steps, 7.5 CFG, 1024x1024, float16 on CUDA, float32 on CPU.

## Constraints

- Windows 10/11 only (start.bat, path conventions)
- Requires NVIDIA GPU with 8GB+ VRAM (24GB recommended for video)
- Python 3.12 specifically (PyTorch/xformers compatibility)
- Fully offline after initial model downloads — no runtime internet access (except Model Browser for CivitAI)

## server.py Patterns & Pitfalls

### WebSocket progress from sync code
Pipeline callbacks run in synchronous threads. Use `sync_broadcast(msg)` which calls `asyncio.run_coroutine_threadsafe()` to push progress messages to all connected WebSocket clients from synchronous pipeline code.

### Encoder download confirmation (single-file models)
When a user selects a single-file Krea 2 or Flux checkpoint that requires downloading text encoder / VAE components, the `/api/model` endpoint returns a warning and sets `_pending_download_model`. The frontend must re-send the request to confirm the download.

### Global state variables
- `_last_t2i_image` / `_last_i2i_image` — separate per mode, each mode's save button saves only its own image.
- `_last_t2i_params` / `_last_i2i_params` — generation parameters dict, used when "Save with history" is checked.
- `_last_t2i_images` / `_last_batch_index` — batch results list and selected index for the output gallery.
- `_last_video_path` / `_last_anim_path` — separate per video type.
- `_active_arch` / `_active_video_arch` — track which pipeline architecture is selected.
- `generators` — dict of instantiated generator objects, keyed by architecture name.
- `video_generator` / `cogvideox_generator` / `animatediff_generator` — lazy-loaded video generators, separate from image generators.

### Pipeline interface contract
All generator classes (ImageGenerator, FluxGenerator, Krea2Generator, PonyGenerator, IllustriousGenerator, VideoGenerator, CogVideoXGenerator, AnimateDiffGenerator) must implement: `load_model()`, `unload_model()`, `get_available_models()`, `get_available_loras()`, `load_loras()`, `unload_loras()`, `interrupt()`, `was_interrupted` (property). Image generators also need `generate()`, `img2img()`, `inpaint()`, `flush_vram()`, `get_available_vaes()`, `load_vae()`. Krea2Generator raises `RuntimeError` for `img2img()` and `inpaint()` (not yet supported).

## Backup & Revert

A pre-UI-redesign backup of all source files exists at `c:\Users\jfait\Desktop\ImaGen_backup\`. It contains all `.py`, `.txt`, `.md`, `.bat`, and `.png` files plus empty `profiles/` and `training/` directories — no models, loras, upscalers, outputs, or venv.

To revert to the backup:
```bash
cp -r c:/Users/jfait/Desktop/ImaGen_backup/* c:/Users/jfait/Desktop/ImaGen/
```

This overwrites the source/config files in the project with the backed-up versions. Model files, outputs, and venv are unaffected since they were not included in the backup.
