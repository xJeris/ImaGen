# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Description

ImaGen is a fully self-contained, offline-capable AI image and video generation application. It provides text-to-image, image-to-image, inpainting, text-to-video, and image animation through a Gradio web UI. Supports SDXL / SD 1.5, Pony, Illustrious, Flux, and Krea 2 image architectures, plus WAN 2.1 and CogVideoX for video. It does not access the internet or external sources after initial model download.

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
- **flux_pipeline.py** — `FluxGenerator` class for Flux architecture models. Uses NF4 quantization for the transformer and lazy text encoder loading/caching. Single-file and diffusers directory loading.
- **krea2_pipeline.py** — `Krea2Generator` class for Krea 2 architecture (12.9B DiT). Uses `Krea2Pipeline` with sequential CPU offload. Supports both diffusers directory and single-file `.safetensors` loading (single-file requires downloading text encoder + VAE on first use, cached in `models/krea2/_encoders/`). Single-file loading auto-detects and converts ComfyUI key names to diffusers format, handles bf16/fp16/fp8 dtype casting, and keeps norm layers in fp32 for numerical stability. FP8-scaled checkpoints (e.g. Comfy-Org `fp8_scaled`) and INT8 checkpoints are rejected with a clear error — they require ComfyUI-specific inference kernels. Krea 2 Turbo: 8 steps, guidance_scale=0.0, no negative prompts, no Compel prompt weighting. bfloat16 throughout.
- **video_pipeline.py** — `VideoGenerator` class for WAN 2.1 models (1.3B lite, 14B full with 4-bit quantization). Single-pass diffusion for temporal coherence.
- **cogvideox_pipeline.py** — `CogVideoXGenerator` class for CogVideoX models (2b, 5b). Two-phase generation: diffusion in fp16 with sequential CPU offload, then manual VAE decode in float32 on GPU with spatial tiling. Works around cuDNN conv3d hangs on Windows.
- **video_chunker.py** — Chunked VAE decoding for video frames to stay within VRAM limits. Routes CogVideoX through single-pass pipeline path.
- **animatediff_pipeline.py** — AnimateDiff + SparseCtrl for SD 1.5 image animation (max 16 frames).

### Utilities
- **upscaler.py** — AI upscaling via Spandrel (Real-ESRGAN, SwinIR, ESRGAN model formats).
- **training.py** — LoRA fine-tuning for SDXL using PEFT library.
- **civitai_browser.py** — CivitAI model search and download integration. Extracts trigger words and recommended settings from API responses. Saves LoRA metadata sidecars on download.
- **preview_files.py** — Output gallery and file management.
- **prompt_parser.py** — Weighted prompt token parsing.
- **config.py** — Central settings: device detection, dtype, default parameters, directory paths.

### Key Directories
- `models/` — Pre-loaded model checkpoints (SDXL, SD 1.5, WAN, CogVideoX, AnimateDiff)
- `models/krea2/` — Krea 2 architecture models (diffusers dirs or `.safetensors` files)
- `models/krea2/_encoders/` — Auto-cached Krea 2 text encoder + VAE (for single-file loading)
- `loras/` — LoRA adapter files (auto-created)
- `loras/krea2/` — Krea 2 LoRA adapter files
- `models/vaes/` — Custom VAE files (.safetensors or diffusers directories, auto-created)
- `upscalers/` — Upscaler model files (auto-created)
- `outputs/` — Generated images and videos (auto-created, JSON sidecar files for history)
- `profiles/` — Saved prompt presets (auto-created)

## Key Technical Details

- **GPU Memory Management**: Models are offloaded/unloaded between pipelines. VAE tiling enabled for large images. 4-bit quantization (bitsandbytes) for 14B video model. CogVideoX uses sequential CPU offload for diffusion and manual float32 VAE decode with tiling to avoid cuDNN conv3d hangs on Windows.
- **Model Formats**: Supports both diffusers-format directories and single-file safetensors/ckpt checkpoints. Krea 2 single-file loading auto-converts ComfyUI-format key names (e.g. `blocks.N.attn.wq.weight` → `transformer_blocks.N.attn.to_q.weight`) and handles dtype conversion (bf16, fp16, fp8-no-scale). FP8-scaled (Comfy-Org `_fp8_scaled`) and INT8 checkpoints are not supported — they require ComfyUI's native FP8 inference path (`torch._scaled_mm`).
- **Prompt Weighting**: Uses `[token:weight]` syntax parsed by `prompt_parser.py`, then processed by Compel for attention scaling.
- **Hires Fix**: Two-pass generation — first at lower resolution, then img2img upscale pass at target resolution.
- **VAE Selection**: Custom VAEs can be placed in `models/vaes/`. The VAE dropdown appears in both T2I and I2I tabs (cross-tab synced). When a model is loaded, any previously selected custom VAE is re-applied automatically. VAE selection is disabled for Flux and Krea 2 (different VAE architectures).
- **Batch Generation**: T2I supports batch sizes 1–8 via `num_images_per_prompt`. Batch results display in the output gallery as a grid. Click an image to select it for "Save as PNG"; "Save All" button appears for batch runs. Hires fix and upscaler are applied to each image in the batch.
- **Generation History**: Opt-in via "Save with history" checkbox next to Save button. When enabled, `_save_image_impl()` writes a JSON sidecar (`img_*.json`) and embeds params in PNG `tEXt` chunks under the `ImaGen:params` key.
- **Model Browser Details**: When a tile is selected, the info panel shows trigger words, recommended settings (CFG, steps, sampler, clip skip), and links to both CivitAI and HuggingFace search. When a LoRA is downloaded, a JSON metadata sidecar is saved alongside it containing trigger words and settings.
- **LoRA Trigger Words**: When a LoRA with a metadata sidecar (`.json` next to the `.safetensors`) is selected in the T2I tab, trigger words display below the dropdown. Sidecars are auto-created on download from the Model Browser.
- **Prompting Guide**: A dynamic `prompt_guide` Markdown component below the Generate button shows architecture-specific prompting tips (tags vs natural language, score tags for Pony, etc.). Updates via a second `.change` handler on `arch_dropdown`. Content is in the `_PROMPTING_GUIDES` dict.
- **Config Defaults** (in `config.py`): 30 steps, 7.5 CFG, 1024×1024, float16 on CUDA, float32 on CPU.

## Constraints

- Windows 10/11 only (start.bat, path conventions)
- Requires NVIDIA GPU with 8GB+ VRAM (24GB recommended for video)
- Python 3.12 specifically (PyTorch/xformers compatibility)
- Fully offline after initial model downloads — no runtime internet access

## app.py Patterns & Pitfalls

### Gradio component cross-tab wiring
The Text-to-Image and Image-to-Image tabs share state (same model, same LoRAs, same architecture). Any `.change()` handler that references components from *both* tabs must be wired **after** all tabs are built (in the post-tabs section at the bottom of `build_ui()`), not inline within a tab. Inline wiring will fail because the other tab's components don't exist yet. See the `model_dropdown.change` and `i2i_model_dropdown.change` calls near the architecture switching block for the correct pattern.

### Multi-tab sync requirements
When a function updates shared state (model loading, architecture switching, VAE switching), it must return updates for **both** tabs' UI components. `switch_model` returns 8 values (4 for the calling tab + 4 to sync the other tab's status, model dropdown, and LoRA dropdowns). `switch_architecture` returns 13 values covering both tabs (statuses, model dropdowns, LoRA dropdowns, default settings, and the other tab's arch dropdown sync). `switch_vae` returns 3 values (status for calling tab, status for other tab, other tab's VAE dropdown). Keep this in sync if adding new shared UI elements.

### Encoder download confirmation (single-file models)
When a user selects a single-file Krea 2 or Flux checkpoint that requires downloading text encoder / VAE components, `switch_model` shows a warning and resets the dropdown to the previously loaded model. This reset fires the `.change()` handler again — the handler detects `_pending_download_model` is set and returns `no_change` for all outputs, preserving the warning. The user re-selects the model to confirm download. If `_pending_download_model` is not guarded correctly on the re-fire path, the warning will flash and disappear.

### CivitAI browser pagination
CivitAI uses cursor-based (forward-only) pagination. A `browse_cursor_history` state (list of cursors, one per page) enables backward navigation by re-fetching with a previous cursor. `history[0]` is always `None` (page 1). The Next handler appends; the Previous handler pops.

### Global state variables
- `_last_t2i_image` / `_last_i2i_image` — separate per tab, each tab's save button saves only its own image.
- `_last_t2i_params` / `_last_i2i_params` — generation parameters dict, used when "Save with history" is checked.
- `_last_t2i_images` / `_last_batch_index` — batch results list and selected index for the output gallery.
- `_last_video_path` / `_last_anim_path` — separate per video type, no cross-tab issue.
- `_active_arch` / `_active_video_arch` — track which pipeline architecture is selected.

### CSS
Custom CSS is applied via the `css=` parameter on `app.launch()`. Do not also inject it via `gr.HTML("<style>...")` inside `build_ui()` — that causes duplicate application.

### Pipeline interface contract
All generator classes (ImageGenerator, FluxGenerator, Krea2Generator, PonyGenerator, IllustriousGenerator, VideoGenerator, CogVideoXGenerator, AnimateDiffGenerator) must implement: `load_model()`, `unload_model()`, `get_available_models()`, `get_available_loras()`, `load_loras()`, `unload_loras()`, `interrupt()`, `was_interrupted` (property). Image generators also need `generate()`, `img2img()`, `inpaint()`, `flush_vram()`, `get_available_vaes()`, `load_vae()`. Krea2Generator raises `gr.Error` for `img2img()` and `inpaint()` (not yet supported).
