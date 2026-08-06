"""
ImaGen — FastAPI backend server.

Replaces the Gradio UI (app.py) with a REST API + WebSocket progress.
All generation pipelines (pipeline.py, flux_pipeline.py, etc.) are unchanged.
"""

import asyncio
import base64
import io
import json
import os
import sys
import signal
import traceback
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from pipeline import ImageGenerator, SCHEDULER_NAMES
from upscaler import Upscaler
from video_chunker import generate_video_chunked
import civitai_browser
import training

# ── Prompting guides (per architecture) ──────────────────────────────────────

PROMPTING_GUIDES = {
    "SDXL / SD 1.5": (
        "<b>SDXL / SD 1.5</b> — Natural language or comma-separated tags. "
        "Use <code>[word:1.5]</code> for emphasis.<br><br>"
        "<b>Positive:</b> descriptive scene, lighting, style, "
        "<code>masterpiece, best quality, detailed</code><br>"
        "<b>Negative:</b> <code>worst quality, low quality, blurry, deformed, watermark, text</code><br>"
        "<b>Settings:</b> CFG 5–9 · 20–40 steps"
    ),
    "Pony": (
        "<b>Pony V6</b> — Tag-based prompts (Danbooru-style), NOT natural language sentences.<br><br>"
        "<b>Positive:</b> <code>score_9, score_8_up, score_7_up, source_anime, "
        "[subject tags], [scene tags]</code><br>"
        "Source options: <code>source_anime</code>, <code>source_pony</code>, <code>source_furry</code>, "
        "<code>source_cartoon</code>, <code>source_filmmaker</code><br>"
        "<b>Negative:</b> <code>score_4, score_3, score_2, score_1, "
        "worst quality, low quality, blurry</code><br>"
        "<b>Settings:</b> CFG 2–4 · 20–30 steps · Euler Ancestral"
    ),
    "Illustrious": (
        "<b>Illustrious</b> — Tag-based prompts (Danbooru tags), similar to Pony "
        "but without score tags.<br><br>"
        "<b>Positive:</b> <code>masterpiece, best quality, [subject tags], [scene tags]</code><br>"
        "<b>Negative:</b> <code>worst quality, low quality, blurry, bad anatomy</code><br>"
        "<b>Settings:</b> CFG 4–7 · 25–35 steps · Euler Ancestral"
    ),
    "Flux": (
        "<b>Flux</b> — Natural language descriptions (full sentences work best).<br><br>"
        "<b>Positive:</b> Describe the scene naturally with specific details.<br>"
        "<b>Negative:</b> Not supported — Flux ignores negative prompts entirely.<br>"
        "<b>Note:</b> Prompt weighting <code>[word:1.5]</code> is also not supported.<br>"
        "<b>Settings:</b> CFG 3–4 · 20–30 steps"
    ),
    "Krea 2": (
        "<b>Krea 2</b> — Natural language descriptions with rich detail.<br><br>"
        "<b>Positive:</b> Describe the scene naturally. Krea 2 excels at aesthetic, "
        "stylistic imagery.<br>"
        "<b>Negative:</b> Not supported — Krea 2 Turbo does not use classifier-free "
        "guidance.<br>"
        "<b>Note:</b> Prompt weighting <code>[word:1.5]</code> is not supported. "
        "Image-to-image and inpainting are not yet available.<br>"
        "<b>Settings:</b> CFG 0 · 8 steps (Turbo)"
    ),
}

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="ImaGen", version="2.0")

# Mount static files
app.mount("/static", StaticFiles(directory=str(config.PROJECT_ROOT / "static")), name="static")

# ── Global state ─────────────────────────────────────────────────────────────

generators: dict = {"SDXL / SD 1.5": ImageGenerator()}
_active_arch: str = "SDXL / SD 1.5"
_pending_download_model: str | None = None

# Video generators (lazy-loaded)
video_generator = None
cogvideox_generator = None
animatediff_generator = None
_active_video_arch: str = "WAN"

upscaler_instance = Upscaler()

# Last generation results (per-mode)
_last_t2i_image: Image.Image | None = None
_last_t2i_images: list[Image.Image] = []
_last_t2i_params: dict = {}
_last_batch_index: int = 0

_last_i2i_image: Image.Image | None = None
_last_i2i_params: dict = {}

_last_video_path: str | None = None
_last_anim_path: str | None = None

# LoRA training state
_lora_trainer: training.LoRATrainer | None = None
_training_in_progress: bool = False

# WebSocket connections for progress
ws_connections: list[WebSocket] = []

# ── Helpers ──────────────────────────────────────────────────────────────────


def get_generator():
    """Get the generator for the active architecture, creating if needed."""
    gen = generators.get(_active_arch)
    if gen is None:
        if _active_arch == "Pony":
            from pony_pipeline import PonyGenerator
            gen = PonyGenerator()
        elif _active_arch == "Illustrious":
            from illustrious_pipeline import IllustriousGenerator
            gen = IllustriousGenerator()
        elif _active_arch == "Flux":
            from flux_pipeline import FluxGenerator
            gen = FluxGenerator()
        elif _active_arch == "Krea 2":
            from krea2_pipeline import Krea2Generator
            gen = Krea2Generator()
        generators[_active_arch] = gen
    return gen


def _list_models():
    return get_generator().get_available_models()


def _list_loras():
    return ["None"] + get_generator().get_available_loras()


def _list_vaes():
    return get_generator().get_available_vaes()


def _get_vram_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "free_gb": round(free / (1024**3), 1),
        "total_gb": round(total / (1024**3), 1),
        "used_gb": round((total - free) / (1024**3), 1),
    }


def _resolve_seed(seed: int) -> int:
    actual = int(seed)
    if actual < 0:
        actual = torch.randint(0, 2**32, (1,)).item()
    return actual


def _build_prompt(positive_prompt: str, description: str = "") -> str:
    parts = [p for p in [positive_prompt.strip(), description.strip()] if p]
    full = ", ".join(parts)
    if not full:
        raise ValueError("Please enter a prompt.")
    return full


def _apply_loras(gen, lora1_name, lora1_weight, lora2_name, lora2_weight):
    lora_dir = config.ARCH_LORA_DIRS[_active_arch]
    lora_list = []
    if lora1_name and lora1_name != "None":
        lora_list.append((str(lora_dir / lora1_name), float(lora1_weight)))
    if lora2_name and lora2_name != "None":
        lora_list.append((str(lora_dir / lora2_name), float(lora2_weight)))
    if lora_list:
        gen.load_loras(lora_list)
    else:
        gen.unload_loras()


def _apply_upscaler(image: Image.Image, upscaler_name: str | None) -> Image.Image:
    if upscaler_name and upscaler_name != "None":
        upscaler_instance.load(upscaler_name)
        return upscaler_instance.upscale(image)
    return image


def _postprocess_single(gen, image, full_prompt, negative_prompt, guidance,
                         width, height, actual_seed, sampler, upscaler_name,
                         hires_active, hires_upscaler, hires_scale,
                         hires_denoise, hires_steps):
    """Apply hires fix and upscaler to a single image. Returns (image, interrupted)."""
    if hires_active:
        target_w = int(int(width) * hires_scale)
        target_h = int(int(height) * hires_scale)

        gen.flush_vram()

        if hires_upscaler and hires_upscaler != "Lanczos":
            upscaler_instance.load(hires_upscaler)
            image = upscaler_instance.upscale(image)
            upscaler_instance.unload()
            if image.size != (target_w, target_h):
                image = image.resize((target_w, target_h), Image.LANCZOS)
        else:
            image = image.resize((target_w, target_h), Image.LANCZOS)

        gen.flush_vram()

        image = gen.img2img(
            source_image=image,
            positive_prompt=full_prompt,
            negative_prompt=negative_prompt,
            strength=hires_denoise,
            steps=int(hires_steps),
            guidance_scale=guidance,
            seed=actual_seed,
            scheduler_name=sampler,
            offload_encoders=True,
            use_cached_embeds=True,
        )

        if gen.was_interrupted:
            return None, True

    image = _apply_upscaler(image, upscaler_name)
    return image, False


def _image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _save_image_impl(image: Image.Image, params: dict | None = None) -> str:
    if image is None:
        return "No image to save."
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.OUTPUT_DIR / f"img_{timestamp}.png"

    if params:
        png_info = PngInfo()
        png_info.add_text("ImaGen:params", json.dumps(params))
        image.save(str(path), "PNG", pnginfo=png_info)
        json_path = path.with_suffix(".json")
        json_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    else:
        image.save(str(path), "PNG")

    return str(path)


def _get_lora_trigger_words(lora_name: str) -> list[str]:
    if not lora_name or lora_name == "None":
        return []
    lora_dir = config.ARCH_LORA_DIRS.get(_active_arch, config.ARCH_LORA_DIRS["SDXL / SD 1.5"])
    sidecar = lora_dir / (Path(lora_name).stem + ".json")
    if not sidecar.exists():
        return []
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return []
    return meta.get("trained_words", [])


def _list_upscalers() -> list[str]:
    models = []
    if config.UPSCALER_DIR.exists():
        for f in sorted(config.UPSCALER_DIR.iterdir()):
            if f.suffix in (".pth", ".pt", ".safetensors", ".onnx"):
                models.append(f.name)
    return ["None"] + models


# ── Video helpers ─────────────────────────────────────────────────────────────

# Constants inlined from video pipelines (avoid heavy imports at startup)
WAN_FPS = 24
MIN_FPS = 6
MAX_FPS = 30
VIDEO_SCHEDULER_NAMES = ["UniPC", "Euler", "DPM++ 2M"]
COGVIDEOX_FPS = 8
COGVIDEOX_SCHEDULER_NAMES = ["DDIM", "DPM++"]
ANIMATEDIFF_FPS = 12
ANIMATEDIFF_MIN_FPS = 6
ANIMATEDIFF_MAX_CONTEXT = 16
ANIMATEDIFF_SCHEDULER_NAMES = ["Euler", "Euler Ancestral", "DPM++ 2M Karras", "DDIM", "UniPC"]


def _get_video_generator():
    """Lazy-load VideoGenerator on first use."""
    global video_generator
    if video_generator is None:
        from video_pipeline import VideoGenerator
        video_generator = VideoGenerator()
    return video_generator


def _get_cogvideox_generator():
    """Lazy-load CogVideoXGenerator on first use."""
    global cogvideox_generator
    if cogvideox_generator is None:
        from cogvideox_pipeline import CogVideoXGenerator
        cogvideox_generator = CogVideoXGenerator()
    return cogvideox_generator


def _get_animatediff_generator():
    """Lazy-load AnimateDiffGenerator on first use."""
    global animatediff_generator
    if animatediff_generator is None:
        from animatediff_pipeline import AnimateDiffGenerator
        animatediff_generator = AnimateDiffGenerator()
    return animatediff_generator


def _get_active_video_generator():
    """Return the video generator for the currently active video architecture."""
    if _active_video_arch == "CogVideoX":
        return _get_cogvideox_generator()
    return _get_video_generator()


def _get_active_video_lora_dir():
    """Return the LoRA directory for the active video architecture."""
    return config.VIDEO_ARCH_LORA_DIRS[_active_video_arch]


def _apply_video_loras(gen, lora1_name, lora1_weight, lora2_name, lora2_weight):
    """Apply LoRAs to a video generator using the active video lora dir."""
    lora_dir = _get_active_video_lora_dir()
    lora_list = []
    if lora1_name and lora1_name != "None":
        lora_list.append((str(lora_dir / lora1_name), float(lora1_weight)))
    if lora2_name and lora2_name != "None":
        lora_list.append((str(lora_dir / lora2_name), float(lora2_weight)))
    if lora_list:
        gen.load_loras(lora_list)
    else:
        gen.unload_loras()


def _round_video_frames(raw_frames: int) -> int:
    """Round raw frame count to a valid value for the active video architecture."""
    if _active_video_arch == "CogVideoX":
        num_frames = max(round(raw_frames / 4) * 4, 4)
        return num_frames
    else:
        k = round((raw_frames - 1) / 4)
        k = max(k, 1)
        return 4 * k + 1


def _estimate_video_vram_gb(num_frames: int, is_lite: bool = True) -> float:
    """Estimate peak VRAM usage in GB for a WAN video generation."""
    if is_lite:
        base_gb = 5.0
        per_frame_gb = 0.025
    else:
        base_gb = 7.0
        per_frame_gb = 0.02
    diffusion_gb = base_gb + num_frames * per_frame_gb
    vae_decode_gb = 2.0 + num_frames * 0.015
    if is_lite:
        peak_gb = max(diffusion_gb, base_gb + vae_decode_gb)
    else:
        peak_gb = max(diffusion_gb, vae_decode_gb + 1.0)
    return round(peak_gb, 1)


def _estimate_cogvideox_vram_gb(num_frames: int, is_2b: bool = True) -> float:
    """Estimate peak VRAM usage in GB for a CogVideoX video generation."""
    if is_2b:
        base_gb = 5.0
        per_frame_gb = 0.04
        vae_overhead_gb = 2.0 + num_frames * 0.03
        diffusion_gb = base_gb + num_frames * per_frame_gb
        peak_gb = max(diffusion_gb, base_gb + vae_overhead_gb)
    else:
        base_gb = 11.0
        per_frame_gb = 0.03
        diffusion_gb = base_gb + num_frames * per_frame_gb
        vae_decode_gb = 3.0 + num_frames * 0.02
        peak_gb = max(diffusion_gb, vae_decode_gb + 1.0)
    return round(peak_gb, 1)


def _estimate_animatediff_vram_gb(num_frames: int) -> float:
    """Estimate peak VRAM usage in GB for AnimateDiff generation."""
    base_gb = 4.5
    per_frame_gb = 0.12
    return round(base_gb + num_frames * per_frame_gb, 1)


def _video_save_impl(video_path: str | None, prefix: str = "vid") -> str:
    """Copy a temp video to the outputs directory."""
    if video_path is None:
        return ""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = config.OUTPUT_DIR / f"{prefix}_{timestamp}.mp4"
    import shutil
    shutil.copy2(video_path, str(dest))
    return str(dest)


# ── WebSocket progress ───────────────────────────────────────────────────────

async def broadcast_progress(data: dict):
    """Send progress update to all connected WebSocket clients."""
    msg = json.dumps(data)
    for ws in ws_connections[:]:
        try:
            await ws.send_text(msg)
        except Exception:
            ws_connections.remove(ws)


def sync_broadcast(data: dict):
    """Broadcast from synchronous pipeline code via the event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast_progress(data), loop)
    except RuntimeError:
        pass


@app.websocket("/ws/progress")
async def ws_progress(ws: WebSocket):
    await ws.accept()
    ws_connections.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in ws_connections:
            ws_connections.remove(ws)


# ── Serve index.html ─────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    index_path = config.PROJECT_ROOT / "static" / "index.html"
    return FileResponse(str(index_path), media_type="text/html")


# ── Status & Config endpoints ────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    gen = get_generator()
    model_name = getattr(gen, "_model_name", None)
    model_type = getattr(gen, "_model_type", None)
    vae_name = getattr(gen, "_vae_name", None) or "Default"
    loaded = gen.pipe is not None

    return {
        "architecture": _active_arch,
        "model": model_name,
        "model_type": model_type,
        "vae": vae_name,
        "loaded": loaded,
        "vram": _get_vram_info(),
        "device": str(config.DEVICE),
    }


@app.get("/api/architectures")
async def api_architectures():
    return {
        "architectures": config.ARCHITECTURES,
        "active": _active_arch,
        "defaults": config.ARCH_DEFAULTS,
        "i2i_architectures": config.I2I_ARCHITECTURES,
        "guides": PROMPTING_GUIDES,
    }


@app.post("/api/architecture")
async def api_switch_architecture(body: dict):
    global _active_arch, video_generator, cogvideox_generator, animatediff_generator

    arch_name = body.get("architecture")
    if arch_name not in config.ARCHITECTURES:
        return JSONResponse({"error": f"Unknown architecture: {arch_name}"}, status_code=400)

    if arch_name == _active_arch:
        gen = get_generator()
        return {
            "status": f"Already on {arch_name}",
            "models": _list_models(),
            "loras": _list_loras(),
            "vaes": _list_vaes(),
            "defaults": config.ARCH_DEFAULTS[arch_name],
            "loaded": gen.pipe is not None,
            "model": getattr(gen, "_model_name", None),
            "guide": PROMPTING_GUIDES.get(arch_name, ""),
        }

    # Unload current generator to free VRAM
    old_gen = generators.get(_active_arch)
    if old_gen is not None and old_gen.pipe is not None:
        old_gen.unload_model()
    if video_generator is not None:
        video_generator.unload_model()
    if cogvideox_generator is not None:
        cogvideox_generator.unload_model()
    if animatediff_generator is not None:
        animatediff_generator.unload_model()

    _active_arch = arch_name
    gen = get_generator()

    return {
        "status": f"Switched to {arch_name}",
        "models": gen.get_available_models(),
        "loras": ["None"] + gen.get_available_loras(),
        "vaes": gen.get_available_vaes(),
        "defaults": config.ARCH_DEFAULTS[arch_name],
        "loaded": False,
        "model": None,
        "schedulers": SCHEDULER_NAMES if arch_name in ("SDXL / SD 1.5", "Pony", "Illustrious") else ["Euler"],
        "supports_i2i": arch_name in config.I2I_ARCHITECTURES,
        "supports_negative": arch_name not in ("Krea 2",),
        "guide": PROMPTING_GUIDES.get(arch_name, ""),
    }


# ── Model management ────────────────────────────────────────────────────────

@app.get("/api/models")
async def api_models():
    return {"models": _list_models(), "active": getattr(get_generator(), "_model_name", None)}


@app.post("/api/model")
async def api_load_model(body: dict):
    global _pending_download_model, video_generator, cogvideox_generator, animatediff_generator

    model_name = body.get("model")
    if not model_name:
        return JSONResponse({"error": "No model specified"}, status_code=400)

    gen = get_generator()

    # Already loaded?
    if model_name == getattr(gen, "_model_name", None):
        return {"status": f"Already loaded: {model_name}", "model": model_name}

    # Check encoder download requirement (Flux/Krea2 single-file)
    if hasattr(gen, "needs_encoder_download"):
        download_msg = gen.needs_encoder_download(model_name)
        if download_msg and _pending_download_model != model_name:
            _pending_download_model = model_name
            return {
                "status": "confirm_download",
                "message": download_msg,
                "model": model_name,
            }

    _pending_download_model = None

    try:
        # Unload video generators to free VRAM
        if video_generator is not None:
            video_generator.unload_model()
        if cogvideox_generator is not None:
            cogvideox_generator.unload_model()
        if animatediff_generator is not None:
            animatediff_generator.unload_model()

        def progress_cb(msg):
            sync_broadcast({"type": "progress", "message": msg})

        gen.load_model(model_name, progress_callback=progress_cb)

        model_type = getattr(gen, "_model_type", None)
        return {
            "status": f"Loaded: {model_name} ({model_type})",
            "model": model_name,
            "model_type": model_type,
            "loras": _list_loras(),
            "vaes": _list_vaes(),
        }
    except Exception as e:
        return JSONResponse({"error": f"Failed to load {model_name}: {e}"}, status_code=500)


# ── LoRA management ──────────────────────────────────────────────────────────

@app.get("/api/loras")
async def api_loras():
    return {"loras": _list_loras()}


@app.get("/api/lora-triggers")
async def api_lora_triggers(name: str = ""):
    return {"triggers": _get_lora_trigger_words(name)}


# ── VAE management ───────────────────────────────────────────────────────────

@app.get("/api/vaes")
async def api_vaes():
    return {"vaes": _list_vaes()}


@app.post("/api/vae")
async def api_load_vae(body: dict):
    vae_name = body.get("vae", "Default")
    gen = get_generator()
    try:
        def progress_cb(msg):
            sync_broadcast({"type": "progress", "message": msg})
        gen.load_vae(vae_name if vae_name != "Default" else None, progress_callback=progress_cb)
        return {"status": f"VAE: {vae_name}", "vae": vae_name}
    except Exception as e:
        return JSONResponse({"error": f"Failed to load VAE: {e}"}, status_code=500)


# ── Upscalers ────────────────────────────────────────────────────────────────

@app.get("/api/upscalers")
async def api_upscalers():
    return {"upscalers": _list_upscalers()}


# ── Schedulers ───────────────────────────────────────────────────────────────

@app.get("/api/schedulers")
async def api_schedulers():
    arch = _active_arch
    if arch in ("Flux", "Krea 2"):
        return {"schedulers": ["Euler"]}
    return {"schedulers": SCHEDULER_NAMES}


# ── Text-to-Image generation ────────────────────────────────────────────────

@app.post("/api/generate")
async def api_generate(body: dict):
    global _last_t2i_image, _last_t2i_images, _last_t2i_params, _last_batch_index

    gen = get_generator()
    batch_size = int(body.get("batch_size", 1))

    # Auto-load model if none loaded
    if gen.pipe is None:
        models = _list_models()
        if not models:
            return JSONResponse({"error": f"No models found for {_active_arch}"}, status_code=400)
        def progress_cb(msg):
            sync_broadcast({"type": "progress", "message": msg})
        gen.load_model(models[0], progress_callback=progress_cb)

    try:
        full_prompt = _build_prompt(
            body.get("positive_prompt", ""),
            body.get("description", ""),
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    _apply_loras(
        gen,
        body.get("lora1_name"),
        body.get("lora1_weight", 1.0),
        body.get("lora2_name"),
        body.get("lora2_weight", 1.0),
    )

    actual_seed = _resolve_seed(int(body.get("seed", -1)))

    steps = int(body.get("steps", config.DEFAULT_STEPS))
    guidance = float(body.get("guidance_scale", config.DEFAULT_GUIDANCE_SCALE))
    width = int(body.get("width", config.DEFAULT_WIDTH))
    height = int(body.get("height", config.DEFAULT_HEIGHT))
    sampler = body.get("scheduler", "Euler")
    upscaler_name = body.get("upscaler")

    hires_enable = body.get("hires_enable", False)
    hires_upscaler = body.get("hires_upscaler", "Lanczos")
    hires_scale = float(body.get("hires_scale", 1.5))
    hires_denoise = float(body.get("hires_denoise", 0.4))
    hires_steps = int(body.get("hires_steps", 15))

    # Determine encoder offloading
    lora1 = body.get("lora1_name")
    lora2 = body.get("lora2_name")
    has_loras = (lora1 and lora1 != "None") or (lora2 and lora2 != "None")
    hires_active = hires_enable and hires_scale > 1.0
    heavy = has_loras or hires_active

    sync_broadcast({"type": "status", "message": "Generating..."})

    try:
        result = gen.generate(
            positive_prompt=full_prompt,
            negative_prompt=body.get("negative_prompt", ""),
            steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            seed=actual_seed,
            scheduler_name=sampler,
            offload_encoders=heavy,
            keep_encoders_offloaded=hires_active,
            batch_size=batch_size,
        )
    except Exception as e:
        sync_broadcast({"type": "error", "message": str(e)})
        return JSONResponse({"error": str(e)}, status_code=500)

    if gen.was_interrupted:
        sync_broadcast({"type": "status", "message": "Generation stopped."})
        return {"status": "interrupted", "seed": actual_seed}

    # Store params for history
    _last_t2i_params = {
        "positive_prompt": body.get("positive_prompt", ""),
        "negative_prompt": body.get("negative_prompt", ""),
        "description": body.get("description", ""),
        "steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height,
        "seed": actual_seed,
        "sampler": sampler,
        "model": gen._model_name,
        "model_type": getattr(gen, "_model_type", None),
        "architecture": _active_arch,
        "lora1": lora1 if lora1 and lora1 != "None" else None,
        "lora1_weight": float(body.get("lora1_weight", 1.0)) if lora1 and lora1 != "None" else None,
        "lora2": lora2 if lora2 and lora2 != "None" else None,
        "lora2_weight": float(body.get("lora2_weight", 1.0)) if lora2 and lora2 != "None" else None,
        "upscaler": upscaler_name if upscaler_name and upscaler_name != "None" else None,
        "hires_fix": hires_active,
        "vae": getattr(gen, "_vae_name", None) or "Default",
    }
    if hires_active:
        _last_t2i_params.update({
            "hires_upscaler": hires_upscaler,
            "hires_scale": hires_scale,
            "hires_denoise": hires_denoise,
            "hires_steps": hires_steps,
        })

    # Post-process (hires fix + upscaler)
    if batch_size > 1:
        images = result if isinstance(result, list) else [result]
        processed = []
        for img in images:
            img, interrupted = _postprocess_single(
                gen, img, full_prompt, body.get("negative_prompt", ""),
                guidance, width, height, actual_seed, sampler,
                upscaler_name, hires_active, hires_upscaler,
                hires_scale, hires_denoise, hires_steps,
            )
            if interrupted:
                break
            processed.append(img)

        if not processed:
            return {"status": "interrupted", "seed": actual_seed}

        _last_t2i_images = processed
        _last_batch_index = 0
        _last_t2i_image = processed[0]

        sync_broadcast({"type": "status", "message": f"Done. Seed: {actual_seed} | Batch: {len(processed)}"})
        return {
            "status": "ok",
            "seed": actual_seed,
            "batch_size": len(processed),
            "images": [_image_to_base64(img) for img in processed],
        }
    else:
        image = result
        image, interrupted = _postprocess_single(
            gen, image, full_prompt, body.get("negative_prompt", ""),
            guidance, width, height, actual_seed, sampler,
            upscaler_name, hires_active, hires_upscaler,
            hires_scale, hires_denoise, hires_steps,
        )

        if interrupted:
            return {"status": "interrupted", "seed": actual_seed}

        _last_t2i_image = image
        _last_t2i_images = []

        sync_broadcast({"type": "status", "message": f"Done. Seed: {actual_seed}"})
        return {
            "status": "ok",
            "seed": actual_seed,
            "images": [_image_to_base64(image)],
        }


# ── Image-to-Image generation ───────────────────────────────────────────────

@app.post("/api/img2img")
async def api_img2img(
    source_image: UploadFile = File(...),
    positive_prompt: str = Form(""),
    negative_prompt: str = Form(""),
    description: str = Form(""),
    strength: float = Form(0.7),
    steps: int = Form(30),
    guidance_scale: float = Form(7.5),
    seed: int = Form(-1),
    scheduler: str = Form("Euler"),
    lora1_name: str = Form("None"),
    lora1_weight: float = Form(1.0),
    lora2_name: str = Form("None"),
    lora2_weight: float = Form(1.0),
    upscaler: str = Form("None"),
):
    global _last_i2i_image, _last_i2i_params

    gen = get_generator()

    if _active_arch not in config.I2I_ARCHITECTURES:
        return JSONResponse(
            {"error": f"{_active_arch} does not support img2img"},
            status_code=400,
        )

    if gen.pipe is None:
        models = _list_models()
        if not models:
            return JSONResponse({"error": f"No models found for {_active_arch}"}, status_code=400)
        gen.load_model(models[0])

    try:
        full_prompt = _build_prompt(positive_prompt, description)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    _apply_loras(gen, lora1_name, lora1_weight, lora2_name, lora2_weight)
    actual_seed = _resolve_seed(seed)

    # Read source image
    img_bytes = await source_image.read()
    src = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    sync_broadcast({"type": "status", "message": "Generating (img2img)..."})

    try:
        image = gen.img2img(
            source_image=src,
            positive_prompt=full_prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=actual_seed,
            scheduler_name=scheduler,
        )
    except Exception as e:
        sync_broadcast({"type": "error", "message": str(e)})
        return JSONResponse({"error": str(e)}, status_code=500)

    if gen.was_interrupted:
        sync_broadcast({"type": "status", "message": "Generation stopped."})
        return {"status": "interrupted", "seed": actual_seed}

    image = _apply_upscaler(image, upscaler)
    _last_i2i_image = image
    _last_i2i_params = {
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "description": description,
        "strength": strength,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": actual_seed,
        "sampler": scheduler,
        "model": gen._model_name,
        "model_type": getattr(gen, "_model_type", None),
        "architecture": _active_arch,
        "mode": "img2img",
        "lora1": lora1_name if lora1_name != "None" else None,
        "lora1_weight": lora1_weight if lora1_name != "None" else None,
        "lora2": lora2_name if lora2_name != "None" else None,
        "lora2_weight": lora2_weight if lora2_name != "None" else None,
        "upscaler": upscaler if upscaler != "None" else None,
        "vae": getattr(gen, "_vae_name", None) or "Default",
    }

    sync_broadcast({"type": "status", "message": f"Done. Seed: {actual_seed}"})
    return {
        "status": "ok",
        "seed": actual_seed,
        "images": [_image_to_base64(image)],
    }


# ── Inpainting ───────────────────────────────────────────────────────────────

@app.post("/api/inpaint")
async def api_inpaint(
    source_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    positive_prompt: str = Form(""),
    negative_prompt: str = Form(""),
    description: str = Form(""),
    strength: float = Form(0.7),
    steps: int = Form(30),
    guidance_scale: float = Form(7.5),
    seed: int = Form(-1),
    scheduler: str = Form("Euler"),
    lora1_name: str = Form("None"),
    lora1_weight: float = Form(1.0),
    lora2_name: str = Form("None"),
    lora2_weight: float = Form(1.0),
):
    global _last_i2i_image, _last_i2i_params

    gen = get_generator()

    if _active_arch not in config.I2I_ARCHITECTURES:
        return JSONResponse(
            {"error": f"{_active_arch} does not support inpainting"},
            status_code=400,
        )

    if gen.pipe is None:
        models = _list_models()
        if not models:
            return JSONResponse({"error": f"No models found for {_active_arch}"}, status_code=400)
        gen.load_model(models[0])

    try:
        full_prompt = _build_prompt(positive_prompt, description)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    _apply_loras(gen, lora1_name, lora1_weight, lora2_name, lora2_weight)
    actual_seed = _resolve_seed(seed)

    src_bytes = await source_image.read()
    src = Image.open(io.BytesIO(src_bytes)).convert("RGB")

    mask_bytes = await mask_image.read()
    mask = Image.open(io.BytesIO(mask_bytes)).convert("RGB")

    sync_broadcast({"type": "status", "message": "Generating (inpaint)..."})

    try:
        image = gen.inpaint(
            source_image=src,
            mask_image=mask,
            positive_prompt=full_prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=actual_seed,
            scheduler_name=scheduler,
        )
    except Exception as e:
        sync_broadcast({"type": "error", "message": str(e)})
        return JSONResponse({"error": str(e)}, status_code=500)

    if gen.was_interrupted:
        sync_broadcast({"type": "status", "message": "Generation stopped."})
        return {"status": "interrupted", "seed": actual_seed}

    _last_i2i_image = image
    _last_i2i_params = {
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "description": description,
        "strength": strength,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": actual_seed,
        "sampler": scheduler,
        "model": gen._model_name,
        "model_type": getattr(gen, "_model_type", None),
        "architecture": _active_arch,
        "mode": "inpaint",
        "vae": getattr(gen, "_vae_name", None) or "Default",
    }

    sync_broadcast({"type": "status", "message": f"Done. Seed: {actual_seed}"})
    return {
        "status": "ok",
        "seed": actual_seed,
        "images": [_image_to_base64(image)],
    }


# ── Interrupt ────────────────────────────────────────────────────────────────

@app.post("/api/interrupt")
async def api_interrupt():
    get_generator().interrupt()
    return {"status": "Stopping..."}


# ── Save images ──────────────────────────────────────────────────────────────

@app.post("/api/save")
async def api_save(body: dict):
    mode = body.get("mode", "t2i")
    save_history = body.get("save_history", False)
    index = body.get("index", None)

    if mode == "t2i":
        if index is not None and _last_t2i_images:
            image = _last_t2i_images[int(index)]
        else:
            image = _last_t2i_image
        params = _last_t2i_params if save_history else None
    elif mode == "i2i":
        image = _last_i2i_image
        params = _last_i2i_params if save_history else None
    else:
        return JSONResponse({"error": "Unknown mode"}, status_code=400)

    if image is None:
        return JSONResponse({"error": "No image to save"}, status_code=400)

    path = _save_image_impl(image, params)
    return {"status": f"Saved", "path": path}


@app.post("/api/save-all")
async def api_save_all(body: dict):
    save_history = body.get("save_history", False)
    if not _last_t2i_images:
        return JSONResponse({"error": "No batch images to save"}, status_code=400)

    params = _last_t2i_params if save_history else None
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []
    for i, image in enumerate(_last_t2i_images, 1):
        path = config.OUTPUT_DIR / f"img_{timestamp}_{i}.png"
        if params:
            png_info = PngInfo()
            png_info.add_text("ImaGen:params", json.dumps(params))
            image.save(str(path), "PNG", pnginfo=png_info)
            json_path = path.with_suffix(".json")
            json_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
        else:
            image.save(str(path), "PNG")
        paths.append(str(path))

    return {"status": f"Saved {len(paths)} images", "paths": paths}


# ── Output files ─────────────────────────────────────────────────────────────

@app.get("/api/outputs")
async def api_outputs(filter_type: str = "All", sort_order: str = "Newest First"):
    if not config.OUTPUT_DIR.exists():
        return {"files": []}

    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    video_exts = {".mp4"}
    all_exts = image_exts | video_exts

    if filter_type == "Images":
        allowed = image_exts
    elif filter_type == "Videos":
        allowed = video_exts
    else:
        allowed = all_exts

    files = []
    for f in config.OUTPUT_DIR.iterdir():
        if f.suffix.lower() in allowed:
            stat = f.stat()
            meta = None
            sidecar = f.with_suffix(".json")
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                except Exception:
                    pass
            files.append({
                "name": f.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "has_metadata": meta is not None,
                "metadata": meta,
            })

    if sort_order == "Oldest First":
        files.sort(key=lambda x: x["modified"])
    elif sort_order == "Name A-Z":
        files.sort(key=lambda x: x["name"].lower())
    else:
        files.sort(key=lambda x: x["modified"], reverse=True)

    return {"files": files}


@app.get("/api/outputs/thumb/{filename}")
async def api_output_thumb(filename: str):
    """Return a JPEG thumbnail of the first frame of an MP4 video."""
    path = config.OUTPUT_DIR / filename
    if not path.exists() or not path.suffix.lower() == ".mp4":
        return JSONResponse({"error": "Not found or not a video"}, status_code=404)

    thumbs_dir = config.OUTPUT_DIR / ".thumbs"
    thumbs_dir.mkdir(exist_ok=True)
    thumb_path = thumbs_dir / (Path(filename).stem + ".jpg")

    if not thumb_path.exists():
        try:
            import imageio
            reader = imageio.get_reader(str(path))
            frame = reader.get_data(0)
            reader.close()
            img = Image.fromarray(frame)
            img.thumbnail((320, 320))
            img.save(str(thumb_path), "JPEG", quality=80)
        except Exception:
            return JSONResponse({"error": "Failed to generate thumbnail"}, status_code=500)

    return FileResponse(str(thumb_path), media_type="image/jpeg")


@app.get("/api/outputs/{filename}")
async def api_output_file(filename: str):
    path = config.OUTPUT_DIR / filename
    if not path.exists() or not path.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(str(path))


@app.delete("/api/outputs/{filename}")
async def api_delete_output(filename: str):
    path = config.OUTPUT_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    path.unlink()
    # Also delete JSON sidecar if it exists
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        sidecar.unlink()
    return {"status": f"Deleted {filename}"}


@app.post("/api/outputs/delete-batch")
async def api_delete_batch(body: dict):
    filenames = body.get("filenames", [])
    deleted = []
    for fn in filenames:
        path = config.OUTPUT_DIR / fn
        if path.exists():
            path.unlink()
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                sidecar.unlink()
            deleted.append(fn)
    return {"status": f"Deleted {len(deleted)} files", "deleted": deleted}


# ── Video: architecture & model management ───────────────────────────────────

@app.get("/api/video/architectures")
async def api_video_architectures():
    defaults = config.VIDEO_ARCH_DEFAULTS
    return {
        "architectures": config.VIDEO_ARCHITECTURES,
        "active": _active_video_arch,
        "defaults": {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in defaults.items()
        },
    }


@app.post("/api/video/architecture")
async def api_switch_video_architecture(body: dict):
    global _active_video_arch

    arch_name = body.get("architecture")
    if arch_name not in config.VIDEO_ARCHITECTURES:
        return JSONResponse({"error": f"Unknown video architecture: {arch_name}"}, status_code=400)

    if arch_name != _active_video_arch:
        # Unload current video model to free VRAM
        if _active_video_arch == "WAN" and video_generator is not None:
            video_generator.unload_model()
        elif _active_video_arch == "CogVideoX" and cogvideox_generator is not None:
            cogvideox_generator.unload_model()

    _active_video_arch = arch_name
    vg = _get_active_video_generator()
    defaults = config.VIDEO_ARCH_DEFAULTS[arch_name]

    return {
        "status": f"Video architecture: {arch_name}",
        "models": vg.get_available_video_models(),
        "loras": ["None"] + vg.get_available_loras(),
        "defaults": defaults,
        "loaded": vg.pipe is not None,
        "model": getattr(vg, "_model_name", None),
    }


@app.get("/api/video/models")
async def api_video_models():
    vg = _get_active_video_generator()
    return {
        "models": vg.get_available_video_models(),
        "active": getattr(vg, "_model_name", None),
    }


@app.post("/api/video/model")
async def api_load_video_model(body: dict):
    model_name = body.get("model")
    if not model_name:
        return JSONResponse({"error": "No video model specified"}, status_code=400)

    vg = _get_active_video_generator()

    if model_name == getattr(vg, "_model_name", None):
        return {"status": f"Already loaded: {model_name}", "model": model_name}

    try:
        # Unload image and other video generators to free VRAM
        old_gen = generators.get(_active_arch)
        if old_gen is not None and old_gen.pipe is not None:
            old_gen.unload_model()
        if _active_video_arch == "CogVideoX":
            if video_generator is not None:
                video_generator.unload_model()
        else:
            if cogvideox_generator is not None:
                cogvideox_generator.unload_model()
        if animatediff_generator is not None:
            animatediff_generator.unload_model()

        def progress_cb(msg):
            sync_broadcast({"type": "progress", "message": msg})

        vg.load_model(model_name, progress_callback=progress_cb)
        return {
            "status": f"Loaded: {model_name}",
            "model": model_name,
            "loras": ["None"] + vg.get_available_loras(),
        }
    except Exception as e:
        return JSONResponse({"error": f"Failed to load {model_name}: {e}"}, status_code=500)


@app.get("/api/video/loras")
async def api_video_loras():
    vg = _get_active_video_generator()
    return {"loras": ["None"] + vg.get_available_loras()}


@app.get("/api/video/vram-estimate")
async def api_video_vram_estimate(duration: float = 2, fps: int = 24):
    raw_frames = int(duration * fps)
    # Match the over-request logic used during generation
    if _active_video_arch != "CogVideoX":
        num_frames = _round_video_frames(raw_frames + 4)
    else:
        num_frames = _round_video_frames(raw_frames)

    if _active_video_arch == "CogVideoX":
        vg = _get_active_video_generator()
        is_2b = not getattr(vg, "_model_name", "") or "2b" in (getattr(vg, "_model_name", "") or "").lower()
        estimated = _estimate_cogvideox_vram_gb(num_frames, is_2b=is_2b)
    else:
        vg_name = video_generator._model_name if video_generator is not None else None
        is_lite = vg_name and "1.3B" in vg_name
        estimated = _estimate_video_vram_gb(num_frames, is_lite=is_lite)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vram = _get_vram_info()

    text = f"{raw_frames} frames | ~{estimated} GB VRAM needed"
    if vram["available"]:
        text += f" | {vram['free_gb']} GB free / {vram['total_gb']} GB total"
        if estimated > vram["free_gb"]:
            text += " — likely to crash!"
        elif estimated > vram["free_gb"] * 0.85:
            text += " — tight, may OOM"

    return {"estimate": text, "num_frames": num_frames, "estimated_gb": estimated}


# ── Video: generation ────────────────────────────────────────────────────────

@app.post("/api/video/generate")
async def api_video_generate(body: dict):
    global _last_video_path

    vg = _get_active_video_generator()
    if vg.pipe is None:
        return JSONResponse({"error": "Please select and load a video model first."}, status_code=400)

    try:
        full_prompt = _build_prompt(
            body.get("positive_prompt", ""),
            body.get("description", ""),
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    _apply_video_loras(
        vg,
        body.get("lora1_name"),
        body.get("lora1_weight", 1.0),
        body.get("lora2_name"),
        body.get("lora2_weight", 1.0),
    )

    duration = float(body.get("duration", 2))
    fps = int(body.get("fps", WAN_FPS))
    target_frames = int(duration * fps)
    # WAN's 3D VAE produces ~3 fewer frames than requested due to temporal
    # convolution boundary effects.  Over-request by 4 (one extra latent
    # temporal step) so the decoded output meets or exceeds the target,
    # then trim after decode.
    if _active_video_arch != "CogVideoX":
        num_frames = _round_video_frames(target_frames + 4)
    else:
        num_frames = _round_video_frames(target_frames)
    actual_seed = _resolve_seed(int(body.get("seed", -1)))

    print(f"[video] duration={duration}s, fps={fps}, target_frames={target_frames}, "
          f"num_frames={num_frames} (over-requested), "
          f"expected_duration={target_frames/fps:.2f}s")

    steps = int(body.get("steps", 25))
    guidance = float(body.get("guidance_scale", 9.0))
    sampler = body.get("scheduler", "UniPC")

    # VRAM safety check
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if _active_video_arch == "CogVideoX":
        is_2b = "2b" in (getattr(vg, "_model_name", "") or "").lower()
        estimated_vram = _estimate_cogvideox_vram_gb(num_frames, is_2b=is_2b)
    else:
        is_lite = getattr(vg, "_model_name", "") and "1.3B" in vg._model_name
        estimated_vram = _estimate_video_vram_gb(num_frames, is_lite=is_lite)

    sync_broadcast({
        "type": "status",
        "message": f"Generating {num_frames} frames (~{estimated_vram} GB VRAM)...",
    })

    try:
        frames = generate_video_chunked(
            video_generator=vg,
            positive_prompt=full_prompt,
            negative_prompt=body.get("negative_prompt", ""),
            num_frames_total=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=actual_seed,
            scheduler_name=sampler,
            progress_callback=lambda msg: sync_broadcast({"type": "progress", "message": msg}),
            vae_batch_frames=8,
        )
    except Exception as e:
        sync_broadcast({"type": "error", "message": str(e)})
        return JSONResponse({"error": str(e)}, status_code=500)

    if frames is None or len(frames) == 0:
        sync_broadcast({"type": "status", "message": "Generation stopped."})
        return {"status": "interrupted", "seed": actual_seed}

    # Export to temp MP4
    if _last_video_path:
        try:
            Path(_last_video_path).unlink(missing_ok=True)
        except OSError:
            pass

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    # Trim to target frame count (VAE may produce more or fewer than requested)
    if len(frames) > target_frames:
        print(f"[video] Trimming {len(frames)} → {target_frames} frames")
        frames = frames[:target_frames]
    print(f"[video] Exporting {len(frames)} frames at {fps} fps "
          f"→ {len(frames)/fps:.2f}s video")
    sync_broadcast({"type": "status", "message": "Exporting video..."})
    vg.export_video(frames, tmp.name, fps=fps)
    _last_video_path = tmp.name

    sync_broadcast({"type": "status", "message": f"Done. Seed: {actual_seed}"})
    return {
        "status": "ok",
        "seed": actual_seed,
        "num_frames": len(frames),
        "video_url": f"/api/video/preview?t={int(datetime.now().timestamp())}",
    }


@app.get("/api/video/preview")
async def api_video_preview():
    if _last_video_path and Path(_last_video_path).exists():
        return FileResponse(_last_video_path, media_type="video/mp4")
    return JSONResponse({"error": "No video available"}, status_code=404)


@app.post("/api/video/interrupt")
async def api_video_interrupt():
    vg = _get_active_video_generator()
    vg.interrupt()
    return {"status": "Stopping..."}


@app.post("/api/video/save")
async def api_video_save():
    path = _video_save_impl(_last_video_path, "vid")
    if not path:
        return JSONResponse({"error": "No video to save"}, status_code=400)
    return {"status": "Saved", "path": path}


# ── AnimateDiff: model management ─────────────────────────────────────────────

@app.get("/api/animate/models")
async def api_animate_models():
    ag = _get_animatediff_generator()
    return {
        "base_models": ag.get_available_base_models(),
        "motion_adapters": ag.get_available_motion_adapters(),
        "sparsectrls": ag.get_available_sparsectrls(),
        "loras": ["None"] + ag.get_available_loras(),
        "loaded": ag.pipe is not None,
    }


@app.post("/api/animate/load")
async def api_animate_load(body: dict):
    base_model = body.get("base_model")
    motion_adapter = body.get("motion_adapter")
    sparsectrl = body.get("sparsectrl")

    if not base_model:
        return JSONResponse({"error": "No base model selected."}, status_code=400)
    if not motion_adapter:
        return JSONResponse({"error": "No motion adapter selected."}, status_code=400)
    if not sparsectrl:
        return JSONResponse({"error": "No SparseControlNet selected."}, status_code=400)

    try:
        # Unload other models to free VRAM
        old_gen = generators.get(_active_arch)
        if old_gen is not None and old_gen.pipe is not None:
            old_gen.unload_model()
        if video_generator is not None:
            video_generator.unload_model()
        if cogvideox_generator is not None:
            cogvideox_generator.unload_model()

        ag = _get_animatediff_generator()

        def progress_cb(msg):
            sync_broadcast({"type": "progress", "message": msg})

        ag.load_model(base_model, motion_adapter, sparsectrl, progress_callback=progress_cb)
        return {
            "status": f"Loaded: AnimateDiff ({base_model})",
            "loras": ["None"] + ag.get_available_loras(),
        }
    except Exception as e:
        return JSONResponse({"error": f"Failed to load AnimateDiff: {e}"}, status_code=500)


@app.get("/api/animate/vram-estimate")
async def api_animate_vram_estimate(num_frames: int = 16, fps: int = 12):
    estimated = _estimate_animatediff_vram_gb(num_frames)
    duration = num_frames / max(fps, 1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vram = _get_vram_info()

    text = f"{num_frames} frames @ {fps} FPS = {duration:.1f}s | ~{estimated} GB VRAM"
    if vram["available"]:
        text += f" | {vram['free_gb']} GB free / {vram['total_gb']} GB total"
        if estimated > vram["free_gb"]:
            text += " — likely to crash!"
        elif estimated > vram["free_gb"] * 0.85:
            text += " — tight, may OOM"

    return {"estimate": text, "num_frames": num_frames, "estimated_gb": estimated}


@app.post("/api/animate/generate")
async def api_animate_generate(
    source_image: UploadFile = File(...),
    positive_prompt: str = Form(""),
    negative_prompt: str = Form(""),
    description: str = Form(""),
    num_frames: int = Form(16),
    fps: int = Form(12),
    steps: int = Form(25),
    guidance_scale: float = Form(7.5),
    conditioning_scale: float = Form(1.0),
    seed: int = Form(-1),
    scheduler: str = Form("Euler"),
    lora1_name: str = Form("None"),
    lora1_weight: float = Form(1.0),
    lora2_name: str = Form("None"),
    lora2_weight: float = Form(1.0),
):
    global _last_anim_path

    ag = _get_animatediff_generator()
    if ag.pipe is None:
        return JSONResponse({"error": "Please load AnimateDiff models first."}, status_code=400)

    try:
        full_prompt = _build_prompt(positive_prompt, description)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Apply LoRAs (AnimateDiff uses SDXL lora dir)
    lora_dir = config.ARCH_LORA_DIRS["SDXL / SD 1.5"]
    lora_list = []
    if lora1_name and lora1_name != "None":
        lora_list.append((str(lora_dir / lora1_name), float(lora1_weight)))
    if lora2_name and lora2_name != "None":
        lora_list.append((str(lora_dir / lora2_name), float(lora2_weight)))
    if lora_list:
        ag.load_loras(lora_list)
    else:
        ag.unload_loras()

    actual_seed = _resolve_seed(seed)

    # Read source image
    img_bytes = await source_image.read()
    src = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    estimated_vram = _estimate_animatediff_vram_gb(num_frames)
    sync_broadcast({
        "type": "status",
        "message": f"Generating {num_frames} frames (~{estimated_vram} GB VRAM)...",
    })

    try:
        frames = generate_video_chunked(
            video_generator=ag,
            positive_prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_frames_total=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=actual_seed,
            scheduler_name=scheduler,
            progress_callback=lambda msg: sync_broadcast({"type": "progress", "message": msg}),
            source_image=src,
            controlnet_conditioning_scale=conditioning_scale,
            vae_batch_frames=8,
        )
    except Exception as e:
        sync_broadcast({"type": "error", "message": str(e)})
        return JSONResponse({"error": str(e)}, status_code=500)

    if frames is None or len(frames) == 0:
        sync_broadcast({"type": "status", "message": "Generation stopped."})
        return {"status": "interrupted", "seed": actual_seed}

    # Export to temp MP4
    if _last_anim_path:
        try:
            Path(_last_anim_path).unlink(missing_ok=True)
        except OSError:
            pass

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    sync_broadcast({"type": "status", "message": "Exporting animation..."})
    ag.export_video(frames, tmp.name, fps=fps)
    _last_anim_path = tmp.name

    sync_broadcast({"type": "status", "message": f"Done. Seed: {actual_seed}"})
    return {
        "status": "ok",
        "seed": actual_seed,
        "num_frames": len(frames),
        "video_url": f"/api/animate/preview?t={int(datetime.now().timestamp())}",
    }


@app.get("/api/animate/preview")
async def api_animate_preview():
    if _last_anim_path and Path(_last_anim_path).exists():
        return FileResponse(_last_anim_path, media_type="video/mp4")
    return JSONResponse({"error": "No animation available"}, status_code=404)


@app.post("/api/animate/interrupt")
async def api_animate_interrupt():
    ag = _get_animatediff_generator()
    ag.interrupt()
    return {"status": "Stopping..."}


@app.post("/api/animate/save")
async def api_animate_save():
    path = _video_save_impl(_last_anim_path, "anim")
    if not path:
        return JSONResponse({"error": "No animation to save"}, status_code=400)
    return {"status": "Saved", "path": path}


# ── CivitAI Model Browser ────────────────────────────────────────────────────

@app.get("/api/civitai/search")
async def api_civitai_search(
    query: str = "",
    model_type: str = "All",
    base_model: str = "All",
    sort: str = "Most Downloaded",
    content_filter: str = "Show All",
    limit: int = 20,
    cursor: str = None,
):
    try:
        results, next_cursor = await asyncio.to_thread(
            civitai_browser.search_models,
            query=query,
            model_type=model_type,
            sort=sort,
            limit=limit,
            base_model=base_model,
            content_filter=content_filter,
            cursor=cursor,
        )
        return {"results": results, "next_cursor": next_cursor}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/civitai/download")
async def api_civitai_download(body: dict):
    download_url = body.get("download_url", "")
    model_type = body.get("model_type", "Checkpoint")
    base_model = body.get("base_model", "")
    filename = body.get("filename", "")
    metadata = body.get("metadata")

    if not download_url or not filename:
        return JSONResponse({"error": "download_url and filename required"}, status_code=400)

    try:
        dest_dir = civitai_browser.get_download_dir(model_type, base_model)
        api_key = civitai_browser.get_api_key()

        def progress_cb(msg):
            sync_broadcast({"type": "download", "message": msg})

        path = await asyncio.to_thread(
            civitai_browser.download_model,
            download_url=download_url,
            dest_dir=dest_dir,
            filename=filename,
            api_key=api_key or None,
            progress_callback=progress_cb,
        )

        # Save LoRA metadata sidecar if applicable
        if metadata and model_type == "LORA":
            civitai_browser.save_lora_metadata(dest_dir, filename, metadata)

        return {"status": "ok", "path": path}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/civitai/apikey")
async def api_civitai_get_key():
    return {"key": civitai_browser.get_api_key()}


@app.post("/api/civitai/apikey")
async def api_civitai_save_key(body: dict):
    key = body.get("key", "")
    civitai_browser.save_api_key(key)
    return {"status": "ok"}


# ── LoRA Training ─────────────────────────────────────────────────────────────

@app.post("/api/train/start")
async def api_train_start(body: dict):
    global _lora_trainer, _training_in_progress

    if _training_in_progress:
        raise HTTPException(400, "Training is already in progress.")

    image_dir = body.get("image_dir", "").strip()
    output_name = body.get("output_name", "").strip()
    steps = int(body.get("steps", config.TRAINING_STEPS))
    learning_rate = float(body.get("learning_rate", config.LEARNING_RATE))
    rank = int(body.get("rank", config.LORA_RANK))

    if not image_dir:
        raise HTTPException(400, "Image folder path is required.")
    if not Path(image_dir).is_dir():
        raise HTTPException(400, f"Image folder not found: {image_dir}")
    if not output_name:
        raise HTTPException(400, "Output name is required.")
    if _active_arch != "SDXL / SD 1.5":
        raise HTTPException(400, "LoRA training requires SDXL architecture. Switch to SDXL / SD 1.5 first.")

    gen = generators.get("SDXL / SD 1.5")
    if gen is None or gen.pipe is None:
        raise HTTPException(400, "No SDXL model is loaded. Load a model first.")

    _lora_trainer = training.LoRATrainer(gen)
    _training_in_progress = True

    def progress_cb(log_text):
        sync_broadcast({"type": "training", "log": log_text})

    try:
        path = await asyncio.to_thread(
            _lora_trainer.train,
            image_dir, output_name, steps, learning_rate, rank, progress_cb,
        )
        return {"status": "ok", "path": path}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        _training_in_progress = False


@app.post("/api/train/stop")
async def api_train_stop():
    if _lora_trainer is not None:
        _lora_trainer.interrupt()
    return {"status": "stopping"}


# ── Profiles ──────────────────────────────────────────────────────────────────

def _list_profiles():
    """Scan profiles/ folder and return sorted list of profile names."""
    names = set()
    if (config.PROJECT_ROOT / "default_positive.txt").exists():
        names.add("default")
    for f in config.PROFILES_DIR.glob("*_positive.txt"):
        names.add(f.name.replace("_positive.txt", ""))
    return sorted(names)


def _profile_paths(name):
    """Return (pos_path, neg_path) for a profile name."""
    if name == "default":
        return (
            config.PROJECT_ROOT / "default_positive.txt",
            config.PROJECT_ROOT / "default_negative.txt",
        )
    return (
        config.PROFILES_DIR / f"{name}_positive.txt",
        config.PROFILES_DIR / f"{name}_negative.txt",
    )


@app.get("/api/profiles")
async def api_profiles_list():
    return {"profiles": _list_profiles()}


@app.get("/api/profiles/{name}")
async def api_profile_load(name: str):
    pos_path, neg_path = _profile_paths(name)
    positive = " ".join(pos_path.read_text(encoding="utf-8").split()) if pos_path.exists() else ""
    negative = " ".join(neg_path.read_text(encoding="utf-8").split()) if neg_path.exists() else ""
    return {"positive": positive, "negative": negative}


@app.post("/api/profiles/{name}")
async def api_profile_save(name: str, body: dict):
    import re
    name = name.strip()
    name = re.sub(r'[^a-zA-Z0-9]', '', name)[:30]
    if not name:
        raise HTTPException(400, "Profile name must contain letters or numbers only.")
    positive = body.get("positive", "")
    negative = body.get("negative", "")
    pos_path, neg_path = _profile_paths(name)
    pos_path.write_text(positive, encoding="utf-8")
    neg_path.write_text(negative, encoding="utf-8")
    return {"status": "ok", "profiles": _list_profiles()}


@app.delete("/api/profiles/{name}")
async def api_profile_delete(name: str):
    if not name:
        raise HTTPException(400, "No profile specified.")
    if name == "default":
        pos_path, neg_path = _profile_paths("default")
        pos_path.write_text("", encoding="utf-8")
        neg_path.write_text("", encoding="utf-8")
    else:
        (config.PROFILES_DIR / f"{name}_positive.txt").unlink(missing_ok=True)
        (config.PROFILES_DIR / f"{name}_negative.txt").unlink(missing_ok=True)
    return {"status": "ok", "profiles": _list_profiles()}


# ── Shutdown ─────────────────────────────────────────────────────────────────

@app.post("/api/shutdown")
async def api_shutdown():
    sync_broadcast({"type": "status", "message": "Shutting down..."})
    # Schedule shutdown after response is sent
    asyncio.get_event_loop().call_later(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
    return {"status": "Shutting down..."}


# ── Run server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    url = "http://127.0.0.1:7860"

    print("=" * 60)
    print("  ImaGen v2.0 — FastAPI Backend")
    print(f"  Device: {config.DEVICE}")
    if torch.cuda.is_available():
        vram = _get_vram_info()
        print(f"  VRAM: {vram['total_gb']} GB total, {vram['free_gb']} GB free")
    print(f"  Opening {url}")
    print("=" * 60)

    # Open browser after a short delay to let the server start
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=7860,
        log_level="info",
    )
