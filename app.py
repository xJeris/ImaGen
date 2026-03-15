import os
import signal
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch

import config
from pipeline import ImageGenerator, SCHEDULER_NAMES
from upscaler import Upscaler
from training import LoRATrainer
from video_chunker import generate_video_chunked
from video_pipeline import (
    VideoGenerator, WAN_FPS, MIN_FPS, MAX_FPS, VIDEO_SCHEDULER_NAMES,
    estimate_video_vram_gb, get_available_vram_gb, get_total_vram_gb,
)
from animatediff_pipeline import (
    AnimateDiffGenerator, ANIMATEDIFF_SCHEDULER_NAMES,
    ANIMATEDIFF_FPS, ANIMATEDIFF_MIN_FPS, ANIMATEDIFF_MAX_FPS,
    estimate_animatediff_vram_gb,
)
from preview_files import list_output_files, get_file_info, delete_files

generator = ImageGenerator()
video_generator = VideoGenerator()
animatediff_generator = AnimateDiffGenerator()
upscaler = Upscaler()
trainer = None
_last_image = None
_last_video_path = None
_last_anim_path = None


def list_models():
    """Get available model names for the dropdown."""
    return generator.get_available_models()


def list_loras():
    """Get available LoRA names for the dropdown."""
    return ["None"] + generator.get_available_loras()


def list_upscalers():
    """Get available upscaler names for the dropdown."""
    return ["None"] + upscaler.get_available_upscalers()


def switch_model(model_name):
    """Hot-swap to a different base model."""
    if not model_name:
        return "No model selected.", gr.update(), gr.update()
    if model_name == generator._model_name:
        return f"Already loaded: {model_name}", gr.update(), gr.update()
    try:
        # Unload other models first to free VRAM for image generation.
        video_generator.unload_model()
        animatediff_generator.unload_model()
        generator.load_model(model_name, progress_callback=print)

        new_loras = list_loras()

        status = f"Loaded: {model_name} ({generator._model_type})"
        return (
            status,
            gr.update(choices=new_loras, value="None"),
            gr.update(choices=new_loras, value="None"),
        )
    except Exception as e:
        return f"Failed to load {model_name}: {e}", gr.update(), gr.update()


def shutdown_app():
    """Unload all models, free VRAM, and shut down the server."""
    generator.unload_model()
    video_generator.unload_model()
    animatediff_generator.unload_model()
    upscaler.unload()
    # Give Gradio a moment to send the response, then exit
    import threading
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    return "Shutting down..."


_profiles_dir = config.PROJECT_ROOT / "profiles"


def list_profiles():
    """Scan profiles/ folder and return list of profile names.
    Also includes 'default' if root-level default_positive.txt exists."""
    _profiles_dir.mkdir(exist_ok=True)
    names = set()
    # Check for root-level default files
    if (config.PROJECT_ROOT / "default_positive.txt").exists():
        names.add("default")
    for f in _profiles_dir.glob("*_positive.txt"):
        names.add(f.name.replace("_positive.txt", ""))
    return sorted(names) if names else []


def _profile_paths(name):
    """Return (pos_path, neg_path) for a profile name."""
    if name == "default":
        return (
            config.PROJECT_ROOT / "default_positive.txt",
            config.PROJECT_ROOT / "default_negative.txt",
        )
    return (
        _profiles_dir / f"{name}_positive.txt",
        _profiles_dir / f"{name}_negative.txt",
    )


def save_profile(name, positive, negative):
    """Save positive and negative prompts to profiles/{name}_*.txt."""
    if not name or not name.strip():
        gr.Warning("Please enter a profile name.")
        return gr.update()
    import re
    name = name.strip()
    # Sanitize: only letters, numbers; max 30 characters
    name = re.sub(r'[^a-zA-Z0-9]', '', name)[:30]
    if not name:
        gr.Warning("Profile name must contain letters or numbers only.")
        return gr.update()
    _profiles_dir.mkdir(exist_ok=True)
    pos_path, neg_path = _profile_paths(name)
    pos_path.write_text(positive, encoding="utf-8")
    neg_path.write_text(negative, encoding="utf-8")
    gr.Info(f"Profile '{name}' saved.")
    return gr.update(choices=list_profiles(), value=name)


def load_profile(name):
    """Load a profile and return (positive, negative) x 4 tabs = 8 outputs."""
    if not name:
        gr.Warning("No profile selected.")
        return (gr.update(),) * 8
    pos_path, neg_path = _profile_paths(name)
    pos = " ".join(pos_path.read_text(encoding="utf-8").split()) if pos_path.exists() else ""
    neg = " ".join(neg_path.read_text(encoding="utf-8").split()) if neg_path.exists() else ""
    gr.Info(f"Profile '{name}' loaded.")
    return (pos, neg) * 4


def delete_profile(name):
    """Delete a profile's files. Default profile gets emptied instead of removed."""
    if not name:
        gr.Warning("No profile selected.")
        return gr.update()
    if name == "default":
        # Clear contents but keep files so "default" always exists
        pos_path, neg_path = _profile_paths("default")
        pos_path.write_text("", encoding="utf-8")
        neg_path.write_text("", encoding="utf-8")
        gr.Info("Default profile cleared.")
        return gr.update(choices=list_profiles(), value=None)
    (_profiles_dir / f"{name}_positive.txt").unlink(missing_ok=True)
    (_profiles_dir / f"{name}_negative.txt").unlink(missing_ok=True)
    gr.Info(f"Profile '{name}' deleted.")
    return gr.update(choices=list_profiles(), value=None)


def _build_prompt(positive_prompt, description):
    """Build the full positive prompt from positive + description. Raises gr.Error if empty."""
    full = positive_prompt.strip()
    if description.strip():
        full = f"{full}, {description.strip()}"
    if not full:
        raise gr.Error("Please enter a prompt.")
    return full


def _apply_loras(gen, lora1_name, lora1_weight, lora2_name, lora2_weight):
    """Build LoRA list from two slots and apply to the given generator."""
    lora_list = []
    if lora1_name and lora1_name != "None":
        lora_list.append((str(config.LORA_DIR / lora1_name), lora1_weight))
    if lora2_name and lora2_name != "None":
        lora_list.append((str(config.LORA_DIR / lora2_name), lora2_weight))
    if lora_list:
        gen.load_loras(lora_list)
    else:
        gen.unload_loras()


def _resolve_seed(seed):
    """Return a concrete seed value; randomize if negative."""
    actual = int(seed)
    if actual < 0:
        actual = torch.randint(0, 2**32, (1,)).item()
    return actual


def _apply_upscaler(image, upscaler_name):
    """Apply upscaler to image if one is selected."""
    if upscaler_name and upscaler_name != "None":
        upscaler.load(upscaler_name)
        return upscaler.upscale(image)
    return image


def generate_image(
    positive_prompt, negative_prompt, description,
    steps, guidance, width, height, seed, sampler,
    lora1_name, lora1_weight, lora2_name, lora2_weight, upscaler_name,
    hires_enable, hires_upscaler, hires_scale, hires_denoise, hires_steps,
):
    global _last_image

    full_prompt = _build_prompt(positive_prompt, description)
    _apply_loras(generator, lora1_name, lora1_weight, lora2_name, lora2_weight)
    actual_seed = _resolve_seed(seed)

    # Offload text encoders when VRAM pressure is high (LoRAs or hires fix)
    has_loras = (lora1_name and lora1_name != "None") or (lora2_name and lora2_name != "None")
    hires_active = hires_enable and hires_scale > 1.0
    heavy = has_loras or hires_active

    image = generator.generate(
        positive_prompt=full_prompt,
        negative_prompt=negative_prompt,
        steps=int(steps),
        guidance_scale=guidance,
        width=int(width),
        height=int(height),
        seed=actual_seed,
        scheduler_name=sampler,
        offload_encoders=heavy,
        keep_encoders_offloaded=hires_active,  # skip GPU restore if hires follows
    )

    if generator.was_interrupted:
        return None, "Generation stopped."

    # Hires Fix: upscale then img2img second pass for real detail
    if hires_active:
        from PIL import Image
        target_w = int(int(width) * hires_scale)
        target_h = int(int(height) * hires_scale)

        # Free VRAM after first pass before upscaling
        generator.flush_vram()

        if hires_upscaler and hires_upscaler != "Lanczos":
            # Use AI upscaler model (runs on CPU if VRAM is full)
            upscaler.load(hires_upscaler)
            image = upscaler.upscale(image)
            upscaler.unload()  # free upscaler VRAM before img2img
            # Resize to exact target if upscaler scale doesn't match
            if image.size != (target_w, target_h):
                image = image.resize((target_w, target_h), Image.LANCZOS)
        else:
            # Lanczos resize (zero VRAM)
            image = image.resize((target_w, target_h), Image.LANCZOS)

        # Free VRAM before the memory-intensive high-res diffusion pass
        generator.flush_vram()

        image = generator.img2img(
            source_image=image,
            positive_prompt=full_prompt,
            negative_prompt=negative_prompt,
            strength=hires_denoise,
            steps=int(hires_steps),
            guidance_scale=guidance,
            seed=actual_seed,
            scheduler_name=sampler,
            offload_encoders=True,
            use_cached_embeds=True,  # reuse embeddings from generate()
        )

        if generator.was_interrupted:
            return None, "Generation stopped."

    # Post-process upscaler (simple enlarge, separate from hires fix)
    image = _apply_upscaler(image, upscaler_name)
    _last_image = image
    return image, f"Seed: {actual_seed}"


def stop_generation():
    """Signal the pipeline to stop after the current step."""
    generator.interrupt()
    return "Stopping..."


def save_image():
    global _last_image
    if _last_image is None:
        return "No image to save. Generate an image first."

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.OUTPUT_DIR / f"img_{timestamp}.png"
    _last_image.save(str(path), "PNG")
    return f"Saved to {path}"


def img2img_generate(
    source_image, editor_value, inpaint_enabled,
    positive_prompt, negative_prompt, description,
    strength, steps, guidance, seed, sampler,
    lora1_name, lora1_weight, lora2_name, lora2_weight, upscaler_name,
):
    global _last_image

    full_prompt = _build_prompt(positive_prompt, description)
    _apply_loras(generator, lora1_name, lora1_weight, lora2_name, lora2_weight)
    actual_seed = _resolve_seed(seed)

    if inpaint_enabled:
        # Inpainting mode — extract image and mask from editor
        if editor_value is None:
            raise gr.Error("Please upload an image in the inpainting editor.")
        bg = editor_value.get("background")
        layers = editor_value.get("layers", [])
        if bg is None:
            raise gr.Error("Please upload an image in the inpainting editor.")

        # Build mask from drawn layers (white = inpaint, black = keep)
        from PIL import Image as PILImage
        mask = PILImage.new("RGB", bg.size, (0, 0, 0))
        for layer in layers:
            if layer is not None:
                # Layer has transparency — any non-transparent pixel is masked
                if layer.mode == "RGBA":
                    alpha = layer.split()[3]
                    layer_mask = PILImage.new("RGB", bg.size, (255, 255, 255))
                    mask.paste(layer_mask, mask=alpha)
                else:
                    mask.paste(layer, (0, 0))

        image = generator.inpaint(
            source_image=bg,
            mask_image=mask,
            positive_prompt=full_prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=int(steps),
            guidance_scale=guidance,
            seed=actual_seed,
            scheduler_name=sampler,
        )
    else:
        # Normal img2img mode
        if source_image is None:
            raise gr.Error("Please upload a source image.")

        image = generator.img2img(
            source_image=source_image,
            positive_prompt=full_prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=int(steps),
            guidance_scale=guidance,
            seed=actual_seed,
            scheduler_name=sampler,
        )

    if generator.was_interrupted:
        return None, "Generation stopped."

    image = _apply_upscaler(image, upscaler_name)
    _last_image = image
    return image, f"Seed: {actual_seed}"


def start_training(image_dir, lora_name, steps, learning_rate, rank):
    global trainer
    if not image_dir or not image_dir.strip():
        raise gr.Error("Please provide a training images directory.")
    if not lora_name or not lora_name.strip():
        raise gr.Error("Please provide a name for the LoRA.")

    image_dir = image_dir.strip()
    if not Path(image_dir).is_dir():
        raise gr.Error(f"Directory not found: {image_dir}")

    if generator._model_type != "sdxl":
        raise gr.Error("LoRA training currently requires an SDXL model.")

    trainer = LoRATrainer(generator)
    log_output = []

    def on_progress(msg):
        log_output.clear()
        log_output.append(msg)

    try:
        trainer.train(
            image_dir=image_dir,
            output_name=lora_name.strip(),
            steps=int(steps),
            learning_rate=learning_rate,
            rank=int(rank),
            progress_callback=on_progress,
        )
    except Exception as e:
        return f"Training failed: {e}"

    return log_output[-1] if log_output else "Training complete."


# === Video functions ===

def video_list_models():
    return video_generator.get_available_video_models()


def video_list_loras():
    return ["None"] + video_generator.get_available_loras()


def video_switch_model(model_name):
    if not model_name:
        return "No video model selected."
    if model_name == video_generator._model_name:
        return f"Already loaded: {model_name}"
    try:
        # Unload other models first to free VRAM for video generation.
        generator.unload_model()
        animatediff_generator.unload_model()
        video_generator.load_model(model_name, progress_callback=print)
        return f"Loaded: {model_name}"
    except Exception as e:
        return f"Failed to load {model_name}: {e}"


def video_estimate_vram(duration, fps):
    """Calculate and return a VRAM estimate string for the video settings."""
    raw_frames = int(duration) * int(fps)
    k = round((raw_frames - 1) / 4)
    k = max(k, 1)
    num_frames = 4 * k + 1

    is_lite = video_generator._model_name and "1.3B" in video_generator._model_name
    estimated = estimate_video_vram_gb(num_frames, is_lite=is_lite)
    available = get_available_vram_gb()
    total = get_total_vram_gb()

    text = f"{num_frames} frames | ~{estimated} GB VRAM needed"
    if available is not None and total is not None:
        text += f" | {available} GB free / {total} GB total"
        if estimated > available:
            text += " — likely to crash!"
        elif estimated > available * 0.85:
            text += " — tight, may OOM"
    return text


def video_generate(
    positive_prompt, negative_prompt, description,
    duration, fps, steps, guidance, seed, sampler,
    lora1_name, lora1_weight, lora2_name, lora2_weight,
):
    global _last_video_path

    if video_generator.pipe is None:
        raise gr.Error("Please select and load a video model first.")

    full_prompt = _build_prompt(positive_prompt, description)
    _apply_loras(video_generator, lora1_name, lora1_weight, lora2_name, lora2_weight)

    raw_frames = int(duration) * int(fps)
    # WAN requires (num_frames - 1) divisible by 4, i.e. num_frames = 4k + 1.
    # Round to the nearest valid value.
    k = round((raw_frames - 1) / 4)
    k = max(k, 1)  # at least 5 frames
    num_frames = 4 * k + 1
    actual_seed = _resolve_seed(seed)

    # VRAM safety check — compare against free VRAM (accounts for loaded
    # LoRAs, other models, and any other GPU consumers).
    is_lite = video_generator._model_name and "1.3B" in video_generator._model_name
    estimated_vram = estimate_video_vram_gb(num_frames, is_lite=is_lite)
    available_vram = get_available_vram_gb()
    if available_vram is not None and estimated_vram > available_vram:
        total_vram = get_total_vram_gb() or available_vram
        gr.Warning(
            f"VRAM warning: ~{estimated_vram} GB needed, but only {available_vram} GB free "
            f"(of {total_vram} GB total). Chunked generation will attempt to proceed. "
            f"({num_frames} frames)"
        )

    yield None, f"Generating {num_frames} frames (~{estimated_vram} GB VRAM)..."

    frames = generate_video_chunked(
        video_generator=video_generator,
        positive_prompt=full_prompt,
        negative_prompt=negative_prompt,
        num_frames_total=num_frames,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        seed=actual_seed,
        scheduler_name=sampler,
        progress_callback=lambda msg: gr.Info(msg),
        vae_batch_frames=8,
    )

    if frames is None:
        yield None, "Generation stopped."
        return

    # Export to temp MP4 (clean up previous temp file)
    if _last_video_path:
        try:
            Path(_last_video_path).unlink(missing_ok=True)
        except OSError:
            pass
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    yield None, "Exporting video..."
    video_generator.export_video(frames, tmp.name, fps=int(fps))
    _last_video_path = tmp.name
    yield tmp.name, f"Seed: {actual_seed}"


def video_stop():
    video_generator.interrupt()
    return "Stopping..."


def video_save():
    global _last_video_path
    if _last_video_path is None:
        return "No video to save. Generate a video first."

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = config.OUTPUT_DIR / f"vid_{timestamp}.mp4"
    shutil.copy2(_last_video_path, str(dest))
    return f"Saved to {dest}"


# === AnimateDiff functions ===

def anim_list_base_models():
    return animatediff_generator.get_available_base_models()


def anim_list_motion_adapters():
    return animatediff_generator.get_available_motion_adapters()


def anim_list_sparsectrls():
    return animatediff_generator.get_available_sparsectrls()


def anim_list_loras():
    return ["None"] + animatediff_generator.get_available_loras()


def anim_load_models(base_model, motion_adapter, sparsectrl):
    if not base_model:
        return "No base model selected."
    if not motion_adapter:
        return "No motion adapter selected."
    if not sparsectrl:
        return "No SparseControlNet selected."
    try:
        # Unload other models first to free VRAM.
        generator.unload_model()
        video_generator.unload_model()
        animatediff_generator.load_model(
            base_model, motion_adapter, sparsectrl,
            progress_callback=print,
        )
        return f"Loaded: AnimateDiff ({base_model})"
    except Exception as e:
        return f"Failed to load: {e}"


ANIMATEDIFF_MAX_FRAMES = 32  # motion adapter positional embedding limit


def anim_estimate_vram(duration, fps):
    """Calculate and return a VRAM estimate string for AnimateDiff settings."""
    raw_frames = int(duration) * int(fps)
    num_frames = max(min(raw_frames, ANIMATEDIFF_MAX_FRAMES), 2)

    estimated = estimate_animatediff_vram_gb(num_frames)
    available = get_available_vram_gb()
    total = get_total_vram_gb()

    text = f"{num_frames} frames | ~{estimated} GB VRAM needed"
    if raw_frames > ANIMATEDIFF_MAX_FRAMES:
        text += f" (capped from {raw_frames} — max {ANIMATEDIFF_MAX_FRAMES})"
    if available is not None and total is not None:
        text += f" | {available} GB free / {total} GB total"
        if estimated > available:
            text += " — likely to crash!"
        elif estimated > available * 0.85:
            text += " — tight, may OOM"
    return text


def anim_generate(
    source_image, positive_prompt, negative_prompt, description,
    duration, fps, steps, guidance, conditioning_scale, seed, sampler,
    lora1_name, lora1_weight, lora2_name, lora2_weight,
):
    global _last_anim_path

    if animatediff_generator.pipe is None:
        raise gr.Error("Please load AnimateDiff models first.")

    if source_image is None:
        raise gr.Error("Please upload a source image.")

    full_prompt = _build_prompt(positive_prompt, description)
    _apply_loras(animatediff_generator, lora1_name, lora1_weight, lora2_name, lora2_weight)

    num_frames = int(duration) * int(fps)
    num_frames = max(min(num_frames, ANIMATEDIFF_MAX_FRAMES), 2)
    actual_seed = _resolve_seed(seed)

    # VRAM safety check
    estimated_vram = estimate_animatediff_vram_gb(num_frames)
    available_vram = get_available_vram_gb()
    if available_vram is not None and estimated_vram > available_vram:
        total_vram = get_total_vram_gb() or available_vram
        gr.Warning(
            f"VRAM warning: ~{estimated_vram} GB needed, but only {available_vram} GB free "
            f"(of {total_vram} GB total). Chunked generation will attempt to proceed. "
            f"({num_frames} frames)"
        )

    yield None, f"Generating {num_frames} frames (~{estimated_vram} GB VRAM)..."

    frames = generate_video_chunked(
        video_generator=animatediff_generator,
        positive_prompt=full_prompt,
        negative_prompt=negative_prompt,
        num_frames_total=num_frames,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        seed=actual_seed,
        scheduler_name=sampler,
        progress_callback=lambda msg: gr.Info(msg),
        source_image=source_image,
        controlnet_conditioning_scale=conditioning_scale,
        vae_batch_frames=8,
    )

    if frames is None:
        yield None, "Generation stopped."
        return

    # Export to temp MP4
    if _last_anim_path:
        try:
            Path(_last_anim_path).unlink(missing_ok=True)
        except OSError:
            pass
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    yield None, "Exporting video..."
    animatediff_generator.export_video(frames, tmp.name, fps=int(fps))
    _last_anim_path = tmp.name
    yield tmp.name, f"Seed: {actual_seed}"


def anim_stop():
    animatediff_generator.interrupt()
    return "Stopping..."


def anim_save():
    global _last_anim_path
    if _last_anim_path is None:
        return "No animation to save. Generate one first."

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = config.OUTPUT_DIR / f"anim_{timestamp}.mp4"
    shutil.copy2(_last_anim_path, str(dest))
    return f"Saved to {dest}"


CUSTOM_CSS = """
/* ── Global ── */
.gradio-container,
div.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}
.contain {
    gap: 1.2rem !important;
}

/* ── Header ── */
#imagen-header {
    text-align: center;
    padding: 1.2rem 0 0.6rem 0;
    border-bottom: 2px solid rgba(100, 180, 255, 0.15);
    margin-bottom: 0.8rem;
}
#imagen-header h1 {
    font-size: 2rem !important;
    letter-spacing: 0.04em;
    background: linear-gradient(135deg, #60a5fa, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.15rem !important;
}
#imagen-header p {
    opacity: 0.55;
    font-size: 0.85rem;
    margin-top: 0 !important;
}
#imagen-header p:last-of-type {
    opacity: 0.35;
    font-size: 0.72rem;
    margin-top: 0.3rem !important;
    letter-spacing: 0.02em;
}

/* ── Tabs ── */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.4rem !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    background: rgba(96, 165, 250, 0.12) !important;
    border-bottom: 2px solid #60a5fa !important;
}

/* ── Accordion headers ── */
.accordion > .label-wrap {
    font-weight: 600 !important;
    padding: 0.5rem 0.75rem !important;
    border-radius: 6px !important;
    transition: background 0.15s ease !important;
}
.accordion > .label-wrap:hover {
    background: rgba(96, 165, 250, 0.08) !important;
}

/* ── Buttons ── */
button.primary {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.25) !important;
}
button.primary:hover {
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4) !important;
    transform: translateY(-1px);
}
button.stop {
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

/* ── Input fields ── */
textarea, input[type="text"], input[type="number"] {
    border-radius: 6px !important;
    transition: border-color 0.2s ease !important;
}
textarea:focus, input[type="text"]:focus, input[type="number"]:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.15) !important;
}

/* ── Sliders ── */
input[type="range"] {
    accent-color: #60a5fa !important;
}

/* ── Tip / info text ── */
.prose {
    opacity: 0.8;
}
.prose strong {
    color: #93c5fd !important;
}

/* ── Save button ── */
button.secondary {
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    transition: all 0.2s ease !important;
}
button.secondary:hover {
    border-color: rgba(96, 165, 250, 0.3) !important;
    background: rgba(96, 165, 250, 0.06) !important;
}

/* ── Spacing tweaks ── */
.block {
    border-radius: 8px !important;
}

/* ── Shutdown button ── */
#shutdown-btn {
    position: relative;
    font-size: 1.4rem !important;
    padding: 0.3rem 0.6rem !important;
    min-width: 42px !important;
    max-width: 42px !important;
    height: 42px !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
    margin-top: 1.2rem;
}
#shutdown-btn::after {
    content: "Shut down ImaGen";
    position: absolute;
    bottom: -2rem;
    right: 0;
    background: #1e293b;
    color: #e2e8f0;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    font-size: 0.7rem;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s ease;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
#shutdown-btn:hover::after {
    opacity: 1;
}

/* ── Profile icon buttons ── */
.profile-btn {
    position: relative;
    min-width: 36px !important;
    max-width: 36px !important;
    height: 36px !important;
    padding: 0 !important;
    font-size: 1.1rem !important;
    border-radius: 6px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background: transparent !important;
    cursor: pointer;
}
.profile-btn:hover {
    border-color: rgba(96, 165, 250, 0.4) !important;
    background: rgba(96, 165, 250, 0.08) !important;
}
.profile-btn::after {
    position: absolute;
    bottom: -1.8rem;
    left: 50%;
    transform: translateX(-50%);
    background: #1e293b;
    color: #e2e8f0;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.65rem;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s ease;
    border: 1px solid rgba(255, 255, 255, 0.1);
    z-index: 10;
}
.profile-btn:hover::after {
    opacity: 1;
}
#profile-save-btn::after, #i2i-profile-save-btn::after,
#vid-profile-save-btn::after, #anim-profile-save-btn::after { content: "Save profile"; }
#profile-load-btn::after, #i2i-profile-load-btn::after,
#vid-profile-load-btn::after, #anim-profile-load-btn::after { content: "Load profile"; }

/* Prompts label row alignment */
[id$="-prompts-label"] {
    flex-grow: 1 !important;
    margin: auto 0 !important;
    padding: 0 !important;
}
[id$="-prompts-label"] p {
    margin: 0 !important;
}

/* ── Profile panel ── */
#profile-panel {
    border: 1px solid rgba(96, 165, 250, 0.15) !important;
    border-radius: 8px !important;
    padding: 0.8rem !important;
    background: rgba(30, 41, 59, 0.4) !important;
    margin-bottom: 0.5rem !important;
}

/* ── Preview Files gallery ── */
#preview-gallery .gallery-item {
    transition: all 0.15s ease !important;
    border: 2px solid transparent !important;
    border-radius: 6px !important;
}
#preview-gallery .gallery-item:hover {
    border-color: rgba(96, 165, 250, 0.4) !important;
}
#preview-gallery .gallery-item.selected {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.25) !important;
}

/* Video thumbnail overlay — shows a play icon */
.video-thumb-overlay {
    position: relative;
}
.video-thumb-overlay::after {
    content: "\\25B6";
    position: absolute;
    bottom: 0.4rem;
    right: 0.4rem;
    background: rgba(0, 0, 0, 0.6);
    color: white;
    font-size: 0.75rem;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
}
"""


def _get_system_stats():
    """Gather CPU, RAM, and VRAM info once at startup. Returns empty string on failure."""
    try:
        import platform
        try:
            import psutil
            has_psutil = True
        except ImportError:
            has_psutil = False

        # CPU
        try:
            cpu = platform.processor() or platform.machine() or "Unknown CPU"
            for remove in ["(R)", "(TM)", "CPU ", "  "]:
                cpu = cpu.replace(remove, "")
            cpu = cpu.strip()
            if has_psutil:
                cores = psutil.cpu_count(logical=False) or "?"
                threads = psutil.cpu_count(logical=True) or "?"
                cpu_info = f"{cpu} ({cores}C/{threads}T)"
            else:
                cpu_info = cpu
        except Exception:
            cpu_info = "Unknown CPU"

        # RAM
        try:
            if has_psutil:
                ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
                ram_info = f"{ram_gb} GB RAM"
            else:
                ram_info = None
        except Exception:
            ram_info = None

        # GPU / VRAM
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
                gpu_info = f"{gpu_name} — {vram_gb} GB VRAM"
            else:
                gpu_info = "No CUDA GPU detected"
        except Exception:
            gpu_info = None

        parts = [p for p in [cpu_info, ram_info, gpu_info] if p]
        return " &nbsp;|&nbsp; ".join(parts) if parts else ""
    except Exception:
        return ""


def build_ui():
    with gr.Blocks(title="ImaGen — Text to Image & Video", fill_width=True) as app:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        with gr.Row():
            with gr.Column(scale=9):
                sys_stats = _get_system_stats()
                header_text = "# ImaGen\nOffline text-to-image, image-to-image & text-to-video generation"
                if sys_stats:
                    header_text += f"\n\n{sys_stats}"
                gr.Markdown(header_text, elem_id="imagen-header")
            with gr.Column(scale=1, min_width=60):
                shutdown_btn = gr.Button(
                    "⏻",
                    variant="stop",
                    elem_id="shutdown-btn",
                )
        shutdown_status = gr.Textbox(visible=False)
        shutdown_btn.click(
            fn=shutdown_app,
            outputs=[shutdown_status],
            js="() => { document.title = 'ImaGen — Shut Down'; setTimeout(() => { document.body.innerHTML = '<h1 style=\"color:#e2e8f0;text-align:center;margin-top:40vh;font-family:sans-serif\">ImaGen has shut down. You can close this tab.</h1>'; }, 300); }",
        )

        # ── Shared Profile Panel (hidden by default) ──
        with gr.Group(visible=False, elem_id="profile-panel") as profile_panel:
            profile_name_input = gr.Textbox(
                label="Profile Name (letters and numbers only, max 30)",
                placeholder="e.g. cinematic, anime, portrait...",
                max_lines=1,
                max_length=30,
            )
            profile_save_action = gr.Button("Save Profile", variant="primary", size="sm")
            profile_dropdown = gr.Dropdown(
                choices=list_profiles(),
                label="Load Profile",
                allow_custom_value=False,
            )
            profile_load_action = gr.Button("Load Profile", size="sm")
            profile_delete_action = gr.Button("Delete Profile", size="sm")
            profile_close_btn = gr.Button("Close", size="sm", variant="stop")

        with gr.Tabs():
            # === Text to Image tab ===
            with gr.Tab("Text to Image"):
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=list_models(),
                        value=generator._model_name,
                        label="Base Model",
                        scale=3,
                    )
                    upscaler_dropdown = gr.Dropdown(
                        choices=list_upscalers(),
                        value="None",
                        label="Upscaler",
                        scale=2,
                    )
                    model_status = gr.Textbox(
                        value=f"Loaded: {generator._model_name} ({generator._model_type})",
                        label="Status",
                        interactive=False,
                        scale=2,
                    )

                model_dropdown.focus(
                    fn=lambda: gr.update(choices=list_models()),
                    outputs=[model_dropdown],
                )
                upscaler_dropdown.focus(
                    fn=lambda: gr.update(choices=list_upscalers()),
                    outputs=[upscaler_dropdown],
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            gr.Markdown("**Prompts**", elem_id="t2i-prompts-label")
                            t2i_profile_save = gr.Button("💾", elem_classes=["profile-btn"], elem_id="profile-save-btn", size="sm")
                            t2i_profile_load = gr.Button("📂", elem_classes=["profile-btn"], elem_id="profile-load-btn", size="sm")
                        positive_prompt = gr.Textbox(
                            label="Positive Prompt",
                            value=config.DEFAULT_POSITIVE,
                            placeholder="A beautiful sunset over mountains...",
                            lines=3,
                            max_lines=3,
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value=config.DEFAULT_NEGATIVE,
                            placeholder="blurry, low quality, deformed, watermark...",
                            lines=2,
                            max_lines=2,
                        )
                        description = gr.Textbox(
                            label="Description",
                            placeholder="Additional scene details (appended to positive prompt)...",
                            lines=2,
                            max_lines=2,
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            steps = gr.Slider(
                                1, 100, value=config.DEFAULT_STEPS,
                                step=1, label="Inference Steps",
                            )
                            guidance = gr.Slider(
                                1.0, 20.0, value=config.DEFAULT_GUIDANCE_SCALE,
                                step=0.5, label="Guidance Scale",
                            )
                            sampler = gr.Dropdown(
                                choices=SCHEDULER_NAMES,
                                value="Euler",
                                label="Sampler",
                            )
                            width = gr.Slider(
                                512, 1536, value=config.DEFAULT_WIDTH,
                                step=64, label="Width",
                            )
                            height = gr.Slider(
                                512, 1536, value=config.DEFAULT_HEIGHT,
                                step=64, label="Height",
                            )
                            seed = gr.Number(
                                value=config.DEFAULT_SEED,
                                label="Seed (-1 = random)",
                            )

                        with gr.Accordion("LoRA", open=False):
                            lora_dropdown_1 = gr.Dropdown(
                                choices=list_loras(),
                                value="None",
                                label="LoRA 1",
                            )
                            lora_weight_1 = gr.Slider(
                                0.0, 1.5, value=1.0,
                                step=0.05, label="LoRA 1 Weight",
                            )
                            lora_dropdown_2 = gr.Dropdown(
                                choices=list_loras(),
                                value="None",
                                label="LoRA 2",
                            )
                            lora_weight_2 = gr.Slider(
                                0.0, 1.5, value=1.0,
                                step=0.05, label="LoRA 2 Weight",
                            )

                        with gr.Accordion("Hires Fix", open=False):
                            hires_enable = gr.Checkbox(
                                label="Enable Hires Fix", value=False,
                            )
                            hires_upscaler = gr.Dropdown(
                                choices=["Lanczos"] + upscaler.get_available_upscalers(),
                                value="Lanczos",
                                label="Hires Upscaler",
                                info="Lanczos = fast, zero VRAM. Others = AI upscaler (slower, runs on CPU)",
                            )
                            hires_scale = gr.Slider(
                                1.0, 2.5, value=1.5, step=0.1,
                                label="Upscale Factor",
                                info="1.5x = 1024→1536, 2.0x = 1024→2048",
                            )
                            hires_denoise = gr.Slider(
                                0.1, 0.8, value=0.4, step=0.05,
                                label="Denoise Strength",
                                info="Lower = closer to original, higher = more new detail",
                            )
                            hires_steps = gr.Slider(
                                1, 100, value=20, step=1,
                                label="Hires Steps",
                                info="Inference steps for the second pass",
                            )

                        with gr.Row():
                            generate_btn = gr.Button("Generate", variant="primary")
                            stop_btn = gr.Button("Stop", variant="stop")
                        gr.Markdown(
                            "**Tip:** Use `[word:1.5]` for weighted prompts, "
                            "e.g. `[green curtains:1.5] in a cozy room`"
                        )

                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            label="Generated Image",
                            show_label=False,
                            type="pil",
                            interactive=False,
                        )
                        seed_display = gr.Textbox(
                            label="", interactive=False, show_label=False,
                            placeholder="Seed will appear here after generation",
                        )
                        save_btn = gr.Button("Save as PNG")
                        save_status = gr.Textbox(
                            label="", interactive=False, show_label=False,
                        )

                model_dropdown.change(
                    fn=switch_model,
                    inputs=[model_dropdown],
                    outputs=[model_status, lora_dropdown_1, lora_dropdown_2],
                )
                lora_dropdown_1.focus(
                    fn=lambda current: gr.update(choices=list_loras(), value=current),
                    inputs=[lora_dropdown_1],
                    outputs=[lora_dropdown_1],
                )
                lora_dropdown_2.focus(
                    fn=lambda current: gr.update(choices=list_loras(), value=current),
                    inputs=[lora_dropdown_2],
                    outputs=[lora_dropdown_2],
                )
                hires_upscaler.focus(
                    fn=lambda: gr.update(choices=["Lanczos"] + upscaler.get_available_upscalers()),
                    outputs=[hires_upscaler],
                )
                generate_btn.click(
                    fn=generate_image,
                    inputs=[
                        positive_prompt, negative_prompt, description,
                        steps, guidance, width, height, seed, sampler,
                        lora_dropdown_1, lora_weight_1, lora_dropdown_2, lora_weight_2,
                        upscaler_dropdown,
                        hires_enable, hires_upscaler, hires_scale, hires_denoise, hires_steps,
                    ],
                    outputs=[output_image, seed_display],
                )
                stop_btn.click(fn=stop_generation, outputs=[seed_display])
                save_btn.click(fn=save_image, outputs=[save_status])

            # === Img2Img tab ===
            with gr.Tab("Image to Image"):
                with gr.Row():
                    i2i_model_dropdown = gr.Dropdown(
                        choices=list_models(),
                        value=generator._model_name,
                        label="Base Model",
                        scale=3,
                    )
                    i2i_upscaler_dropdown = gr.Dropdown(
                        choices=list_upscalers(),
                        value="None",
                        label="Upscaler",
                        scale=2,
                    )
                    i2i_model_status = gr.Textbox(
                        value=f"Loaded: {generator._model_name} ({generator._model_type})",
                        label="Status",
                        interactive=False,
                        scale=2,
                    )

                i2i_model_dropdown.focus(
                    fn=lambda: gr.update(choices=list_models()),
                    outputs=[i2i_model_dropdown],
                )
                i2i_upscaler_dropdown.focus(
                    fn=lambda: gr.update(choices=list_upscalers()),
                    outputs=[i2i_upscaler_dropdown],
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        i2i_inpaint_enable = gr.Checkbox(
                            label="Enable Inpainting",
                            value=False,
                            info="Paint a mask over the area you want to regenerate",
                        )
                        i2i_source = gr.Image(
                            label="Source Image",
                            type="pil",
                        )
                        i2i_editor = gr.ImageEditor(
                            label="Inpaint — paint white over the area to regenerate",
                            type="pil",
                            brush=gr.Brush(
                                colors=["#FFFFFF"],
                                default_color="#FFFFFF",
                                color_mode="fixed",
                                default_size=30,
                            ),
                            eraser=gr.Eraser(default_size=30),
                            visible=False,
                        )
                        with gr.Row():
                            gr.Markdown("**Prompts**", elem_id="i2i-prompts-label")
                            i2i_profile_save = gr.Button("💾", elem_classes=["profile-btn"], elem_id="i2i-profile-save-btn", size="sm")
                            i2i_profile_load = gr.Button("📂", elem_classes=["profile-btn"], elem_id="i2i-profile-load-btn", size="sm")
                        i2i_positive = gr.Textbox(
                            label="Positive Prompt",
                            value=config.DEFAULT_POSITIVE,
                            placeholder="Describe the changes you want...",
                            lines=3,
                            max_lines=3,
                        )
                        i2i_negative = gr.Textbox(
                            label="Negative Prompt",
                            value=config.DEFAULT_NEGATIVE,
                            placeholder="blurry, low quality, deformed...",
                            lines=2,
                            max_lines=2,
                        )
                        i2i_description = gr.Textbox(
                            label="Description",
                            placeholder="Additional details...",
                            lines=2,
                            max_lines=2,
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            i2i_strength = gr.Slider(
                                0.0, 1.0, value=0.7,
                                step=0.05, label="Strength (Denoise)",
                                info="How much to change. 0.0 = keep original, 1.0 = fully reimagine.",
                            )
                            i2i_steps = gr.Slider(
                                1, 100, value=config.DEFAULT_STEPS,
                                step=1, label="Inference Steps",
                            )
                            i2i_guidance = gr.Slider(
                                1.0, 20.0, value=config.DEFAULT_GUIDANCE_SCALE,
                                step=0.5, label="Guidance Scale",
                            )
                            i2i_sampler = gr.Dropdown(
                                choices=SCHEDULER_NAMES,
                                value="Euler",
                                label="Sampler",
                            )
                            i2i_seed = gr.Number(
                                value=config.DEFAULT_SEED,
                                label="Seed (-1 = random)",
                            )

                        with gr.Accordion("LoRA", open=False):
                            i2i_lora_1 = gr.Dropdown(
                                choices=list_loras(),
                                value="None",
                                label="LoRA 1",
                            )
                            i2i_lora_weight_1 = gr.Slider(
                                0.0, 1.5, value=1.0,
                                step=0.05, label="LoRA 1 Weight",
                            )
                            i2i_lora_2 = gr.Dropdown(
                                choices=list_loras(),
                                value="None",
                                label="LoRA 2",
                            )
                            i2i_lora_weight_2 = gr.Slider(
                                0.0, 1.5, value=1.0,
                                step=0.05, label="LoRA 2 Weight",
                            )

                        with gr.Row():
                            i2i_btn = gr.Button("Generate", variant="primary")
                            i2i_stop_btn = gr.Button("Stop", variant="stop")
                        gr.Markdown(
                            "**Strength guide:** 0.3 = subtle tweaks, "
                            "0.5 = moderate changes, 0.7 = significant rework, "
                            "0.9+ = almost fully new image"
                        )

                    with gr.Column(scale=1):
                        i2i_output = gr.Image(
                            label="Result",
                            show_label=False,
                            type="pil",
                            interactive=False,
                        )
                        i2i_seed_display = gr.Textbox(
                            label="", interactive=False, show_label=False,
                            placeholder="Seed will appear here after generation",
                        )
                        i2i_save_btn = gr.Button("Save as PNG")
                        i2i_save_status = gr.Textbox(
                            label="", interactive=False, show_label=False,
                        )

                i2i_model_dropdown.change(
                    fn=switch_model,
                    inputs=[i2i_model_dropdown],
                    outputs=[i2i_model_status, i2i_lora_1, i2i_lora_2],
                )
                i2i_lora_1.focus(
                    fn=lambda current: gr.update(choices=list_loras(), value=current),
                    inputs=[i2i_lora_1],
                    outputs=[i2i_lora_1],
                )
                i2i_lora_2.focus(
                    fn=lambda current: gr.update(choices=list_loras(), value=current),
                    inputs=[i2i_lora_2],
                    outputs=[i2i_lora_2],
                )
                i2i_inpaint_enable.change(
                    fn=lambda enabled: (
                        gr.update(visible=not enabled),
                        gr.update(visible=enabled),
                    ),
                    inputs=[i2i_inpaint_enable],
                    outputs=[i2i_source, i2i_editor],
                )

                i2i_btn.click(
                    fn=img2img_generate,
                    inputs=[
                        i2i_source, i2i_editor, i2i_inpaint_enable,
                        i2i_positive, i2i_negative, i2i_description,
                        i2i_strength, i2i_steps, i2i_guidance, i2i_seed, i2i_sampler,
                        i2i_lora_1, i2i_lora_weight_1, i2i_lora_2, i2i_lora_weight_2,
                        i2i_upscaler_dropdown,
                    ],
                    outputs=[i2i_output, i2i_seed_display],
                )
                i2i_stop_btn.click(fn=stop_generation, outputs=[i2i_seed_display])
                i2i_save_btn.click(fn=save_image, outputs=[i2i_save_status])

            # === Text to Video tab ===
            with gr.Tab("Text to Video"):
                with gr.Row():
                    vid_model = gr.Dropdown(
                        choices=video_list_models(),
                        value=None,
                        label="Video Model",
                        scale=3,
                    )
                    vid_status = gr.Textbox(
                        value="No video model loaded",
                        label="Status",
                        interactive=False,
                        scale=2,
                    )

                vid_model.change(
                    fn=video_switch_model,
                    inputs=[vid_model],
                    outputs=[vid_status],
                )
                vid_model.focus(
                    fn=lambda: gr.update(choices=video_list_models()),
                    outputs=[vid_model],
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            gr.Markdown("**Prompts**", elem_id="vid-prompts-label")
                            vid_profile_save = gr.Button("💾", elem_classes=["profile-btn"], elem_id="vid-profile-save-btn", size="sm")
                            vid_profile_load = gr.Button("📂", elem_classes=["profile-btn"], elem_id="vid-profile-load-btn", size="sm")
                        vid_positive = gr.Textbox(
                            label="Positive Prompt",
                            value=config.DEFAULT_POSITIVE,
                            placeholder="A cinematic scene of...",
                            lines=3,
                            max_lines=3,
                        )
                        vid_negative = gr.Textbox(
                            label="Negative Prompt",
                            value=config.DEFAULT_NEGATIVE,
                            placeholder="blurry, low quality, static...",
                            lines=2,
                            max_lines=2,
                        )
                        vid_description = gr.Textbox(
                            label="Description",
                            placeholder="Additional scene details...",
                            lines=2,
                            max_lines=2,
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            vid_duration = gr.Slider(
                                1, 5, value=3, step=1,
                                label="Duration (seconds)",
                                info="Total frames = duration × FPS (rounded to nearest valid count)",
                            )
                            vid_fps = gr.Slider(
                                MIN_FPS, MAX_FPS, value=WAN_FPS, step=1,
                                label="Frames Per Second (FPS)",
                                info=f"{MIN_FPS}–{MAX_FPS} FPS. 24 = cinematic, 30 = smooth. Max: 5s × 30fps = 150 frames",
                            )
                            vid_vram_estimate = gr.Textbox(
                                value=video_estimate_vram(3, WAN_FPS),
                                label="VRAM Estimate",
                                interactive=False,
                            )
                            vid_steps = gr.Slider(
                                1, 100, value=30,
                                step=1, label="Inference Steps",
                            )
                            vid_guidance = gr.Slider(
                                1.0, 20.0, value=5.0,
                                step=0.5, label="Guidance Scale",
                            )
                            vid_seed = gr.Number(
                                value=config.DEFAULT_SEED,
                                label="Seed (-1 = random)",
                            )
                            vid_sampler = gr.Dropdown(
                                choices=VIDEO_SCHEDULER_NAMES,
                                value="UniPC",
                                label="Sampler",
                            )

                        with gr.Accordion("LoRA", open=False):
                            vid_lora_1 = gr.Dropdown(
                                choices=video_list_loras(),
                                value="None",
                                label="LoRA 1",
                            )
                            vid_lora_weight_1 = gr.Slider(
                                0.0, 1.5, value=1.0,
                                step=0.05, label="LoRA 1 Weight",
                            )
                            vid_lora_2 = gr.Dropdown(
                                choices=video_list_loras(),
                                value="None",
                                label="LoRA 2",
                            )
                            vid_lora_weight_2 = gr.Slider(
                                0.0, 1.5, value=1.0,
                                step=0.05, label="LoRA 2 Weight",
                            )

                        with gr.Row():
                            vid_btn = gr.Button("Generate", variant="primary")
                            vid_stop_btn = gr.Button("Stop", variant="stop")

                    with gr.Column(scale=1):
                        vid_output = gr.Video(
                            label="Generated Video",
                            show_label=False,
                        )
                        vid_seed_display = gr.Textbox(
                            label="", interactive=False, show_label=False,
                            placeholder="Seed will appear here after generation",
                        )
                        vid_save_btn = gr.Button("Save Video")
                        vid_save_status = gr.Textbox(
                            label="", interactive=False, show_label=False,
                        )

                vid_duration.change(
                    fn=video_estimate_vram,
                    inputs=[vid_duration, vid_fps],
                    outputs=[vid_vram_estimate],
                )
                vid_fps.change(
                    fn=video_estimate_vram,
                    inputs=[vid_duration, vid_fps],
                    outputs=[vid_vram_estimate],
                )

                vid_lora_1.focus(
                    fn=lambda current: gr.update(choices=video_list_loras(), value=current),
                    inputs=[vid_lora_1],
                    outputs=[vid_lora_1],
                )
                vid_lora_2.focus(
                    fn=lambda current: gr.update(choices=video_list_loras(), value=current),
                    inputs=[vid_lora_2],
                    outputs=[vid_lora_2],
                )

                vid_btn.click(
                    fn=video_generate,
                    inputs=[
                        vid_positive, vid_negative, vid_description,
                        vid_duration, vid_fps, vid_steps, vid_guidance, vid_seed, vid_sampler,
                        vid_lora_1, vid_lora_weight_1, vid_lora_2, vid_lora_weight_2,
                    ],
                    outputs=[vid_output, vid_seed_display],
                )
                vid_stop_btn.click(fn=video_stop, outputs=[vid_seed_display])
                vid_save_btn.click(fn=video_save, outputs=[vid_save_status])

            # === Animate Image tab ===
            with gr.Tab("Animate Image"):
                gr.Markdown(
                    "Animate a still image using **AnimateDiff + SparseCtrl**. "
                    "Requires SD 1.5 base model, motion adapter, and SparseControlNet "
                    "in `models/animatediff/`."
                )
                with gr.Row():
                    anim_base_model = gr.Dropdown(
                        choices=anim_list_base_models(),
                        value=None,
                        label="SD 1.5 Base Model",
                        scale=2,
                    )
                    anim_motion_adapter = gr.Dropdown(
                        choices=anim_list_motion_adapters(),
                        value=None,
                        label="Motion Adapter",
                        scale=2,
                    )
                    anim_sparsectrl = gr.Dropdown(
                        choices=anim_list_sparsectrls(),
                        value=None,
                        label="SparseControlNet",
                        scale=2,
                    )
                with gr.Row():
                    anim_load_btn = gr.Button("Load Models", variant="primary", scale=1)
                    anim_status = gr.Textbox(
                        value="No AnimateDiff models loaded",
                        label="Status",
                        interactive=False,
                        scale=3,
                    )

                anim_load_btn.click(
                    fn=anim_load_models,
                    inputs=[anim_base_model, anim_motion_adapter, anim_sparsectrl],
                    outputs=[anim_status],
                )
                anim_base_model.focus(
                    fn=lambda: gr.update(choices=anim_list_base_models()),
                    outputs=[anim_base_model],
                )
                anim_motion_adapter.focus(
                    fn=lambda: gr.update(choices=anim_list_motion_adapters()),
                    outputs=[anim_motion_adapter],
                )
                anim_sparsectrl.focus(
                    fn=lambda: gr.update(choices=anim_list_sparsectrls()),
                    outputs=[anim_sparsectrl],
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        anim_source = gr.Image(
                            label="Source Image",
                            type="pil",
                        )
                        with gr.Row():
                            gr.Markdown("**Prompts**", elem_id="anim-prompts-label")
                            anim_profile_save = gr.Button("💾", elem_classes=["profile-btn"], elem_id="anim-profile-save-btn", size="sm")
                            anim_profile_load = gr.Button("📂", elem_classes=["profile-btn"], elem_id="anim-profile-load-btn", size="sm")
                        anim_positive = gr.Textbox(
                            label="Positive Prompt",
                            value=config.DEFAULT_POSITIVE,
                            placeholder="Describe the motion or scene...",
                            lines=3,
                            max_lines=3,
                        )
                        anim_negative = gr.Textbox(
                            label="Negative Prompt",
                            value=config.DEFAULT_NEGATIVE,
                            placeholder="blurry, low quality, static...",
                            lines=2,
                            max_lines=2,
                        )
                        anim_description = gr.Textbox(
                            label="Description",
                            placeholder="Additional motion details...",
                            lines=2,
                            max_lines=2,
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            anim_duration = gr.Slider(
                                1, 5, value=2, step=1,
                                label="Duration (seconds)",
                                info="Total frames = duration × FPS",
                            )
                            anim_fps = gr.Slider(
                                ANIMATEDIFF_MIN_FPS, ANIMATEDIFF_MAX_FPS,
                                value=ANIMATEDIFF_FPS, step=1,
                                label="Frames Per Second (FPS)",
                                info=f"{ANIMATEDIFF_MIN_FPS}–{ANIMATEDIFF_MAX_FPS} FPS. Max: 5s × 30fps = 150 frames",
                            )
                            anim_vram_estimate = gr.Textbox(
                                value=anim_estimate_vram(2, ANIMATEDIFF_FPS),
                                label="VRAM Estimate",
                                interactive=False,
                            )
                            anim_steps = gr.Slider(
                                1, 100, value=25,
                                step=1, label="Inference Steps",
                            )
                            anim_guidance = gr.Slider(
                                1.0, 20.0, value=7.5,
                                step=0.5, label="Guidance Scale",
                            )
                            anim_conditioning = gr.Slider(
                                0.0, 2.0, value=1.5,
                                step=0.05, label="Image Conditioning Scale",
                                info="How strongly to follow the source image. Higher = more faithful.",
                            )
                            anim_seed = gr.Number(
                                value=config.DEFAULT_SEED,
                                label="Seed (-1 = random)",
                            )
                            anim_sampler = gr.Dropdown(
                                choices=ANIMATEDIFF_SCHEDULER_NAMES,
                                value="DPM++ 2M Karras",
                                label="Sampler",
                            )

                        with gr.Accordion("LoRA", open=False):
                            anim_lora_1 = gr.Dropdown(
                                choices=anim_list_loras(),
                                value="None",
                                label="LoRA 1",
                            )
                            anim_lora_weight_1 = gr.Slider(
                                0.0, 1.5, value=1.0,
                                step=0.05, label="LoRA 1 Weight",
                            )
                            anim_lora_2 = gr.Dropdown(
                                choices=anim_list_loras(),
                                value="None",
                                label="LoRA 2",
                            )
                            anim_lora_weight_2 = gr.Slider(
                                0.0, 1.5, value=1.0,
                                step=0.05, label="LoRA 2 Weight",
                            )

                        with gr.Row():
                            anim_btn = gr.Button("Animate", variant="primary")
                            anim_stop_btn = gr.Button("Stop", variant="stop")
                        gr.Markdown(
                            "**Tip:** Use descriptive motion prompts like "
                            "`wind blowing through hair, gentle swaying` for better results."
                        )

                    with gr.Column(scale=1):
                        anim_output = gr.Video(
                            label="Animated Result",
                            show_label=False,
                        )
                        anim_seed_display = gr.Textbox(
                            label="", interactive=False, show_label=False,
                            placeholder="Seed will appear here after generation",
                        )
                        anim_save_btn = gr.Button("Save Video")
                        anim_save_status = gr.Textbox(
                            label="", interactive=False, show_label=False,
                        )

                anim_duration.change(
                    fn=anim_estimate_vram,
                    inputs=[anim_duration, anim_fps],
                    outputs=[anim_vram_estimate],
                )
                anim_fps.change(
                    fn=anim_estimate_vram,
                    inputs=[anim_duration, anim_fps],
                    outputs=[anim_vram_estimate],
                )

                anim_lora_1.focus(
                    fn=lambda current: gr.update(choices=anim_list_loras(), value=current),
                    inputs=[anim_lora_1],
                    outputs=[anim_lora_1],
                )
                anim_lora_2.focus(
                    fn=lambda current: gr.update(choices=anim_list_loras(), value=current),
                    inputs=[anim_lora_2],
                    outputs=[anim_lora_2],
                )

                anim_btn.click(
                    fn=anim_generate,
                    inputs=[
                        anim_source, anim_positive, anim_negative, anim_description,
                        anim_duration, anim_fps, anim_steps, anim_guidance, anim_conditioning,
                        anim_seed, anim_sampler,
                        anim_lora_1, anim_lora_weight_1, anim_lora_2, anim_lora_weight_2,
                    ],
                    outputs=[anim_output, anim_seed_display],
                )
                anim_stop_btn.click(fn=anim_stop, outputs=[anim_seed_display])
                anim_save_btn.click(fn=anim_save, outputs=[anim_save_status])

            # === Train tab ===
            with gr.Tab("Train LoRA"):
                gr.Markdown("### Fine-tune a LoRA on your own images")
                gr.Markdown(
                    "Place images in a folder. Optionally add a `.txt` caption file "
                    "next to each image (e.g. `photo.png` + `photo.txt`)."
                )

                training_dir = gr.Textbox(
                    label="Training Images Directory",
                    placeholder=r"C:\path\to\training\images",
                )
                lora_name = gr.Textbox(
                    label="LoRA Name",
                    placeholder="my-style-lora",
                )
                with gr.Row():
                    train_steps = gr.Slider(
                        100, 5000, value=config.TRAINING_STEPS,
                        step=100, label="Training Steps",
                    )
                    train_lr = gr.Number(
                        value=config.LEARNING_RATE,
                        label="Learning Rate",
                    )
                    train_rank = gr.Slider(
                        1, 64, value=config.LORA_RANK,
                        step=1, label="LoRA Rank",
                    )
                train_btn = gr.Button("Start Training", variant="primary")
                train_log = gr.Textbox(
                    label="Training Log", lines=10, interactive=False,
                )

                train_btn.click(
                    fn=start_training,
                    inputs=[training_dir, lora_name, train_steps, train_lr, train_rank],
                    outputs=[train_log],
                )

            # === Preview Files tab ===
            with gr.Tab("Preview Files") as preview_tab:
                with gr.Row():
                    preview_refresh_btn = gr.Button("Refresh", variant="secondary", scale=1)
                    preview_filter = gr.Dropdown(
                        choices=["All", "Images", "Videos"],
                        value="All", label="Filter", scale=1,
                    )
                    preview_sort = gr.Dropdown(
                        choices=["Newest First", "Oldest First", "Name A-Z"],
                        value="Newest First", label="Sort", scale=1,
                    )
                    preview_select_mode = gr.Checkbox(
                        label="Select for Delete",
                        value=False,
                        scale=1,
                        info="Enable to click-select files for bulk delete",
                    )
                    preview_delete_btn = gr.Button(
                        "Delete Selected (0)", variant="stop", scale=1,
                    )

                preview_gallery = gr.Gallery(
                    label="Output Files",
                    columns=4,
                    height=480,
                    object_fit="cover",
                    allow_preview=False,
                    buttons=["download"],
                )

                # Checkbox list for selecting files to delete (hidden until select mode)
                preview_checklist = gr.CheckboxGroup(
                    choices=[], value=[], label="Select files to delete",
                    visible=False,
                )

                with gr.Row():
                    preview_image = gr.Image(
                        label="Preview", visible=True, interactive=False,
                    )
                    preview_video = gr.Video(
                        label="Preview", visible=False,
                    )

                preview_file_info = gr.Textbox(
                    label="File Info", interactive=False,
                )
                preview_status = gr.Textbox(
                    label="", interactive=False, show_label=False,
                )

                # -- State for tracking file paths --
                preview_file_paths = gr.State([])   # parallel list of paths

                # -- Backend wrappers --
                def _preview_refresh(filter_type, sort_order):
                    gallery, paths, status = list_output_files(
                        config.OUTPUT_DIR, filter_type, sort_order
                    )
                    # Build filename choices for the checklist
                    names = [Path(p).name for p in paths]
                    return (
                        gallery, paths, status,
                        gr.update(value="Delete Selected (0)"),
                        gr.update(choices=names, value=[]),
                    )

                def _preview_select(evt: gr.SelectData, file_paths):
                    if not file_paths or evt.index >= len(file_paths):
                        return (
                            gr.update(visible=True, value=None),
                            gr.update(visible=False, value=None),
                            "",
                        )

                    file_path = file_paths[evt.index]
                    info = get_file_info(file_path)
                    is_video = file_path.lower().endswith(".mp4")
                    if is_video:
                        return (
                            gr.update(visible=False, value=None),
                            gr.update(visible=True, value=file_path),
                            info,
                        )
                    else:
                        return (
                            gr.update(visible=True, value=file_path),
                            gr.update(visible=False, value=None),
                            info,
                        )

                def _preview_checklist_changed(checked):
                    count = len(checked)
                    return gr.update(value=f"Delete Selected ({count})")

                def _preview_delete(checked, file_paths, filter_type, sort_order):
                    if not checked:
                        return (
                            gr.update(), gr.update(), gr.update(),
                            gr.update(value="Nothing selected"),
                            gr.update(value="Delete Selected (0)"),
                            gr.update(),
                            gr.update(),
                        )
                    # Map checked filenames back to full paths
                    name_to_path = {Path(p).name: p for p in file_paths}
                    paths_to_delete = [name_to_path[n] for n in checked if n in name_to_path]

                    thumbs_dir = config.OUTPUT_DIR / ".thumbs"
                    deleted, failed = delete_files(paths_to_delete, thumbs_dir)
                    gallery, paths, status = list_output_files(
                        config.OUTPUT_DIR, filter_type, sort_order
                    )
                    names = [Path(p).name for p in paths]
                    msg = f"Deleted {deleted} file(s)"
                    if failed:
                        msg += f" ({failed} failed)"
                    gr.Info(msg)
                    return (
                        gallery, paths, status,
                        gr.update(value=msg),
                        gr.update(value="Delete Selected (0)"),
                        gr.update(value=False),  # turn off select mode
                        gr.update(choices=names, value=[]),
                    )

                def _preview_select_mode_changed(enabled, file_paths):
                    if enabled:
                        names = [Path(p).name for p in file_paths]
                        return (
                            gr.update(visible=True, choices=names, value=[]),
                            gr.update(value="Delete Selected (0)"),
                        )
                    return (
                        gr.update(visible=False, value=[]),
                        gr.update(value="Delete Selected (0)"),
                    )

                # -- Event wiring --
                _refresh_outputs = [
                    preview_gallery, preview_file_paths, preview_status,
                    preview_delete_btn, preview_checklist,
                ]
                preview_tab.select(
                    fn=_preview_refresh,
                    inputs=[preview_filter, preview_sort],
                    outputs=_refresh_outputs,
                )
                preview_refresh_btn.click(
                    fn=_preview_refresh,
                    inputs=[preview_filter, preview_sort],
                    outputs=_refresh_outputs,
                )
                preview_filter.change(
                    fn=_preview_refresh,
                    inputs=[preview_filter, preview_sort],
                    outputs=_refresh_outputs,
                )
                preview_sort.change(
                    fn=_preview_refresh,
                    inputs=[preview_filter, preview_sort],
                    outputs=_refresh_outputs,
                )
                preview_select_mode.change(
                    fn=_preview_select_mode_changed,
                    inputs=[preview_select_mode, preview_file_paths],
                    outputs=[preview_checklist, preview_delete_btn],
                )
                preview_checklist.change(
                    fn=_preview_checklist_changed,
                    inputs=[preview_checklist],
                    outputs=[preview_delete_btn],
                )

                preview_gallery.select(
                    fn=_preview_select,
                    inputs=[preview_file_paths],
                    outputs=[
                        preview_image, preview_video,
                        preview_file_info,
                    ],
                )

                preview_delete_btn.click(
                    fn=_preview_delete,
                    inputs=[preview_checklist, preview_file_paths, preview_filter, preview_sort],
                    outputs=[
                        preview_gallery, preview_file_paths, preview_status,
                        preview_file_info, preview_delete_btn,
                        preview_select_mode, preview_checklist,
                    ],
                )

        # ── Profile panel wiring ──
        all_prompt_outputs = [
            positive_prompt, negative_prompt,
            i2i_positive, i2i_negative,
            vid_positive, vid_negative,
            anim_positive, anim_negative,
        ]

        def _show_panel_for_save(pos, neg):
            return gr.update(visible=True), pos, neg

        # Save icons — show panel and populate hidden state with current tab's prompts
        _save_pos_state = gr.State("")
        _save_neg_state = gr.State("")

        for btn in [t2i_profile_save, i2i_profile_save, vid_profile_save, anim_profile_save]:
            pos_input = {
                t2i_profile_save: positive_prompt,
                i2i_profile_save: i2i_positive,
                vid_profile_save: vid_positive,
                anim_profile_save: anim_positive,
            }[btn]
            neg_input = {
                t2i_profile_save: negative_prompt,
                i2i_profile_save: i2i_negative,
                vid_profile_save: vid_negative,
                anim_profile_save: anim_negative,
            }[btn]
            btn.click(
                fn=_show_panel_for_save,
                inputs=[pos_input, neg_input],
                outputs=[profile_panel, _save_pos_state, _save_neg_state],
            )

        # Load icons — show panel
        for btn in [t2i_profile_load, i2i_profile_load, vid_profile_load, anim_profile_load]:
            btn.click(
                fn=lambda: (gr.update(visible=True), gr.update(choices=list_profiles())),
                outputs=[profile_panel, profile_dropdown],
            )

        # Save action
        profile_save_action.click(
            fn=save_profile,
            inputs=[profile_name_input, _save_pos_state, _save_neg_state],
            outputs=[profile_dropdown],
        )

        # Load action
        profile_load_action.click(
            fn=load_profile,
            inputs=[profile_dropdown],
            outputs=all_prompt_outputs,
        )

        # Delete action
        profile_delete_action.click(
            fn=delete_profile,
            inputs=[profile_dropdown],
            outputs=[profile_dropdown],
        )

        # Refresh dropdown when focused
        profile_dropdown.focus(
            fn=lambda: gr.update(choices=list_profiles()),
            outputs=[profile_dropdown],
        )

        # Close button
        profile_close_btn.click(
            fn=lambda: gr.update(visible=False),
            outputs=[profile_panel],
        )

    return app


if __name__ == "__main__":
    print(f"Device: {config.DEVICE} | Dtype: {config.DTYPE}")
    print("Loading model...")
    generator.load_model(progress_callback=print)
    print("Model loaded. Starting UI...")
    print("\n\033[1;36m-> http://127.0.0.1:7860\033[0m\n")

    app = build_ui()
    from gradio.themes.utils.fonts import Font
    theme = gr.themes.Base(
        font=[Font("Inter"), Font("Segoe UI"), Font("system-ui"), Font("sans-serif")],
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        # ── Background ──
        body_background_fill="#0f1117",
        body_background_fill_dark="#0f1117",
        background_fill_primary="#161b22",
        background_fill_primary_dark="#161b22",
        background_fill_secondary="#1c2333",
        background_fill_secondary_dark="#1c2333",
        # ── Blocks / panels ──
        block_background_fill="#161b22",
        block_background_fill_dark="#161b22",
        block_border_color="rgba(255,255,255,0.06)",
        block_border_color_dark="rgba(255,255,255,0.06)",
        block_label_background_fill="#1c2333",
        block_label_background_fill_dark="#1c2333",
        block_shadow="0 2px 12px rgba(0,0,0,0.25)",
        block_shadow_dark="0 2px 12px rgba(0,0,0,0.25)",
        # ── Text ──
        body_text_color="#e2e8f0",
        body_text_color_dark="#e2e8f0",
        body_text_color_subdued="#94a3b8",
        body_text_color_subdued_dark="#94a3b8",
        block_label_text_color="#94a3b8",
        block_label_text_color_dark="#94a3b8",
        block_title_text_color="#e2e8f0",
        block_title_text_color_dark="#e2e8f0",
        # ── Inputs ──
        input_background_fill="#1c2333",
        input_background_fill_dark="#1c2333",
        input_border_color="rgba(255,255,255,0.08)",
        input_border_color_dark="rgba(255,255,255,0.08)",
        input_placeholder_color="#64748b",
        input_placeholder_color_dark="#64748b",
        # ── Buttons ──
        button_primary_background_fill="linear-gradient(135deg, #3b82f6, #6366f1)",
        button_primary_background_fill_dark="linear-gradient(135deg, #3b82f6, #6366f1)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="transparent",
        button_secondary_background_fill_dark="transparent",
        button_secondary_text_color="#e2e8f0",
        button_secondary_text_color_dark="#e2e8f0",
        button_secondary_border_color="rgba(255,255,255,0.1)",
        button_secondary_border_color_dark="rgba(255,255,255,0.1)",
        # ── Borders ──
        border_color_primary="rgba(255,255,255,0.08)",
        border_color_primary_dark="rgba(255,255,255,0.08)",
        # ── Shadows ──
        shadow_spread="8px",
    )
    app.launch(server_name="127.0.0.1", theme=theme, css=CUSTOM_CSS, inbrowser=True)
