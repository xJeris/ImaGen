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
from video_pipeline import VideoGenerator, DURATION_TO_FRAMES, WAN_FPS, VIDEO_SCHEDULER_NAMES

generator = ImageGenerator()
video_generator = VideoGenerator()
upscaler = Upscaler()
trainer = None
_last_image = None
_last_video_path = None


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
        return "No model selected."
    if model_name == generator._model_name:
        return f"Already loaded: {model_name}"
    try:
        # Unload video model first to free VRAM for image generation.
        video_generator.unload_model()
        generator.load_model(model_name, progress_callback=print)
        return f"Loaded: {model_name} ({generator._model_type})"
    except Exception as e:
        return f"Failed to load {model_name}: {e}"


def _apply_upscaler(image, upscaler_name):
    """Apply upscaler to image if one is selected."""
    if upscaler_name and upscaler_name != "None":
        upscaler.load(upscaler_name)
        return upscaler.upscale(image)
    return image


def generate_image(
    positive_prompt, negative_prompt, description,
    steps, guidance, width, height, seed, sampler,
    lora_name, lora_weight, upscaler_name,
    hires_enable, hires_upscaler, hires_scale, hires_denoise, hires_steps,
):
    global _last_image

    full_prompt = positive_prompt.strip()
    if description.strip():
        full_prompt = f"{full_prompt}, {description.strip()}"

    if not full_prompt:
        raise gr.Error("Please enter a prompt.")

    if lora_name and lora_name != "None":
        lora_path = str(config.LORA_DIR / lora_name)
        generator.load_lora(lora_path, weight=lora_weight)
    else:
        generator.unload_lora()

    actual_seed = int(seed)
    if actual_seed < 0:
        actual_seed = torch.randint(0, 2**32, (1,)).item()

    image = generator.generate(
        positive_prompt=full_prompt,
        negative_prompt=negative_prompt,
        steps=int(steps),
        guidance_scale=guidance,
        width=int(width),
        height=int(height),
        seed=actual_seed,
        scheduler_name=sampler,
    )

    # Hires Fix: upscale then img2img second pass for real detail
    if hires_enable and hires_scale > 1.0:
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
            offload_encoders=True,  # text encoders → CPU during hires pass
        )

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
    source_image, positive_prompt, negative_prompt, description,
    strength, steps, guidance, seed, sampler,
    lora_name, lora_weight, upscaler_name,
):
    global _last_image

    if source_image is None:
        raise gr.Error("Please upload a source image.")

    full_prompt = positive_prompt.strip()
    if description.strip():
        full_prompt = f"{full_prompt}, {description.strip()}"

    if not full_prompt:
        raise gr.Error("Please enter a prompt.")

    if lora_name and lora_name != "None":
        lora_path = str(config.LORA_DIR / lora_name)
        generator.load_lora(lora_path, weight=lora_weight)
    else:
        generator.unload_lora()

    actual_seed = int(seed)
    if actual_seed < 0:
        actual_seed = torch.randint(0, 2**32, (1,)).item()

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
        # Unload the image model first to free VRAM for video generation.
        generator.unload_model()
        video_generator.load_model(model_name, progress_callback=print)
        return f"Loaded: {model_name}"
    except Exception as e:
        return f"Failed to load {model_name}: {e}"


def video_generate(
    positive_prompt, negative_prompt, description,
    duration, steps, guidance, seed, sampler,
    lora_name, lora_weight,
):
    global _last_video_path

    if video_generator.pipe is None:
        raise gr.Error("Please select and load a video model first.")

    full_prompt = positive_prompt.strip()
    if description.strip():
        full_prompt = f"{full_prompt}, {description.strip()}"

    if not full_prompt:
        raise gr.Error("Please enter a prompt.")

    if lora_name and lora_name != "None":
        lora_path = str(config.LORA_DIR / lora_name)
        video_generator.load_lora(lora_path, weight=lora_weight)
    else:
        video_generator.unload_lora()

    num_frames = DURATION_TO_FRAMES.get(int(duration), 49)

    actual_seed = int(seed)
    if actual_seed < 0:
        actual_seed = torch.randint(0, 2**32, (1,)).item()

    frames = video_generator.generate_video(
        positive_prompt=full_prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        seed=actual_seed,
        scheduler_name=sampler,
    )

    # Export to temp MP4 (clean up previous temp file)
    if _last_video_path:
        try:
            Path(_last_video_path).unlink(missing_ok=True)
        except OSError:
            pass
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    video_generator.export_video(frames, tmp.name, fps=WAN_FPS)
    _last_video_path = tmp.name
    return tmp.name, f"Seed: {actual_seed}"


def video_stop():
    video_generator.interrupt()


def video_save():
    global _last_video_path
    if _last_video_path is None:
        return "No video to save. Generate a video first."

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = config.OUTPUT_DIR / f"vid_{timestamp}.mp4"
    shutil.copy2(_last_video_path, str(dest))
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

/* ── Image / Video output area ── */
.image-container, .video-container {
    border-radius: 12px !important;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
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
"""


def build_ui():
    with gr.Blocks(title="ImaGen — Text to Image & Video", fill_width=True) as app:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        gr.Markdown(
            "# ImaGen\nOffline text-to-image, image-to-image & video generation",
            elem_id="imagen-header",
        )

        # === Global settings bar ===
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

        model_dropdown.change(
            fn=switch_model,
            inputs=[model_dropdown],
            outputs=[model_status],
        )

        # Refresh dropdowns on focus (rescan folders for new files)
        model_dropdown.focus(
            fn=lambda: gr.update(choices=list_models()),
            outputs=[model_dropdown],
        )
        upscaler_dropdown.focus(
            fn=lambda: gr.update(choices=list_upscalers()),
            outputs=[upscaler_dropdown],
        )

        with gr.Tabs():
            # === Text to Image tab ===
            with gr.Tab("Text to Image"):
                with gr.Row():
                    with gr.Column(scale=1):
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
                            lora_dropdown = gr.Dropdown(
                                choices=list_loras(),
                                value="None",
                                label="Select LoRA",
                            )
                            lora_weight = gr.Slider(
                                0.0, 1.5, value=1.0,
                                step=0.05, label="LoRA Weight",
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

                lora_dropdown.focus(
                    fn=lambda: gr.update(choices=list_loras()),
                    outputs=[lora_dropdown],
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
                        lora_dropdown, lora_weight, upscaler_dropdown,
                        hires_enable, hires_upscaler, hires_scale, hires_denoise, hires_steps,
                    ],
                    outputs=[output_image, seed_display],
                )
                stop_btn.click(fn=stop_generation)
                save_btn.click(fn=save_image, outputs=[save_status])

            # === Img2Img tab ===
            with gr.Tab("Image to Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        i2i_source = gr.Image(
                            label="Source Image",
                            type="pil",
                        )
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
                            i2i_lora = gr.Dropdown(
                                choices=list_loras(),
                                value="None",
                                label="Select LoRA",
                            )
                            i2i_lora_weight = gr.Slider(
                                0.0, 1.5, value=1.0,
                                step=0.05, label="LoRA Weight",
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

                i2i_lora.focus(
                    fn=lambda: gr.update(choices=list_loras()),
                    outputs=[i2i_lora],
                )

                i2i_btn.click(
                    fn=img2img_generate,
                    inputs=[
                        i2i_source, i2i_positive, i2i_negative, i2i_description,
                        i2i_strength, i2i_steps, i2i_guidance, i2i_seed, i2i_sampler,
                        i2i_lora, i2i_lora_weight, upscaler_dropdown,
                    ],
                    outputs=[i2i_output, i2i_seed_display],
                )
                i2i_stop_btn.click(fn=stop_generation)
                i2i_save_btn.click(fn=save_image, outputs=[i2i_save_status])

            # === Text to Video tab ===
            with gr.Tab("Text to Video"):
                with gr.Row():
                    vid_model = gr.Dropdown(
                        choices=video_list_models(),
                        value=None,
                        label="Video Model (WAN 2.1)",
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
                                info="1s=17 frames, 2s=33, 3s=49, 4s=65, 5s=81 at 16fps",
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
                            vid_lora = gr.Dropdown(
                                choices=video_list_loras(),
                                value="None",
                                label="Select LoRA",
                            )
                            vid_lora_weight = gr.Slider(
                                0.0, 1.5, value=1.0,
                                step=0.05, label="LoRA Weight",
                            )

                        with gr.Row():
                            vid_btn = gr.Button("Generate", variant="primary")
                            vid_stop_btn = gr.Button("Stop", variant="stop")

                    with gr.Column(scale=1):
                        vid_output = gr.Video(label="Generated Video")
                        vid_seed_display = gr.Textbox(
                            label="", interactive=False, show_label=False,
                            placeholder="Seed will appear here after generation",
                        )
                        vid_save_btn = gr.Button("Save Video")
                        vid_save_status = gr.Textbox(
                            label="", interactive=False, show_label=False,
                        )

                vid_lora.focus(
                    fn=lambda: gr.update(choices=video_list_loras()),
                    outputs=[vid_lora],
                )

                vid_btn.click(
                    fn=video_generate,
                    inputs=[
                        vid_positive, vid_negative, vid_description,
                        vid_duration, vid_steps, vid_guidance, vid_seed, vid_sampler,
                        vid_lora, vid_lora_weight,
                    ],
                    outputs=[vid_output, vid_seed_display],
                )
                vid_stop_btn.click(fn=video_stop)
                vid_save_btn.click(fn=video_save, outputs=[vid_save_status])

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
    app.launch(server_name="127.0.0.1", theme=theme, css=CUSTOM_CSS)
