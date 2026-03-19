# ============================================================
# video_chunker.py
# VRAM-safe video generation: single-pass diffusion + chunked VAE decode
# ============================================================

import gc
import torch


def _flush_vram():
    """Aggressively reclaim GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _decode_latents_chunked(video_generator, latents, vae_batch_frames=8,
                            progress_callback=None):
    """Decode latents through the VAE in small temporal batches.

    Instead of decoding all frames at once (which OOMs on long videos),
    we split the latent tensor along the temporal dimension and decode
    each slice separately, converting to PIL frames immediately and
    freeing GPU memory between batches.

    Supports both WAN (3D VAE with temporal compression) and AnimateDiff
    (SD 1.5 VAE, 2D per-frame decode).

    Args:
        video_generator: VideoGenerator or AnimateDiffGenerator instance
        latents: Raw latent tensor from the diffusion pass
        vae_batch_frames: Number of video frames per VAE decode batch.
            For WAN: these are latent temporal frames (each = ~4 video frames).
            For AnimateDiff: these are actual video frames (1:1).
            Lower = less VRAM, slower. Default 8 is safe for 24GB.
        progress_callback: Optional callable for status updates

    Returns:
        List of PIL frames
    """
    from PIL import Image
    import numpy as np

    is_animatediff = _is_animatediff_generator(video_generator)
    vae = video_generator.pipe.vae

    if is_animatediff:
        return _decode_animatediff_chunked(
            video_generator, latents, vae, vae_batch_frames, progress_callback
        )
    else:
        return _decode_wan_chunked(
            video_generator, latents, vae, vae_batch_frames, progress_callback
        )


def _decode_wan_chunked(video_generator, latents, vae, vae_batch_frames,
                        progress_callback):
    """Chunked VAE decode for WAN 2.1 models.

    WAN's VAE is a 3D convolution-based decoder. Latent shape is
    [batch, channels, temporal_frames, height, width] where temporal
    is compressed 4x (so 8 latent frames = ~32 video frames).
    """
    # Denormalize latents using WAN VAE config
    latents = latents.to(vae.dtype)

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0 / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean

    # Latent shape: [batch, channels, temporal_frames, height, width]
    num_latent_frames = latents.shape[2]
    all_frames = []
    num_batches = (num_latent_frames + vae_batch_frames - 1) // vae_batch_frames

    for batch_idx in range(num_batches):
        start = batch_idx * vae_batch_frames
        end = min(start + vae_batch_frames, num_latent_frames)

        if progress_callback:
            progress_callback(
                f"Decoding VAE batch {batch_idx + 1}/{num_batches} "
                f"(latent frames {start}-{end - 1})"
            )

        latent_slice = latents[:, :, start:end, :, :]

        with torch.inference_mode():
            video_slice = vae.decode(latent_slice, return_dict=False)[0]

        # Convert to PIL immediately
        frames_batch = video_generator.pipe.video_processor.postprocess_video(
            video_slice, output_type="pil"
        )
        if frames_batch and isinstance(frames_batch[0], list):
            frames_batch = frames_batch[0]

        all_frames.extend(frames_batch)

        del latent_slice, video_slice, frames_batch
        _flush_vram()

    return all_frames


def _decode_animatediff_chunked(video_generator, latents, vae, vae_batch_frames,
                                progress_callback):
    """Chunked VAE decode for AnimateDiff (SD 1.5 based).

    AnimateDiff latent shape is [batch, channels, num_frames, height, width].
    The VAE is a standard 2D SD 1.5 decoder — frames are reshaped to
    [batch*num_frames, channels, height, width] and decoded individually.
    We chunk along the frame dimension to limit VRAM.
    """
    from PIL import Image
    import numpy as np

    # Denormalize using SD 1.5 scaling factor
    latents = 1 / vae.config.scaling_factor * latents

    # Shape: [batch, channels, num_frames, height, width]
    batch_size, channels, num_frames, height, width = latents.shape
    all_frames = []
    num_batches = (num_frames + vae_batch_frames - 1) // vae_batch_frames

    for batch_idx in range(num_batches):
        start = batch_idx * vae_batch_frames
        end = min(start + vae_batch_frames, num_frames)
        batch_count = end - start

        if progress_callback:
            progress_callback(
                f"Decoding VAE batch {batch_idx + 1}/{num_batches} "
                f"(frames {start}-{end - 1})"
            )

        # Slice temporal dimension, then reshape to [batch_count, C, H, W]
        latent_slice = latents[:, :, start:end, :, :]
        latent_2d = latent_slice.permute(0, 2, 1, 3, 4).reshape(
            batch_size * batch_count, channels, height, width
        )

        with torch.inference_mode():
            decoded = vae.decode(latent_2d, return_dict=False)[0]

        # Batch convert: [-1, 1] -> [0, 255] uint8 on GPU, single CPU transfer
        decoded = ((decoded + 1) * 0.5).clamp_(0, 1)
        # [batch, 3, H, W] -> [batch, H, W, 3], single .cpu() call
        frames_np = (decoded.permute(0, 2, 3, 1).mul_(255)
                     .to(torch.uint8).cpu().numpy())
        for i in range(frames_np.shape[0]):
            all_frames.append(Image.fromarray(frames_np[i]))

        del latent_slice, latent_2d, decoded, frames_np

    # Single flush after all batches instead of per-batch (avoids CUDA stalls)
    _flush_vram()
    return all_frames


def _is_animatediff_generator(video_generator):
    """Check if this is an AnimateDiffGenerator by looking for source_image param."""
    import inspect
    sig = inspect.signature(video_generator.generate_latents)
    return "source_image" in sig.parameters


def generate_video_chunked(
    video_generator,
    positive_prompt,
    negative_prompt,
    num_frames_total,
    num_inference_steps,
    guidance_scale,
    seed,
    scheduler_name,
    progress_callback=None,
    # AnimateDiff-specific parameters
    source_image=None,
    controlnet_conditioning_scale=1.0,
    # VAE decode batch size (temporal frames per batch)
    vae_batch_frames=8,
):
    """
    VRAM-safe video generation using single-pass diffusion + chunked VAE decode.

    The diffusion pass runs all frames in a single call to preserve temporal
    coherence (no stitching artifacts, no subject drift). The VAE decode is
    then performed in small batches to keep peak VRAM within budget.

    Aggressive memory reclamation happens at every stage:
    - After diffusion: text encoder and transformer caches freed
    - During VAE decode: each batch is freed after converting to PIL
    - After stitching: latent tensor freed

    Target: 5s @ 30fps (149 frames) on 24GB VRAM with 2 LoRAs loaded.
    """

    is_animatediff = _is_animatediff_generator(video_generator)

    # ================================================================
    # Stage 1: Single-pass diffusion — full temporal coherence
    # ================================================================
    if progress_callback:
        progress_callback(f"Running diffusion ({num_frames_total} frames)...")

    with torch.inference_mode():
        if is_animatediff:
            latents = video_generator.generate_latents(
                source_image=source_image,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames_total,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                seed=seed,
                scheduler_name=scheduler_name,
            )
        else:
            latents = video_generator.generate_latents(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames_total,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                scheduler_name=scheduler_name,
            )

    if video_generator.was_interrupted or latents is None:
        if progress_callback:
            progress_callback("Generation interrupted.")
        _flush_vram()
        return None

    # ================================================================
    # Stage 2: Aggressive post-diffusion cleanup
    # Free everything except the latent tensor and the VAE.
    # ================================================================
    if progress_callback:
        progress_callback("Freeing diffusion memory before VAE decode...")

    _flush_vram()

    # ================================================================
    # Stage 3: Chunked VAE decode — small batches to stay in VRAM budget
    # ================================================================
    if progress_callback:
        progress_callback("Starting chunked VAE decode...")

    frames = _decode_latents_chunked(
        video_generator, latents,
        vae_batch_frames=vae_batch_frames,
        progress_callback=progress_callback,
    )

    # ================================================================
    # Stage 4: Final cleanup — free latents and VAE caches
    # ================================================================
    del latents
    _flush_vram()

    if progress_callback:
        progress_callback(f"Done — {len(frames)} frames decoded.")

    return frames
