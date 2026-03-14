from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from peft import LoraConfig, get_peft_model
from torchvision import transforms

import config


class TrainingImageDataset(Dataset):
    """Dataset that loads images with optional .txt caption files."""

    def __init__(self, image_dir: str, resolution: int = 1024):
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.samples = []

        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        for f in self.image_dir.iterdir():
            if f.suffix.lower() in image_extensions:
                caption_file = f.with_suffix(".txt")
                if caption_file.exists():
                    caption = caption_file.read_text(encoding="utf-8").strip()
                else:
                    caption = f.stem.replace("_", " ").replace("-", " ")
                self.samples.append((f, caption))

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, caption


class LoRATrainer:
    def __init__(self, pipeline):
        """Initialize with a loaded ImageGenerator pipeline."""
        self.pipe = pipeline.pipe
        self.device = config.DEVICE
        self.dtype = config.DTYPE

    def train(
        self,
        image_dir: str,
        output_name: str,
        steps: int = config.TRAINING_STEPS,
        learning_rate: float = config.LEARNING_RATE,
        rank: int = config.LORA_RANK,
        progress_callback=None,
    ) -> str:
        """Train a LoRA on images in the given directory.

        Returns the path to the saved LoRA file.
        """
        if self.pipe is None:
            raise ValueError("No model loaded. Please load a model before training.")
        if not hasattr(self.pipe, 'text_encoder_2'):
            raise ValueError("LoRA training requires an SDXL model. SD 1.5 is not supported.")

        # Prepare dataset
        dataset = TrainingImageDataset(image_dir)
        if len(dataset) == 0:
            raise ValueError(f"No images found in {image_dir}")

        dataloader = DataLoader(
            dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            shuffle=True,
        )

        if progress_callback:
            progress_callback(f"Found {len(dataset)} images. Preparing LoRA training...")

        # Configure LoRA on the U-Net
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.0,
        )

        unet = self.pipe.unet
        unet.requires_grad_(False)
        unet = get_peft_model(unet, lora_config)
        unet.train()

        # Enable gradient checkpointing to save memory
        unet.enable_gradient_checkpointing()

        # Freeze VAE and text encoders
        vae = self.pipe.vae
        text_encoder = self.pipe.text_encoder
        text_encoder_2 = self.pipe.text_encoder_2
        tokenizer = self.pipe.tokenizer
        tokenizer_2 = self.pipe.tokenizer_2

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)

        noise_scheduler = self.pipe.scheduler

        # Optimizer — only LoRA params
        trainable_params = [p for p in unet.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        # Training loop
        step = 0
        log_lines = []
        while step < steps:
            for images, captions in dataloader:
                if step >= steps:
                    break

                images = images.to(self.device, dtype=self.dtype)

                # Encode images to latent space (cast to VAE dtype in case of mismatch)
                with torch.no_grad():
                    latents = vae.encode(images.to(dtype=vae.dtype)).latent_dist.sample()
                    latents = (latents * vae.config.scaling_factor).to(self.dtype)

                # Add noise at random timestep
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=self.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode captions with both text encoders (SDXL dual encoder)
                with torch.no_grad():
                    tokens_1 = tokenizer(
                        captions, padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt",
                    ).input_ids.to(self.device)
                    encoder_hidden_states = text_encoder(tokens_1, output_hidden_states=True)
                    encoder_hidden_states_1 = encoder_hidden_states.hidden_states[-2]

                    tokens_2 = tokenizer_2(
                        captions, padding="max_length",
                        max_length=tokenizer_2.model_max_length,
                        truncation=True, return_tensors="pt",
                    ).input_ids.to(self.device)
                    encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True)
                    encoder_hidden_states_2 = encoder_output_2.hidden_states[-2]
                    pooled_output = encoder_output_2[0]

                    encoder_hidden_states = torch.cat(
                        [encoder_hidden_states_1, encoder_hidden_states_2], dim=-1
                    )

                # SDXL requires add_time_ids
                add_time_ids = torch.tensor(
                    [[config.DEFAULT_HEIGHT, config.DEFAULT_WIDTH,
                      0, 0,
                      config.DEFAULT_HEIGHT, config.DEFAULT_WIDTH]],
                    dtype=self.dtype, device=self.device,
                ).repeat(latents.shape[0], 1)

                added_cond_kwargs = {
                    "text_embeds": pooled_output.to(self.dtype),
                    "time_ids": add_time_ids,
                }

                # Predict noise
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.to(self.dtype),
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # Loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                if step % 10 == 0 or step == steps:
                    msg = f"Step {step}/{steps} — loss: {loss.item():.4f}"
                    log_lines.append(msg)
                    if progress_callback:
                        progress_callback("\n".join(log_lines))

        # Save LoRA weights as a single .safetensors file
        config.LORA_DIR.mkdir(parents=True, exist_ok=True)
        save_name = output_name if output_name.endswith(".safetensors") else f"{output_name}.safetensors"
        save_path = config.LORA_DIR / save_name
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file
        lora_state = get_peft_model_state_dict(unet)
        save_file(lora_state, str(save_path))

        # Restore U-Net to non-PEFT state for continued inference
        self.pipe.unet = unet.merge_and_unload()
        self.pipe.unet.requires_grad_(False)
        self.pipe.unet.eval()

        final_msg = f"Training complete. LoRA saved to {save_path}"
        log_lines.append(final_msg)
        if progress_callback:
            progress_callback("\n".join(log_lines))

        return str(save_path)
