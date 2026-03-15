import gc

import torch
import numpy as np
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader

import config


class Upscaler:
    def __init__(self):
        self._model = None
        self._model_name = None

    def get_available_upscalers(self):
        """List upscaler .pth files in the upscalers/ directory."""
        config.UPSCALER_DIR.mkdir(parents=True, exist_ok=True)
        upscalers = []
        for f in config.UPSCALER_DIR.iterdir():
            if f.suffix in (".pth", ".pt", ".safetensors"):
                upscalers.append(f.stem)
        return sorted(upscalers)

    def load(self, name: str):
        """Load an upscaler model by name from upscalers/ directory."""
        if self._model_name == name:
            return  # Already loaded

        self.unload()

        # Find the file (could be .pth or .safetensors)
        path = None
        for ext in (".pth", ".pt", ".safetensors"):
            candidate = config.UPSCALER_DIR / f"{name}{ext}"
            if candidate.exists():
                path = candidate
                break

        if path is None:
            raise FileNotFoundError(f"Upscaler not found: {name}")

        model = ModelLoader().load_from_file(str(path))
        assert isinstance(model, ImageModelDescriptor)
        # Load on CPU — moved to GPU per-tile in upscale() only if VRAM is available.
        self._model = model.eval()
        self._model_name = name

    def unload(self):
        """Free VRAM."""
        self._model = None
        self._model_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _pick_device(self):
        """Use GPU if enough free VRAM, otherwise CPU."""
        if not torch.cuda.is_available():
            return torch.device("cpu")
        free = torch.cuda.mem_get_info()[0]
        # Need at least 2GB free to run upscaler tiles on GPU
        if free > 2 * 1024**3:
            return torch.device("cuda")
        return torch.device("cpu")

    def upscale(self, image: Image.Image, tile_size: int = 512, overlap: int = 32) -> Image.Image:
        """Upscale a PIL Image using tiling to avoid OOM. Returns upscaled PIL Image."""
        if self._model is None:
            return image

        device = self._pick_device()
        model = self._model.to(device)

        img_array = np.array(image).astype(np.float32) / 255.0
        h, w, c = img_array.shape

        # Detect upscale factor from model
        scale = model.scale

        with torch.inference_mode():
            # If the image is small enough, process in one shot
            if h <= tile_size and w <= tile_size:
                tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
                tensor = tensor.to(device=device, dtype=model.dtype)
                output = model(tensor)
                output = output.squeeze(0).clamp(0, 1).permute(1, 2, 0)
                output = (output.cpu().numpy() * 255).astype(np.uint8)
                model.to("cpu")
                return Image.fromarray(output)

            # Tiled upscale for large images
            out_h, out_w = h * scale, w * scale
            result = np.zeros((out_h, out_w, c), dtype=np.float32)
            weight_map = np.zeros((out_h, out_w, c), dtype=np.float32)

            step = tile_size - overlap

            for y in range(0, h, step):
                for x in range(0, w, step):
                    # Clamp tile bounds to image edges
                    y2 = min(y + tile_size, h)
                    x2 = min(x + tile_size, w)
                    y1 = max(y2 - tile_size, 0)
                    x1 = max(x2 - tile_size, 0)

                    tile = img_array[y1:y2, x1:x2]
                    tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0)
                    tensor = tensor.to(device=device, dtype=model.dtype)
                    out_tile = model(tensor)
                    out_tile = out_tile.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

                    # Place tile in output with 1.0 blending weight
                    oy1, ox1 = y1 * scale, x1 * scale
                    oy2, ox2 = y2 * scale, x2 * scale
                    result[oy1:oy2, ox1:ox2] += out_tile
                    weight_map[oy1:oy2, ox1:ox2] += 1.0

        # Move model back to CPU to free VRAM
        model.to("cpu")

        # Average overlapping regions
        result = np.where(weight_map > 0, result / weight_map, result)
        result = (result * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(result)
