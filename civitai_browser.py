# ============================================================
# civitai_browser.py
# CivitAI API client for searching and downloading models/LoRAs
# This is the ONLY module in ImaGen that makes network requests.
# ============================================================

from pathlib import Path

import requests

import config

CIVITAI_API = "https://civitai.com/api/v1"

# API key storage — outside project folder to avoid accidental commits
_KEY_DIR = Path.home() / ".imagen"
_KEY_FILE = _KEY_DIR / "civitai_key.txt"
_ENABLED_FILE = _KEY_DIR / "civitai_enabled.txt"


def get_api_key():
    """Load saved CivitAI API key, or return empty string."""
    if _KEY_FILE.exists():
        return _KEY_FILE.read_text(encoding="utf-8").strip()
    return ""


def save_api_key(key):
    """Persist CivitAI API key to ~/.imagen/civitai_key.txt."""
    _KEY_DIR.mkdir(parents=True, exist_ok=True)
    _KEY_FILE.write_text(key.strip(), encoding="utf-8")


def is_enabled():
    """Check if CivitAI integration is enabled. Enabled by default."""
    if _ENABLED_FILE.exists():
        return _ENABLED_FILE.read_text(encoding="utf-8").strip().lower() == "true"
    return True


def set_enabled(enabled):
    """Persist CivitAI enabled/disabled toggle."""
    _KEY_DIR.mkdir(parents=True, exist_ok=True)
    _ENABLED_FILE.write_text("true" if enabled else "false", encoding="utf-8")


# Base models that ImaGen actually supports.
# These correspond to CivitAI's baseModel enum values.
SUPPORTED_BASE_MODELS = [
    "All",
    "Wan",
    "WanVideo",
    "SDXL 1.0",
    "SD 1.5",
    "Pony",
    "Illustrious",
    "Flux.1 D",
]

# Content rating filter options
CONTENT_FILTERS = ["SFW Only", "NSFW Only", "Show All"]


def search_models(query="", model_type="All", sort="Most Downloaded", limit=20,
                  base_model="All", content_filter="Show All", cursor=None):
    """Search CivitAI models using cursor-based pagination.

    Args:
        query: Search text (model name, keyword, etc.)
        model_type: "All", "Checkpoint", "LORA", or "TextualInversion"
        sort: "Most Downloaded", "Highest Rated", or "Newest"
        limit: Results per page (max 100)
        base_model: "All" or a specific base model name (e.g. "SDXL 1.0", "Wan")
        content_filter: "SFW Only", "NSFW Only", or "Show All"
        cursor: Pagination cursor from previous response (None for first page)

    Returns:
        (results_list, next_cursor) tuple. next_cursor is None if no more pages.
    """
    params = {"limit": limit, "sort": sort}
    if query:
        params["query"] = query
    if cursor:
        params["cursor"] = cursor
    if model_type and model_type != "All":
        params["types"] = model_type
    if base_model and base_model != "All":
        params["baseModels"] = base_model
    if content_filter == "SFW Only":
        params["nsfw"] = "false"
    elif content_filter == "NSFW Only":
        params["nsfw"] = "true"

    resp = requests.get(f"{CIVITAI_API}/models", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("items", []):
        versions = item.get("modelVersions", [])
        first_version = versions[0] if versions else {}
        images = first_version.get("images", [])
        raw_preview = images[0]["url"] if images else None
        # Request a width-limited static image to avoid animated/video previews.
        # CivitAI CDN URLs end with /original — replace with /width=450 for a
        # static JPEG rendition that works reliably in <img> tags.
        if raw_preview and "/original" in raw_preview:
            preview_url = raw_preview.replace("/original", "/width=450")
        elif raw_preview:
            preview_url = raw_preview
        else:
            preview_url = None
        files = first_version.get("files", [])
        primary_file = files[0] if files else {}

        size_kb = primary_file.get("sizeKB", 0)
        if size_kb > 1024 * 1024:
            size_str = f"{size_kb / (1024 * 1024):.1f} GB"
        elif size_kb > 1024:
            size_str = f"{size_kb / 1024:.0f} MB"
        else:
            size_str = f"{size_kb:.0f} KB"

        # Extract trigger words and recommended settings from version metadata
        trained_words = first_version.get("trainedWords", [])
        # CivitAI stores recommended settings at version level
        rec_settings = {}
        if first_version.get("steps"):
            rec_settings["steps"] = first_version["steps"]
        if first_version.get("cfgScale"):
            rec_settings["cfg"] = first_version["cfgScale"]
        if first_version.get("clipSkip"):
            rec_settings["clip_skip"] = first_version["clipSkip"]
        # Sampler can be in the version images' meta or at version level
        if images and images[0].get("meta"):
            meta = images[0]["meta"]
            if meta.get("sampler") and "sampler" not in rec_settings:
                rec_settings["sampler"] = meta["sampler"]
            if meta.get("cfgScale") and "cfg" not in rec_settings:
                rec_settings["cfg"] = meta["cfgScale"]
            if meta.get("steps") and "steps" not in rec_settings:
                rec_settings["steps"] = meta["steps"]
            if meta.get("clipSkip") and "clip_skip" not in rec_settings:
                rec_settings["clip_skip"] = meta["clipSkip"]

        results.append({
            "id": item["id"],
            "name": item["name"],
            "type": item.get("type", "Unknown"),
            "base_model": first_version.get("baseModel", ""),
            "preview_url": preview_url,
            "version_id": first_version.get("id"),
            "version_name": first_version.get("name", ""),
            "file_size_str": size_str,
            "file_size_kb": size_kb,
            "filename": primary_file.get("name", ""),
            "download_url": primary_file.get("downloadUrl", ""),
            "description": item.get("description", ""),
            "trained_words": trained_words,
            "recommended_settings": rec_settings,
            "civitai_url": f"https://civitai.com/models/{item['id']}",
        })

    metadata = data.get("metadata", {})
    next_cursor = metadata.get("nextCursor")
    return results, next_cursor


def save_lora_metadata(dest_dir, filename, metadata):
    """Save a JSON sidecar alongside a downloaded LoRA file.

    Args:
        dest_dir: Directory where the LoRA was saved
        filename: The LoRA filename (e.g. "my_lora.safetensors")
        metadata: Dict with keys like trained_words, recommended_settings, civitai_url
    """
    import json
    sidecar_name = Path(filename).stem + ".json"
    sidecar_path = Path(dest_dir) / sidecar_name
    sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def get_download_dir(model_type, base_model=""):
    """Return the correct destination directory for a model type and base model.

    Routes downloads to the appropriate architecture subdirectory based on the
    CivitAI base model identifier (e.g. "Pony", "Illustrious", "Flux.1 D").
    """
    # Map CivitAI base model names to architecture directory keys
    _BASE_MODEL_TO_ARCH = {
        "Pony": "Pony",
        "Illustrious": "Illustrious",
        "Flux.1 D": "Flux",
        "Flux.1 S": "Flux",
    }
    arch = _BASE_MODEL_TO_ARCH.get(base_model)

    if model_type == "LORA":
        if arch:
            return config.ARCH_LORA_DIRS[arch]
        # Default to SDXL/SD1.5 lora dir for unknown architectures
        return config.ARCH_LORA_DIRS["SDXL / SD 1.5"]
    # Checkpoints and everything else
    if arch:
        return config.ARCH_MODEL_DIRS[arch]
    # Default to SDXL/SD1.5 model dir for unknown architectures
    return config.ARCH_MODEL_DIRS["SDXL / SD 1.5"]


def download_model(download_url, dest_dir, filename, api_key=None, progress_callback=None):
    """Stream-download a model file to the destination directory.

    Args:
        download_url: CivitAI download URL
        dest_dir: Target directory (models/ or loras/)
        filename: Filename to save as
        api_key: Optional CivitAI API key for restricted downloads
        progress_callback: Optional callable(status_string)

    Returns:
        Path to downloaded file on success, or raises on failure.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if progress_callback:
        progress_callback(f"Connecting to CivitAI...")

    resp = requests.get(download_url, headers=headers, stream=True, timeout=30)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):  # 8MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback and total > 0:
                pct = downloaded / total * 100
                size_gb = downloaded / (1024 ** 3)
                total_gb = total / (1024 ** 3)
                progress_callback(
                    f"Downloading... {pct:.0f}% ({size_gb:.2f} / {total_gb:.2f} GB)"
                )

    if progress_callback:
        progress_callback(f"Downloaded: {filename}")

    return str(dest_path)
