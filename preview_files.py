# ============================================================
# preview_files.py
# Backend logic for the Preview Files tab
# ============================================================

from datetime import datetime
from pathlib import Path


def _human_size(size_bytes):
    """Convert bytes to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def generate_video_thumbnail(video_path, thumbs_dir):
    """Extract first frame of MP4 as a thumbnail PNG.

    Caches in thumbs_dir so repeated refreshes are fast.
    Returns Path to thumbnail, or None on failure.
    """
    video_path = Path(video_path)
    thumbs_dir = Path(thumbs_dir)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    thumb_path = thumbs_dir / f"{video_path.stem}_thumb.png"

    # Use cached thumbnail if it exists and is newer than the video
    if thumb_path.exists():
        if thumb_path.stat().st_mtime >= video_path.stat().st_mtime:
            return thumb_path

    try:
        import imageio
        reader = imageio.get_reader(str(video_path))
        frame = reader.get_data(0)
        reader.close()

        from PIL import Image
        img = Image.fromarray(frame)
        img.save(str(thumb_path), "PNG")
        return thumb_path
    except Exception:
        return None


def list_output_files(output_dir, filter_type="All", sort_order="Newest First"):
    """Scan output_dir and return gallery data + status string.

    Args:
        output_dir: Path to the outputs directory
        filter_type: "All", "Images", or "Videos"
        sort_order: "Newest First", "Oldest First", or "Name A-Z"

    Returns:
        (gallery_items, file_paths, status_string) where:
        - gallery_items: list of (image_path, caption) tuples for gr.Gallery
        - file_paths: list of absolute path strings (parallel to gallery_items)
        - status_string: summary like "42 files (128 MB total)"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = output_dir / ".thumbs"

    # Collect files
    files = []
    for f in output_dir.iterdir():
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if filter_type == "Images" and ext != ".png":
            continue
        if filter_type == "Videos" and ext != ".mp4":
            continue
        if ext in (".png", ".mp4"):
            files.append(f)

    # Sort
    if sort_order == "Newest First":
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    elif sort_order == "Oldest First":
        files.sort(key=lambda f: f.stat().st_mtime)
    elif sort_order == "Name A-Z":
        files.sort(key=lambda f: f.name.lower())

    # Build gallery items
    gallery_items = []
    file_paths = []
    total_size = 0

    for f in files:
        size = f.stat().st_size
        total_size += size
        caption = f"{f.name}\n{_human_size(size)}"

        if f.suffix.lower() == ".mp4":
            thumb = generate_video_thumbnail(f, thumbs_dir)
            if thumb:
                gallery_items.append((str(thumb), caption))
            else:
                # No thumbnail available — skip this file in gallery
                # (or we could use a placeholder, but skipping is cleaner)
                gallery_items.append((None, caption))
        else:
            gallery_items.append((str(f), caption))

        file_paths.append(str(f))

    # Filter out items with no thumbnail (failed video extraction)
    filtered_gallery = []
    filtered_paths = []
    for item, path in zip(gallery_items, file_paths):
        if item[0] is not None:
            filtered_gallery.append(item)
            filtered_paths.append(path)

    status = _build_status(len(files), total_size)
    return filtered_gallery, filtered_paths, status


def _build_status(count, total_bytes):
    """Build a status summary string."""
    if count == 0:
        return "No output files yet"
    size_str = _human_size(total_bytes)
    return f"{count} file{'s' if count != 1 else ''} ({size_str} total)"


def get_file_info(file_path):
    """Return a formatted info string for a file."""
    f = Path(file_path)
    if not f.exists():
        return "File not found"

    stat = f.stat()
    size = _human_size(stat.st_size)
    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    file_type = "Image (PNG)" if f.suffix.lower() == ".png" else "Video (MP4)"

    return f"{f.name}  |  {file_type}  |  {size}  |  Modified: {modified}"


def delete_files(file_paths, thumbs_dir=None):
    """Delete files from disk, including cached thumbnails.

    Args:
        file_paths: list of file path strings to delete
        thumbs_dir: Path to thumbnail cache dir (for cleanup)

    Returns:
        (deleted_count, failed_count) tuple
    """
    deleted = 0
    failed = 0

    for path_str in file_paths:
        f = Path(path_str)
        try:
            if f.exists():
                f.unlink()
                deleted += 1

                # Also remove cached thumbnail if it exists
                if thumbs_dir:
                    thumb = Path(thumbs_dir) / f"{f.stem}_thumb.png"
                    thumb.unlink(missing_ok=True)
            else:
                failed += 1
        except OSError:
            failed += 1

    return deleted, failed
