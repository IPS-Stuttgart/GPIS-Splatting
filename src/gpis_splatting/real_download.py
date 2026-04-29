from __future__ import annotations

import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from gpis_splatting.serialization import write_json

NERFSTUDIO_POSTER_API_URL = "https://huggingface.co/api/datasets/nerfstudioteam/datasets/tree/main/poster?recursive=true"
NERFSTUDIO_POSTER_RESOLVE_BASE_URL = "https://huggingface.co/datasets/nerfstudioteam/datasets/resolve/main"
REAL_DOWNLOAD_PRESETS = ("nerfstudio_poster",)


def download_real_scene(
    *,
    dataset: str = "nerfstudio_poster",
    output_root: str | Path = "real_scenes/_downloads",
    image_scale: int = 8,
    max_images: int | None = None,
    force: bool = False,
    api_url: str = NERFSTUDIO_POSTER_API_URL,
    resolve_base_url: str = NERFSTUDIO_POSTER_RESOLVE_BASE_URL,
) -> dict[str, Any]:
    if dataset not in REAL_DOWNLOAD_PRESETS:
        raise ValueError(f"Unsupported real scene dataset {dataset!r}. Expected one of {', '.join(REAL_DOWNLOAD_PRESETS)}.")
    if image_scale <= 0:
        raise ValueError("image_scale must be positive.")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be positive when provided.")

    output_dir = Path(output_root) / f"{dataset}_{image_scale}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tree = fetch_huggingface_tree(api_url)
    files = sorted(item["path"] for item in tree if item.get("type") == "file")
    image_prefix = f"poster/images_{image_scale}/"
    image_paths = [path for path in files if path.startswith(image_prefix) and Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if max_images is not None:
        image_paths = image_paths[:max_images]
    required = ["poster/transforms.json", "poster/sparse_pc.ply"]
    requested = required + image_paths
    if not image_paths:
        raise FileNotFoundError(f"No images were found for Hugging Face prefix {image_prefix!r}.")

    downloaded = []
    skipped = []
    for remote_path in requested:
        relative = remote_path.removeprefix("poster/")
        if relative == "transforms.json":
            relative = "transforms_fullres.json"
        destination = output_dir / Path(relative)
        if destination.exists() and not force:
            skipped.append(str(destination))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        download_url = huggingface_resolve_url(resolve_base_url, remote_path)
        download_file(download_url, destination)
        downloaded.append(str(destination))

    available_images = {Path(path).name for path in image_paths}
    transform_report = write_scaled_nerfstudio_transforms(
        source_path=output_dir / "transforms_fullres.json",
        output_path=output_dir / "transforms.json",
        image_dir=f"images_{image_scale}",
        image_scale=image_scale,
        available_images=available_images,
    )
    report_path = output_dir / "download_report.json"
    report = {
        "schema_version": 1,
        "dataset": dataset,
        "source": "huggingface:nerfstudioteam/datasets/poster",
        "api_url": api_url,
        "resolve_base_url": resolve_base_url,
        "output_dir": str(output_dir),
        "image_scale": image_scale,
        "max_images": max_images,
        "image_count": len(image_paths),
        "downloaded_count": len(downloaded),
        "skipped_count": len(skipped),
        "downloaded": downloaded,
        "skipped": skipped,
        "transforms": transform_report,
        "sparse_point_cloud": str(output_dir / "sparse_pc.ply"),
    }
    write_json(report_path, report)
    return {
        "output_dir": output_dir,
        "report_path": report_path,
        "report": report,
    }


def fetch_huggingface_tree(api_url: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(api_url) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError(f"Expected Hugging Face tree API to return a list, got {type(data).__name__}.")
    return data


def huggingface_resolve_url(resolve_base_url: str, remote_path: str) -> str:
    encoded = "/".join(urllib.parse.quote(part) for part in remote_path.split("/"))
    return f"{resolve_base_url.rstrip('/')}/{encoded}?download=true"


def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def write_scaled_nerfstudio_transforms(
    *,
    source_path: str | Path,
    output_path: str | Path,
    image_dir: str,
    image_scale: int,
    available_images: set[str] | None = None,
) -> dict[str, Any]:
    source = Path(source_path)
    output = Path(output_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    original_count = len(data.get("frames", []))
    for key in ("fl_x", "fl_y", "cx", "cy"):
        if key in data:
            data[key] = float(data[key]) / image_scale
    for key in ("w", "h"):
        if key in data:
            data[key] = int(round(float(data[key]) / image_scale))

    filtered_frames = []
    missing = []
    for frame in data.get("frames", []):
        image_name = Path(frame["file_path"]).name
        if available_images is not None and image_name not in available_images:
            missing.append(image_name)
            continue
        next_frame = dict(frame)
        next_frame["file_path"] = f"./{image_dir}/{image_name}"
        filtered_frames.append(next_frame)
    data["frames"] = filtered_frames
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {
        "source_path": str(source),
        "output_path": str(output),
        "image_dir": image_dir,
        "image_scale": image_scale,
        "original_frame_count": original_count,
        "kept_frame_count": len(filtered_frames),
        "missing_frame_count": len(missing),
    }
