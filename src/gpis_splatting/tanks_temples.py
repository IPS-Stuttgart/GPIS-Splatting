from __future__ import annotations

import http.cookiejar
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from gpis_splatting.real_scene import IMAGE_EXTENSIONS, build_sparse_split, estimate_bounds, validate_real_scene_dir
from gpis_splatting.serialization import read_json, write_json

TANKS_TEMPLES_SOURCE_URL = "https://www.tanksandtemples.org/download/"
TANKS_TEMPLES_TUTORIAL_URL = "https://www.tanksandtemples.org/tutorial/"
TANKS_TEMPLES_LICENSE_URL = "https://www.tanksandtemples.org/license/"
TANKS_TEMPLES_IMAGE_BASE_URL = "https://storage.googleapis.com/t2-downloads/image_sets"
TANKS_TEMPLES_GT_BASE_URL = "https://storage.googleapis.com/t2-training-gt-data"
GOOGLE_DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id={file_id}"
SUPPORTED_TANKS_TEMPLES_SCENES = ("Ignatius",)
TANKS_TEMPLES_RESOURCE_NAMES = ("images", "reconstruction", "camera_log", "alignment", "crop", "ground_truth")


@dataclass(frozen=True)
class TanksTemplesResource:
    name: str
    relative_path: str
    url: str
    source: str
    archive: bool = False


@dataclass(frozen=True)
class TanksTemplesSceneSpec:
    scene: str
    resources: tuple[TanksTemplesResource, ...]


def google_drive_url(file_id: str) -> str:
    return GOOGLE_DRIVE_DOWNLOAD_URL.format(file_id=file_id)


TANKS_TEMPLES_SCENES: dict[str, TanksTemplesSceneSpec] = {
    "Ignatius": TanksTemplesSceneSpec(
        scene="Ignatius",
        resources=(
            TanksTemplesResource(
                name="images",
                relative_path="image_sets/Ignatius.zip",
                url=f"{TANKS_TEMPLES_IMAGE_BASE_URL}/Ignatius.zip",
                source="tanks_temples:image_set",
                archive=True,
            ),
            TanksTemplesResource(
                name="reconstruction",
                relative_path="reconstruction/Ignatius.ply",
                url=google_drive_url("1K4TFKLuD-lvJtU6iY-DsxTlp9y85nb6U"),
                source="tanks_temples:training_colmap_reconstruction",
            ),
            TanksTemplesResource(
                name="camera_log",
                relative_path="camera_poses/Ignatius.log",
                url=google_drive_url("172dDxEcJyA6i2Ih3zy3QNWK1R-2sAIi_"),
                source="tanks_temples:training_camera_poses",
            ),
            TanksTemplesResource(
                name="alignment",
                relative_path="alignment/Ignatius.txt",
                url=google_drive_url("1wSCbCrOT7GsGVLDq0aXs4RHhs4FUzUeM"),
                source="tanks_temples:training_alignment",
            ),
            TanksTemplesResource(
                name="crop",
                relative_path="crop/Ignatius.json",
                url=google_drive_url("1_0fESbNxfNI5NWzQ4RBhh460a9_0J54q"),
                source="tanks_temples:training_crop",
            ),
            TanksTemplesResource(
                name="ground_truth",
                relative_path="ground_truth/Ignatius.ply",
                url=f"{TANKS_TEMPLES_GT_BASE_URL}/Ignatius/Ignatius.ply",
                source="tanks_temples:training_ground_truth",
            ),
        ),
    )
}


def download_tanks_temples_scene(
    *,
    scene: str = "Ignatius",
    output_root: str | Path = "real_scenes/_downloads",
    include_images: bool = True,
    include_auxiliary: bool = True,
    include_ground_truth: bool = True,
    unpack_images: bool = True,
    max_images: int | None = None,
    resources: tuple[str, ...] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if scene not in TANKS_TEMPLES_SCENES:
        raise ValueError(f"Unsupported Tanks and Temples scene {scene!r}. Expected one of {', '.join(SUPPORTED_TANKS_TEMPLES_SCENES)}.")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be positive when provided.")
    if resources is not None:
        unknown = sorted(set(resources) - set(TANKS_TEMPLES_RESOURCE_NAMES))
        if unknown:
            raise ValueError(f"Unsupported Tanks and Temples resources: {', '.join(unknown)}.")

    output_dir = Path(output_root) / "tanks_temples" / scene
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_resources = [
        resource
        for resource in TANKS_TEMPLES_SCENES[scene].resources
        if (include_images or resource.name != "images")
        and (include_auxiliary or resource.name not in {"reconstruction", "camera_log", "alignment", "crop"})
        and (include_ground_truth or resource.name != "ground_truth")
        and (resources is None or resource.name in resources)
    ]

    downloaded = []
    skipped = []
    resources = []
    for resource in selected_resources:
        destination = output_dir / resource.relative_path
        existed = destination.exists()
        if not existed or force:
            destination.parent.mkdir(parents=True, exist_ok=True)
            download_url(resource.url, destination)
            downloaded.append(str(destination))
        else:
            skipped.append(str(destination))
        resources.append(
            {
                "name": resource.name,
                "path": str(destination),
                "url": resource.url,
                "source": resource.source,
                "archive": resource.archive,
            }
        )

    extracted_images: list[str] = []
    image_zip = output_dir / "image_sets" / f"{scene}.zip"
    if include_images and unpack_images and image_zip.exists():
        extracted_images = extract_tanks_temples_images(image_zip, output_dir / "image_sets", max_images=max_images, force=force)

    report_path = output_dir / "tanks_temples_download_report.json"
    report = {
        "schema_version": 1,
        "dataset": "tanks_temples",
        "scene": scene,
        "source_url": TANKS_TEMPLES_SOURCE_URL,
        "tutorial_url": TANKS_TEMPLES_TUTORIAL_URL,
        "license_url": TANKS_TEMPLES_LICENSE_URL,
        "output_dir": str(output_dir),
        "include_images": include_images,
        "include_auxiliary": include_auxiliary,
        "include_ground_truth": include_ground_truth,
        "unpack_images": unpack_images,
        "max_images": max_images,
        "resources_filter": list(resources) if resources is not None else None,
        "downloaded_count": len(downloaded),
        "skipped_count": len(skipped),
        "downloaded": downloaded,
        "skipped": skipped,
        "resources": resources,
        "extracted_image_count": len(extracted_images),
        "extracted_images": extracted_images,
    }
    write_json(report_path, report)
    return {
        "output_dir": output_dir,
        "report_path": report_path,
        "report": report,
    }


def prepare_tanks_temples_scene(
    *,
    input_dir: str | Path,
    output_root: str | Path = "real_scenes",
    scene: str = "Ignatius",
    prepared_scene: str | None = None,
    image_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    reconstruction_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    alignment_path: str | Path | None = None,
    crop_path: str | Path | None = None,
    train_view_count: int = 12,
    copy_images: bool = True,
    focal_length_factor: float = 0.7,
    bounds_scale: float = 1.1,
) -> Path:
    if train_view_count < 1:
        raise ValueError("train_view_count must be positive.")
    if focal_length_factor <= 0.0:
        raise ValueError("focal_length_factor must be positive.")

    source = Path(input_dir)
    if not source.exists():
        raise FileNotFoundError(f"Missing Tanks and Temples input directory: {source}")
    scene_name = prepared_scene or f"{scene.lower()}_tanks_temples"
    images = find_tanks_temples_images(source, scene=scene, image_dir=image_dir)
    log = resolve_tanks_temples_file(source, log_path, candidates=[Path("camera_poses") / f"{scene}.log", Path(f"{scene}.log")], required=True)
    poses = read_tanks_temples_log(log)
    if len(poses) != len(images):
        raise ValueError(f"Tanks and Temples image/log count mismatch: found {len(images)} images and {len(poses)} poses.")

    out_dir = Path(output_root) / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_frames = materialize_tanks_temples_frames(
        images,
        poses,
        out_dir,
        copy_images=copy_images,
        focal_length_factor=focal_length_factor,
    )
    splits = build_sparse_split(len(prepared_frames), train_view_count)
    auxiliary = resolve_tanks_temples_auxiliary(
        source,
        scene=scene,
        reconstruction_path=reconstruction_path,
        ground_truth_path=ground_truth_path,
        alignment_path=alignment_path,
        crop_path=crop_path,
    )
    scene_meta = {
        "schema_version": 1,
        "scene": scene_name,
        "dataset": "tanks_temples",
        "source_scene": scene,
        "source_dir": str(source.resolve()),
        "source_format": "tanks_temples_log",
        "image_count": len(prepared_frames),
        "train_view_count": len(splits["train"]),
        "test_view_count": len(splits["test"]),
        "bounds": estimate_bounds(prepared_frames, scale=bounds_scale),
        "tanks_temples": {
            "source_url": TANKS_TEMPLES_SOURCE_URL,
            "tutorial_url": TANKS_TEMPLES_TUTORIAL_URL,
            "license_url": TANKS_TEMPLES_LICENSE_URL,
            "camera_log_path": str(log),
            "reconstruction_path": str(auxiliary["reconstruction"]) if auxiliary["reconstruction"] is not None else None,
            "ground_truth_path": str(auxiliary["ground_truth"]) if auxiliary["ground_truth"] is not None else None,
            "alignment_path": str(auxiliary["alignment"]) if auxiliary["alignment"] is not None else None,
            "crop_path": str(auxiliary["crop"]) if auxiliary["crop"] is not None else None,
            "focal_length_factor": focal_length_factor,
            "intrinsics_source": "tanks_temples_download_page_recommended_pinhole",
        },
    }
    write_json(out_dir / "real_scene.json", scene_meta)
    write_json(out_dir / "cameras.json", {"schema_version": 1, "frames": prepared_frames})
    write_json(out_dir / "splits.json", splits)
    validation = validate_real_scene_dir(out_dir)
    validation["tanks_temples_assets"] = {key: str(value) if value is not None else None for key, value in auxiliary.items()}
    write_json(out_dir / "validation.json", validation)
    return out_dir


def download_url(url: str, destination: Path) -> None:
    if "drive.google.com" in url:
        download_google_drive_url(url, destination)
        return
    request = urllib.request.Request(url, headers={"User-Agent": "gpis-splatting/0.1"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if looks_like_html(destination):
        raise ValueError(f"Downloaded HTML instead of data for {url}. This usually means the host requires browser confirmation.")


def download_google_drive_url(url: str, destination: Path) -> None:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    request = urllib.request.Request(url, headers={"User-Agent": "gpis-splatting/0.1"})
    response = opener.open(request)
    token = google_drive_confirm_token(cookie_jar)
    if token is not None:
        response.close()
        request = urllib.request.Request(add_query_parameter(url, "confirm", token), headers={"User-Agent": "gpis-splatting/0.1"})
        response = opener.open(request)
    write_response_to_path(response, destination)
    if looks_like_html(destination):
        confirm_url = google_drive_confirm_url_from_html(destination.read_text(encoding="utf-8", errors="ignore"), base_url=url)
        if confirm_url is not None:
            request = urllib.request.Request(confirm_url, headers={"User-Agent": "gpis-splatting/0.1"})
            write_response_to_path(opener.open(request), destination)
    if looks_like_html(destination):
        raise ValueError(f"Downloaded HTML instead of data for {url}. This usually means Google Drive requires browser confirmation.")


def write_response_to_path(response: Any, destination: Path) -> None:
    with response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def google_drive_confirm_token(cookie_jar: http.cookiejar.CookieJar) -> str | None:
    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning"):
            return cookie.value
    return None


def google_drive_confirm_url_from_html(html: str, *, base_url: str) -> str | None:
    parser = GoogleDriveConfirmParser()
    parser.feed(html)
    for action, inputs in parser.forms:
        if "confirm" not in inputs:
            continue
        confirm_url = urllib.parse.urljoin(base_url, action or base_url)
        parsed = urllib.parse.urlparse(confirm_url)
        query = urllib.parse.parse_qs(parsed.query)
        for key, value in inputs.items():
            query[key] = [value]
        return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))

    match = re.search(r"[?&]confirm=([0-9A-Za-z_-]+)", html)
    if match is None:
        match = re.search(r"name=[\"']confirm[\"'][^>]+value=[\"']([^\"']+)[\"']", html)
    if match is None:
        return None
    return add_query_parameter(base_url, "confirm", match.group(1))


class GoogleDriveConfirmParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.forms: list[tuple[str | None, dict[str, str]]] = []
        self._action: str | None = None
        self._inputs: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "form":
            attrs_dict = {key: value for key, value in attrs}
            self._action = attrs_dict.get("action")
            self._inputs = {}
            return
        if tag != "input" or self._inputs is None:
            return
        attrs_dict = {key: value for key, value in attrs}
        name = attrs_dict.get("name")
        value = attrs_dict.get("value")
        if name is not None and value is not None:
            self._inputs[name] = value

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._inputs is not None:
            self.forms.append((self._action, self._inputs))
            self._action = None
            self._inputs = None


def add_query_parameter(url: str, key: str, value: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    query[key] = [value]
    encoded_query = urllib.parse.urlencode(query, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=encoded_query))


def looks_like_html(path: Path) -> bool:
    prefix = path.read_bytes()[:512].lower()
    return b"<html" in prefix or b"<!doctype html" in prefix


def extract_tanks_temples_images(zip_path: Path, output_dir: Path, *, max_images: int | None, force: bool) -> list[str]:
    extracted = []
    with zipfile.ZipFile(zip_path) as archive:
        members = [member for member in archive.infolist() if Path(member.filename).suffix.lower() in IMAGE_EXTENSIONS]
        members = sorted(members, key=lambda member: natural_sort_key(member.filename))
        if max_images is not None:
            members = members[:max_images]
        for member in members:
            target = output_dir / member.filename
            resolved = target.resolve()
            try:
                resolved.relative_to(output_dir.resolve())
            except ValueError:
                raise ValueError(f"Refusing to extract unsafe zip member {member.filename!r}.") from None
            if target.exists() and not force:
                extracted.append(str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(str(target))
    return extracted


def find_tanks_temples_images(source: Path, *, scene: str, image_dir: str | Path | None) -> list[Path]:
    candidates = []
    if image_dir is not None:
        requested = Path(image_dir)
        candidates.append(requested if requested.is_absolute() else source / requested)
    candidates.extend([source / "image_sets" / scene, source / scene, source / "images", source])
    for candidate in candidates:
        if candidate.exists():
            images = [path for path in candidate.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
            images = sorted(images, key=lambda path: natural_sort_key(str(path.relative_to(candidate))))
            if images:
                return images
    raise FileNotFoundError(f"Could not find Tanks and Temples images for {scene!r} under {source}.")


def read_tanks_temples_log(path: str | Path) -> list[dict[str, Any]]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 5 != 0:
        raise ValueError(f"Tanks and Temples log file {path} must contain five lines per pose.")
    poses = []
    for index in range(0, len(lines), 5):
        metadata = [int(value) for value in lines[index].split()]
        if len(metadata) != 3:
            raise ValueError(f"Tanks and Temples log metadata line must contain three integers: {lines[index]!r}")
        matrix = np.asarray([[float(value) for value in lines[index + row + 1].split()] for row in range(4)], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Tanks and Temples log pose at item {index // 5} is not 4x4.")
        poses.append({"metadata": metadata, "camera_to_world": matrix})
    return poses


def materialize_tanks_temples_frames(
    images: list[Path],
    poses: list[dict[str, Any]],
    scene_dir: Path,
    *,
    copy_images: bool,
    focal_length_factor: float,
) -> list[dict[str, Any]]:
    image_out = scene_dir / "images"
    if copy_images:
        image_out.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    frames = []
    for index, (image_path, pose) in enumerate(zip(images, poses, strict=True)):
        width, height = image_size(image_path)
        dest_name = unique_image_name(image_path.name, used_names, index=index)
        if copy_images:
            destination = image_out / dest_name
            shutil.copy2(image_path, destination)
            frame_image_path = Path("images", dest_name).as_posix()
        else:
            frame_image_path = str(image_path.resolve())
        camera_to_world = np.asarray(pose["camera_to_world"], dtype=np.float64)
        frames.append(
            {
                "index": index,
                "image_id": str(pose["metadata"][0]),
                "file_name": dest_name,
                "source_path": str(image_path.resolve()),
                "image_path": frame_image_path,
                "width": width,
                "height": height,
                "camera_id": str(pose["metadata"][1]),
                "tanks_temples_metadata": pose["metadata"],
                "intrinsics": {
                    "model": "PINHOLE",
                    "width": width,
                    "height": height,
                    "fx": float(focal_length_factor * width),
                    "fy": float(focal_length_factor * width),
                    "cx": float(width / 2.0),
                    "cy": float(height / 2.0),
                    "params": [],
                },
                "camera_to_world": camera_to_world.tolist(),
                "world_to_camera": np.linalg.inv(camera_to_world).tolist(),
            }
        )
    return frames


def resolve_tanks_temples_auxiliary(
    source: Path,
    *,
    scene: str,
    reconstruction_path: str | Path | None,
    ground_truth_path: str | Path | None,
    alignment_path: str | Path | None,
    crop_path: str | Path | None,
) -> dict[str, Path | None]:
    return {
        "reconstruction": resolve_tanks_temples_file(source, reconstruction_path, candidates=[Path("reconstruction") / f"{scene}.ply", Path(f"{scene}.ply")], required=False),
        "ground_truth": resolve_tanks_temples_file(source, ground_truth_path, candidates=[Path("ground_truth") / f"{scene}.ply"], required=False),
        "alignment": resolve_tanks_temples_file(source, alignment_path, candidates=[Path("alignment") / f"{scene}.txt"], required=False),
        "crop": resolve_tanks_temples_file(source, crop_path, candidates=[Path("crop") / f"{scene}.json"], required=False),
    }


def resolve_tanks_temples_file(source: Path, path: str | Path | None, *, candidates: list[Path], required: bool) -> Path | None:
    if path is not None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = source / resolved
        if not resolved.exists():
            raise FileNotFoundError(f"Missing Tanks and Temples file: {resolved}")
        return resolved.resolve()
    for candidate in candidates:
        resolved = source / candidate
        if resolved.exists():
            return resolved.resolve()
    if required:
        formatted = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Could not find required Tanks and Temples file under {source}. Tried: {formatted}")
    return None


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def unique_image_name(name: str, used_names: set[str], *, index: int) -> str:
    if name not in used_names:
        used_names.add(name)
        return name
    path = Path(name)
    candidate = f"{index:06d}_{path.name}"
    used_names.add(candidate)
    return candidate


def natural_sort_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def load_tanks_temples_crop(path: str | Path) -> dict[str, Any]:
    return read_json(path)
