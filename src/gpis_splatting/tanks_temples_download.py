from __future__ import annotations

import http.cookiejar
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any

from gpis_splatting.real_scene import IMAGE_EXTENSIONS
from gpis_splatting.serialization import write_json
from gpis_splatting.tanks_temples_common import natural_sort_key
from gpis_splatting.tanks_temples_resources import (
    SUPPORTED_TANKS_TEMPLES_SCENES,
    TANKS_TEMPLES_LICENSE_URL,
    TANKS_TEMPLES_RESOURCE_NAMES,
    TANKS_TEMPLES_SCENES,
    TANKS_TEMPLES_SOURCE_URL,
    TANKS_TEMPLES_TUTORIAL_URL,
)


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
    resource_records = []
    archive_cache: dict[str, Path] = {}
    for resource in selected_resources:
        destination = output_dir / resource.relative_path
        existed = destination.exists()
        if not existed or force:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if resource.archive_member is None:
                download_url(resource.url, destination)
            else:
                archive_path = archive_cache.setdefault(resource.url, output_dir / "_archives" / archive_cache_filename(resource.url))
                if force or not archive_path.exists():
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    download_url(resource.url, archive_path)
                extract_archive_member(archive_path, destination, member_name=resource.archive_member)
            downloaded.append(str(destination))
        else:
            skipped.append(str(destination))
        resource_records.append(
            {
                "name": resource.name,
                "path": str(destination),
                "url": resource.url,
                "source": resource.source,
                "archive": resource.archive,
                "archive_member": resource.archive_member,
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
        "resources": resource_records,
        "extracted_image_count": len(extracted_images),
        "extracted_images": extracted_images,
    }
    write_json(report_path, report)
    return {"output_dir": output_dir, "report_path": report_path, "report": report}


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


def archive_cache_filename(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if "id" in query and query["id"]:
        return f"{sanitize_filename(query['id'][0])}.zip"
    name = Path(parsed.path).name
    return sanitize_filename(name or "archive.zip")


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return sanitized or "archive.zip"


def extract_archive_member(zip_path: Path, destination: Path, *, member_name: str) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and PurePosixPath(member.filename).name == member_name
            and "__MACOSX" not in PurePosixPath(member.filename).parts
            and not PurePosixPath(member.filename).name.startswith("._")
        ]
        if not members:
            raise FileNotFoundError(f"Could not find {member_name!r} in Tanks and Temples archive {zip_path}.")
        if len(members) > 1:
            formatted = ", ".join(member.filename for member in members)
            raise ValueError(f"Ambiguous Tanks and Temples archive member {member_name!r}: {formatted}")
        with archive.open(members[0]) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)


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
