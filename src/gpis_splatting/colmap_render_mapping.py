from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Any

from gpis_splatting.serialization import write_json

RENDER_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
LINK_MODES = ("copy", "hardlink", "symlink")


def write_render_name_map(path: str | Path, frames: list[dict[str, Any]], *, split: str) -> None:
    rows = build_render_name_rows(frames, split=split)
    write_mapping_csv(path, rows)


def build_render_name_rows(frames: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    rows = []
    for render_index, frame in enumerate(frames):
        colmap_name = str(frame["colmap_image_name"])
        rows.append(
            {
                "render_index": render_index,
                "render_name": f"{render_index:05d}.png",
                "colmap_image_id": int(frame["colmap_image_id"]),
                "colmap_image_name": colmap_name,
                "prepared_frame_index": int(frame["index"]),
                "prepared_file_name": str(frame.get("file_name") or Path(colmap_name).name),
                "prepared_image_path": str(frame["image_path"]),
                "split": split,
            }
        )
    return rows


def write_mapping_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "render_index",
        "render_name",
        "colmap_image_id",
        "colmap_image_name",
        "prepared_frame_index",
        "prepared_file_name",
        "prepared_image_path",
        "split",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def map_3dgs_renders_to_prepared_scene(
    *,
    map_path: str | Path,
    renders_dir: str | Path,
    output_dir: str | Path,
    link_mode: str = "copy",
    require_all: bool = True,
    overwrite: bool = True,
) -> dict[str, Any]:
    if link_mode not in LINK_MODES:
        raise ValueError(f"Unsupported link_mode {link_mode!r}. Expected one of {', '.join(LINK_MODES)}.")

    rows = read_mapping_csv(map_path)
    source_root = Path(renders_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    mapped_rows = []
    missing = []
    for row in rows:
        source = find_render_source(source_root, row)
        target = out_root / safe_prepared_output_path(row)
        if source is None:
            missing.append(row)
            if require_all:
                continue
            mapped_rows.append(mapping_status_row(row, source_path=None, target_path=target, status="missing"))
            continue
        materialize_link(source=source, target=target, link_mode=link_mode, overwrite=overwrite)
        mapped_rows.append(mapping_status_row(row=row, source_path=source, target_path=target, status="mapped"))

    if missing and require_all:
        examples = [row.get("render_name") or row.get("colmap_image_name") for row in missing[:5]]
        raise FileNotFoundError(f"Missing {len(missing)} rendered images under {source_root}: {examples}")
    if not mapped_rows:
        raise ValueError(f"No rendered images were mapped from {source_root}.")

    mapping_csv_path = out_root / "mapped_render_images.csv"
    status_path = out_root / "mapped_render_images_status.json"
    report_path = out_root / "mapped_render_images_report.md"
    write_mapping_status_csv(mapping_csv_path, mapped_rows)
    status = {
        "schema_version": 1,
        "map_path": str(map_path),
        "renders_dir": str(source_root),
        "output_dir": str(out_root),
        "link_mode": link_mode,
        "require_all": require_all,
        "overwrite": overwrite,
        "row_count": len(rows),
        "mapped_count": sum(1 for row in mapped_rows if row["status"] == "mapped"),
        "missing_count": len(missing),
        "mapping_csv_path": str(mapping_csv_path),
        "report_path": str(report_path),
    }
    write_json(status_path, status)
    report_path.write_text(format_mapping_report(status), encoding="utf-8")
    return {"status": status, "status_path": status_path, "mapping_csv_path": mapping_csv_path, "report_path": report_path}


def read_mapping_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Render-name map is empty: {path}")
    required = {"render_index", "render_name", "colmap_image_name", "prepared_image_path"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Render-name map {path} is missing columns: {', '.join(missing)}.")
    return rows


def find_render_source(root: Path, row: dict[str, Any]) -> Path | None:
    for candidate in render_source_candidates(root, row):
        if candidate.is_file():
            return candidate
    return None


def render_source_candidates(root: Path, row: dict[str, Any]) -> list[Path]:
    render_index = int(row["render_index"])
    names = [
        str(row.get("render_name") or ""),
        f"{render_index:05d}.png",
        f"{render_index:06d}.png",
        f"{render_index:03d}.png",
        str(row.get("colmap_image_name") or ""),
        str(row.get("prepared_file_name") or ""),
        Path(str(row.get("prepared_image_path") or "")).name,
    ]
    expanded = []
    for name in names:
        if not name:
            continue
        path = Path(name)
        expanded.append(path.name)
        if path.suffix:
            expanded.append(f"{path.stem}.png")
        else:
            expanded.extend(f"{path.name}{suffix}" for suffix in RENDER_IMAGE_SUFFIXES)
    candidates = [root / name for name in unique_names(expanded)]
    return candidates


def safe_prepared_output_path(row: dict[str, Any]) -> Path:
    raw_path = Path(str(row.get("prepared_image_path") or row.get("prepared_file_name") or row["render_name"]))
    if raw_path.is_absolute():
        return Path(raw_path.name)
    parts = []
    for part in raw_path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError(f"Unsafe prepared image path in render map: {raw_path}")
        parts.append(part)
    if not parts:
        return Path(str(row["render_name"]))
    return Path(*parts)


def materialize_link(*, source: Path, target: Path, link_mode: str, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and source.resolve(strict=True) == target.resolve(strict=True):
        return
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Mapped render already exists: {target}")
        target.unlink()
    if link_mode == "copy":
        shutil.copy2(source, target)
    elif link_mode == "hardlink":
        os.link(source, target)
    elif link_mode == "symlink":
        target.symlink_to(source)
    else:
        raise ValueError(f"Unsupported link_mode {link_mode!r}.")


def mapping_status_row(*, row: dict[str, Any], source_path: Path | None, target_path: Path, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "render_index": row["render_index"],
        "render_name": row["render_name"],
        "colmap_image_name": row.get("colmap_image_name"),
        "prepared_frame_index": row.get("prepared_frame_index"),
        "prepared_image_path": row.get("prepared_image_path"),
        "source_path": "" if source_path is None else str(source_path),
        "target_path": str(target_path),
    }


def write_mapping_status_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "status",
        "render_index",
        "render_name",
        "colmap_image_name",
        "prepared_frame_index",
        "prepared_image_path",
        "source_path",
        "target_path",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def unique_names(names: list[str]) -> list[str]:
    seen = set()
    unique = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def format_mapping_report(status: dict[str, Any]) -> str:
    lines = [
        "# 3DGS Render Mapping",
        "",
        f"- Render-name map: `{status['map_path']}`",
        f"- Source renders: `{status['renders_dir']}`",
        f"- Output directory: `{status['output_dir']}`",
        f"- Link mode: `{status['link_mode']}`",
        f"- Mapped images: `{status['mapped_count']}` of `{status['row_count']}`",
        f"- Missing images: `{status['missing_count']}`",
        f"- Mapping CSV: `{status['mapping_csv_path']}`",
    ]
    return "\n".join(lines) + "\n"
