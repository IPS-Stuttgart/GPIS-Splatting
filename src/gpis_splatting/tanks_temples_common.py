from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from gpis_splatting.serialization import read_json


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
