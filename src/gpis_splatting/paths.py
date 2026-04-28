from __future__ import annotations

from pathlib import Path


DEFAULT_EXPERIMENT_ROOT = Path("experiments")


def scene_dir(scene: str, root: str | Path = DEFAULT_EXPERIMENT_ROOT) -> Path:
    return Path(root) / scene


def ensure_scene_dir(scene: str, root: str | Path = DEFAULT_EXPERIMENT_ROOT) -> Path:
    path = scene_dir(scene, root)
    path.mkdir(parents=True, exist_ok=True)
    return path

