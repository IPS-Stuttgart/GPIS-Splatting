from __future__ import annotations

from pathlib import Path
from typing import Any

from gpis_splatting.serialization import read_json


def load_evaluation_preset_file(path: str | Path) -> dict[str, Any]:
    """Load an external evaluation preset JSON file.

    The schema mirrors entries in ``gpis_splatting.evaluation.EVALUATION_PRESETS``
    so benchmark settings can be versioned under ``configs/`` without editing
    Python source for every smoke or benchmark preset.
    """

    preset_path = Path(path)
    preset = read_json(preset_path)
    if not isinstance(preset, dict):
        raise ValueError(f"Evaluation preset must be a JSON object: {preset_path}")
    preset.setdefault("name", preset_path.stem)
    validate_evaluation_preset(preset, source=str(preset_path))
    return preset


def validate_evaluation_preset(preset: dict[str, Any], *, source: str = "external preset") -> None:
    required_top_level = ("description", "ablation", "targets")
    missing_top_level = [key for key in required_top_level if key not in preset]
    if missing_top_level:
        raise ValueError(f"Evaluation preset {source} is missing: {', '.join(missing_top_level)}")

    required_ablation = (
        "shapes",
        "feedback_iterations",
        "feedback_selectors",
        "num_points",
        "noise_std",
        "grid_size",
        "lengthscale",
        "variance",
        "image_size",
        "num_splats",
        "epsilon",
        "view",
        "seed",
    )
    missing_ablation = [key for key in required_ablation if key not in preset["ablation"]]
    if missing_ablation:
        raise ValueError(f"Evaluation preset {source} is missing ablation keys: {', '.join(missing_ablation)}")

    required_targets = ("max_rmse_sdf", "min_iou_inside", "min_psnr_gpis")
    missing_targets = [key for key in required_targets if key not in preset["targets"]]
    if missing_targets:
        raise ValueError(f"Evaluation preset {source} is missing target keys: {', '.join(missing_targets)}")
