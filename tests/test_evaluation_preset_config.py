from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpis_splatting.evaluation import build_ablation_args
from gpis_splatting.evaluation_config import load_evaluation_preset_file


def test_load_evaluation_preset_file_supports_external_config(tmp_path: Path) -> None:
    config_path = tmp_path / "custom_ci.json"
    config_path.write_text(
        json.dumps(
            {
                "description": "External preset",
                "ablation": {
                    "shapes": ["sphere"],
                    "feedback_iterations": [0],
                    "feedback_selectors": ["gate"],
                    "num_points": 16,
                    "noise_std": 0.02,
                    "grid_size": 6,
                    "lengthscale": 0.8,
                    "variance": 1.0,
                    "image_size": 24,
                    "num_splats": 16,
                    "epsilon": 0.1,
                    "view": "front",
                    "seed": 5,
                    "feedback_pseudo_points": 4,
                    "feedback_min_gate": 0.0,
                },
                "targets": {
                    "max_rmse_sdf": 1.0,
                    "min_iou_inside": 0.0,
                    "min_psnr_gpis": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    preset = load_evaluation_preset_file(config_path)
    args = build_ablation_args(preset, output_root="experiments", experiment_name="custom_ci")

    assert preset["name"] == "custom_ci"
    assert "--shapes" in args
    assert "sphere" in args


def test_load_evaluation_preset_file_rejects_missing_targets(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.json"
    config_path.write_text(json.dumps({"description": "Broken", "ablation": {"shapes": ["sphere"]}}), encoding="utf-8")

    with pytest.raises(ValueError, match="targets"):
        load_evaluation_preset_file(config_path)
