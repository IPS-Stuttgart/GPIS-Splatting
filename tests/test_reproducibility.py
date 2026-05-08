from __future__ import annotations

import json
from pathlib import Path

from gpis_splatting.reproducibility import (
    build_reproducibility_report,
    collect_artifact_manifest,
    load_reproducibility_config,
    write_reproducibility_report,
)


def test_load_reproducibility_config_validates_required_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "synthetic_ci.json"
    config_path.write_text(
        json.dumps(
            {
                "description": "Unit config",
                "ablation": {"shapes": ["sphere"]},
                "targets": {"min_psnr_gpis": 5.0},
            }
        ),
        encoding="utf-8",
    )

    config = load_reproducibility_config(config_path)

    assert config["name"] == "synthetic_ci"
    assert config["description"] == "Unit config"


def test_collect_artifact_manifest_hashes_small_files(tmp_path: Path) -> None:
    (tmp_path / "evaluation_status.json").write_text('{"passed": true}', encoding="utf-8")
    (tmp_path / "image.tmp").write_text("ignored", encoding="utf-8")

    manifest = collect_artifact_manifest(tmp_path)

    assert [record.path for record in manifest] == ["evaluation_status.json"]
    assert manifest[0].sha256_12 is not None


def test_build_and_write_reproducibility_report(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "unit",
                "description": "Unit reproducibility config",
                "commands": [{"step": "smoke", "argv": ["run_evaluation", "--preset", "synthetic_ci"]}],
            }
        ),
        encoding="utf-8",
    )
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    (experiment_root / "evaluation_status.json").write_text(
        json.dumps({"preset": "synthetic_ci", "passed": True}),
        encoding="utf-8",
    )
    (experiment_root / "evaluation_checks.csv").write_text("check,passed\nrow_count,True\n", encoding="utf-8")

    report = build_reproducibility_report(
        experiment_root,
        config_path=config_path,
        command_lines=["run_evaluation --preset-config configs/evaluation/synthetic_ci.json"],
        commit="abc123",
    )
    output_path = write_reproducibility_report(experiment_root, config_path=config_path, commit="abc123")

    assert "Git commit: `abc123`" in report
    assert "evaluation_status.json" in report
    assert "evaluation_checks.csv" in report
    assert output_path == experiment_root / "reproducibility_report.md"
    assert output_path.exists()
