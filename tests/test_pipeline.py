from __future__ import annotations

from pathlib import Path

import pandas as pd

from gpis_splatting.cli.evaluate import main as evaluate_main
from gpis_splatting.cli.fit_gpis import main as fit_main
from gpis_splatting.cli.generate_scene import main as generate_main
from gpis_splatting.cli.render_splats import main as render_main


def test_small_end_to_end_pipeline(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    scene = "sphere_regression"

    generate_main(
        [
            "--shape",
            "sphere",
            "--scene",
            scene,
            "--num-points",
            "80",
            "--noise-std",
            "0.03",
            "--seed",
            "5",
            "--output-root",
            str(root),
        ]
    )
    fit_main(
        [
            "--scene",
            scene,
            "--grid-size",
            "12",
            "--output-root",
            str(root),
        ]
    )
    render_main(
        [
            "--scene",
            scene,
            "--view",
            "front",
            "--image-size",
            "48",
            "--num-splats",
            "120",
            "--epsilon",
            "0.11",
            "--output-root",
            str(root),
        ]
    )
    evaluate_main(["--scene", scene, "--output-root", str(root)])

    out_dir = root / scene
    assert (out_dir / "config.json").exists()
    assert (out_dir / "samples.npz").exists()
    assert (out_dir / "gpis_model.npz").exists()
    assert (out_dir / "posterior_grid.npz").exists()
    assert (out_dir / "render_plain_front.png").exists()
    assert (out_dir / "render_gpis_front.png").exists()
    assert (out_dir / "metrics.csv").exists()

    metrics = pd.read_csv(out_dir / "metrics.csv").iloc[0]
    assert metrics["rmse_sdf"] < 0.35
    assert metrics["iou_inside"] > 0.35
    assert metrics["psnr_gpis_front"] >= metrics["psnr_plain_front"]
