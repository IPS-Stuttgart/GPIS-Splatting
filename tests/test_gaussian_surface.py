from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gpis_splatting.cli.evaluate_3dgs_gpis_surfaces import main as evaluate_3dgs_gpis_surfaces_main
from gpis_splatting.cli.extract_3dgs_gpis_surfaces import main as extract_3dgs_gpis_surfaces_main
from gpis_splatting.gaussian_surface import read_surface_ply
from gpis_splatting.serialization import read_json


def test_extract_gpis_gated_gaussian_surfaces_writes_surfel_and_opacity_meshes(tmp_path: Path) -> None:
    ply_path = tmp_path / "point_cloud.ply"
    write_surface_fixture_3dgs_ply(ply_path)
    gate_path = tmp_path / "gate.npz"
    np.savez_compressed(gate_path, gate=np.asarray([0.2, 0.8, 0.9, 0.1], dtype=np.float64))
    output_dir = tmp_path / "surfaces"

    extract_3dgs_gpis_surfaces_main(
        [
            "--input-ply",
            str(ply_path),
            "--gate-path",
            str(gate_path),
            "--output-dir",
            str(output_dir),
            "--method-name",
            "toy_surface",
            "--gate-thresholds",
            "0.5",
            "--extraction-modes",
            "centers",
            "surfels",
            "opacity_field",
            "--opacity-field-resolution",
            "8",
            "--opacity-field-threshold",
            "0.05",
            "--max-field-gaussians",
            "0",
        ]
    )

    manifest = pd.read_csv(output_dir / "toy_surface_surface_manifest.csv")
    assert set(manifest["extraction_method"]) == {"centers", "surfels", "opacity_field"}
    assert set(manifest["variant"]) == {"baseline", "gate_ge_0p5"}
    filtered_centers = manifest[(manifest["variant"] == "gate_ge_0p5") & (manifest["extraction_method"] == "centers")].iloc[0]
    assert int(filtered_centers["retained_count"]) == 2
    assert float(filtered_centers["retention_fraction"]) == 0.5

    surfel = read_surface_ply(output_dir / "toy_surface_gate_ge_0p5_surfels.ply")
    assert surfel.vertices.shape == (8, 3)
    assert surfel.faces.shape == (4, 3)
    assert np.all(np.isfinite(surfel.vertices))

    opacity = read_surface_ply(output_dir / "toy_surface_gate_ge_0p5_opacity_field.ply")
    assert opacity.vertices.shape[1] == 3
    assert opacity.faces.shape[1] == 3
    assert (output_dir / "toy_surface_surface_report.md").exists()
    assert read_json(output_dir / "toy_surface_surface_status.json")["surface_count"] == 6


def test_evaluate_gpis_gated_gaussian_surfaces_writes_geometry_tables(tmp_path: Path) -> None:
    ply_path = tmp_path / "point_cloud.ply"
    write_surface_fixture_3dgs_ply(ply_path)
    gate_path = tmp_path / "gate.npz"
    np.savez_compressed(gate_path, gate=np.asarray([0.2, 0.8, 0.9, 0.1], dtype=np.float64))
    output_dir = tmp_path / "surfaces"
    extract_3dgs_gpis_surfaces_main(
        [
            "--input-ply",
            str(ply_path),
            "--gate-path",
            str(gate_path),
            "--output-dir",
            str(output_dir),
            "--method-name",
            "toy_surface",
            "--gate-thresholds",
            "0.5",
            "--extraction-modes",
            "centers",
            "surfels",
        ]
    )
    ground_truth = tmp_path / "ground_truth.ply"
    write_point_cloud_ply(ground_truth, np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64))

    evaluate_3dgs_gpis_surfaces_main(
        [
            "--manifest-path",
            str(output_dir / "toy_surface_surface_manifest.csv"),
            "--ground-truth-path",
            str(ground_truth),
            "--output-dir",
            str(tmp_path / "surface_eval"),
            "--method-name",
            "toy_surface",
            "--thresholds",
            "0.05",
            "0.25",
            "--max-pred-points",
            "32",
            "--max-gt-points",
            "0",
        ]
    )

    summary = pd.read_csv(tmp_path / "surface_eval" / "toy_surface_surface_geometry_summary.csv")
    thresholds = pd.read_csv(tmp_path / "surface_eval" / "toy_surface_surface_geometry_thresholds.csv")
    assert set(summary["extraction_method"]) == {"centers", "surfels"}
    baseline_centers = summary[(summary["variant"] == "baseline") & (summary["extraction_method"] == "centers")].iloc[0]
    assert baseline_centers["chamfer_l1"] < 1e-9
    assert set(thresholds["threshold"]) == {0.05, 0.25}
    assert (tmp_path / "surface_eval" / "toy_surface_surface_geometry_report.md").exists()


def write_surface_fixture_3dgs_ply(path: Path) -> None:
    rows = [
        [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 2.0, -3.0, -2.0, -2.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.2, 0.3, 0.4, 2.0, -3.0, -2.0, -2.0, 1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.3, 0.4, 0.5, 2.0, -3.0, -2.0, -2.0, 1.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.4, 0.5, 0.6, 2.0, -3.0, -2.0, -2.0, 1.0, 0.0, 0.0, 0.0],
    ]
    header = [
        "ply",
        "format ascii 1.0",
        "element vertex 4",
        "property float x",
        "property float y",
        "property float z",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]
    body = [" ".join(str(value) for value in row) for row in rows]
    path.write_text("\n".join([*header, *body]) + "\n", encoding="ascii")


def write_point_cloud_ply(path: Path, points: np.ndarray) -> None:
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    body = [f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}" for point in points]
    path.write_text("\n".join([*header, *body]) + "\n", encoding="ascii")
