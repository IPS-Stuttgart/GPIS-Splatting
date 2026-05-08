from __future__ import annotations

import argparse
from pathlib import Path

from gpis_splatting.experiment_matrix import KNOWN_ARTIFACT_ROLES, ExperimentMatrixConfig, default_matrix_cases, run_experiment_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the A-F GPIS/3DGS experiment matrix and aggregate available artifacts.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--prepared-root", default="real_scenes")
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="Defaults to <scene-dir>/evaluations/<matrix-name> or experiments/<matrix-name>.")
    parser.add_argument("--matrix-name", default="gpis_3dgs_matrix")
    parser.add_argument("--primary-geometry-threshold", type=float, default=0.05)
    parser.add_argument("--baseline-case", choices=tuple(case.case_id for case in default_matrix_cases()), default="A")
    parser.add_argument("--fail-on-missing", action="store_true", help="Fail if any A-F case has no matched artifact row.")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="Additional artifact mapping. Roles: " + ", ".join(sorted(KNOWN_ARTIFACT_ROLES)),
    )
    for role, description in KNOWN_ARTIFACT_ROLES.items():
        parser.add_argument(f"--{role.replace('_', '-')}", dest=role, default=None, help=description)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    scene_dir = resolve_scene_dir(args.scene_dir, args.prepared_root, args.scene)
    output_dir = resolve_output_dir(args.output_dir, scene_dir, args.matrix_name)
    result = run_experiment_matrix(
        ExperimentMatrixConfig(
            output_dir=output_dir,
            matrix_name=args.matrix_name,
            scene_dir=scene_dir,
            primary_geometry_threshold=args.primary_geometry_threshold,
            baseline_case=args.baseline_case,
            artifact_paths=collect_artifact_paths(args),
            fail_on_missing=args.fail_on_missing,
        )
    )
    available = result["summary"].loc[result["summary"]["configured"], "case_id"].astype(str).tolist()
    missing = result["summary"].loc[~result["summary"]["configured"], "case_id"].astype(str).tolist()
    print(f"Wrote {result['manifest_path']}")
    print(f"Wrote {result['summary_path']}")
    print(f"Wrote {result['checks_path']}")
    print(f"Wrote {result['status_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"available_cases: {', '.join(available) or 'none'}")
    print(f"missing_cases: {', '.join(missing) or 'none'}")


def resolve_scene_dir(scene_dir: str | None, prepared_root: str, scene: str | None) -> Path | None:
    if scene_dir is not None:
        return Path(scene_dir)
    if scene is not None:
        return Path(prepared_root) / scene
    return None


def resolve_output_dir(output_dir: str | None, scene_dir: Path | None, matrix_name: str) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if scene_dir is not None:
        return scene_dir / "evaluations" / matrix_name
    return Path("experiments") / matrix_name


def collect_artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for role in KNOWN_ARTIFACT_ROLES:
        value = getattr(args, role)
        if value:
            paths[role] = Path(value)
    for item in args.artifact:
        if "=" not in item:
            raise ValueError(f"Artifact mapping must use ROLE=PATH, got {item!r}.")
        role, value = item.split("=", 1)
        if role not in KNOWN_ARTIFACT_ROLES:
            known = ", ".join(sorted(KNOWN_ARTIFACT_ROLES))
            raise ValueError(f"Unknown artifact role {role!r}. Known roles: {known}")
        paths[role] = Path(value)
    return paths


if __name__ == "__main__":
    main()
