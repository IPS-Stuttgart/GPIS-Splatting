from __future__ import annotations

import argparse

import numpy as np

from gpis_splatting.gpis import load_model
from gpis_splatting.paths import scene_dir
from gpis_splatting.renderer import render_splats, save_image, selected_views
from gpis_splatting.serialization import read_json, write_json
from gpis_splatting.splats import gpis_gate_for_splats, load_splats, make_candidate_splats, save_splats


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render candidate splats with optional GPIS optical-depth gate.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--use-gpis-gate", type=str_to_bool, default=True)
    parser.add_argument("--view", default="all", help="all, front, side, or top")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-splats", type=int, default=700)
    parser.add_argument("--epsilon", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = scene_dir(args.scene, args.output_root)
    config_path = out_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}. Run generate_scene first.")
    config = read_json(config_path)
    shape = config["shape"]
    bounds = tuple(config["bounds"])

    splat_path = out_dir / "splats.npz"
    if splat_path.exists():
        splats = load_splats(str(splat_path))
    else:
        splats = make_candidate_splats(shape, num_splats=args.num_splats, seed=args.seed)
        save_splats(str(splat_path), splats)

    model = None
    gate = None
    if args.use_gpis_gate:
        model_path = out_dir / "gpis_model.npz"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}. Run fit_gpis before using GPIS gates.")
        model, _ = load_model(str(model_path))
        gate = gpis_gate_for_splats(splats, model, args.epsilon)
        np.savez_compressed(out_dir / "splat_gates.npz", gate=gate.detach().cpu().numpy(), epsilon=np.array(args.epsilon))

    for view in selected_views(args.view):
        reference = render_splats(
            splats,
            image_size=args.image_size,
            bounds=bounds,
            view=view,
            surface_only=True,
        )
        plain = render_splats(splats, image_size=args.image_size, bounds=bounds, view=view, gate=None)
        save_image(out_dir / f"render_reference_{view}.png", reference)
        save_image(out_dir / f"render_plain_{view}.png", plain)

        if args.use_gpis_gate and gate is not None:
            gated = render_splats(splats, image_size=args.image_size, bounds=bounds, view=view, gate=gate)
            save_image(out_dir / f"render_gpis_{view}.png", gated)

    config["render"] = {
        "image_size": args.image_size,
        "num_splats": int(splats.centers.shape[0]),
        "epsilon": args.epsilon,
        "use_gpis_gate": args.use_gpis_gate,
        "views": selected_views(args.view),
    }
    write_json(config_path, config)

    print(f"Wrote {splat_path}")
    for view in selected_views(args.view):
        print(f"Wrote {out_dir / f'render_reference_{view}.png'}")
        print(f"Wrote {out_dir / f'render_plain_{view}.png'}")
        if args.use_gpis_gate:
            print(f"Wrote {out_dir / f'render_gpis_{view}.png'}")


if __name__ == "__main__":
    main()
