from __future__ import annotations

import argparse

import numpy as np

from gpis_splatting.feedback import refine_gpis_with_splat_feedback, save_feedback_trace
from gpis_splatting.gpis import load_model, save_model
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
    parser.add_argument("--feedback-iterations", type=int, default=0, help="Number of GPIS-splat feedback refits.")
    parser.add_argument(
        "--feedback-pseudo-points",
        type=int,
        default=80,
        help="Maximum high-confidence splats promoted to GPIS pseudo observations per iteration.",
    )
    parser.add_argument("--feedback-min-gate", type=float, default=0.55, help="Minimum GPIS gate for feedback splats.")
    parser.add_argument(
        "--feedback-pseudo-noise-std",
        type=float,
        default=None,
        help="Pseudo-observation noise floor. Defaults to the fitted GPIS observation noise.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.feedback_iterations > 0 and not args.use_gpis_gate:
        raise ValueError("--feedback-iterations requires --use-gpis-gate true.")

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
    feedback_gate = None
    if args.use_gpis_gate:
        model_path = out_dir / "gpis_model.npz"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}. Run fit_gpis before using GPIS gates.")
        model, _ = load_model(str(model_path))
        gate = gpis_gate_for_splats(splats, model, args.epsilon)
        np.savez_compressed(out_dir / "splat_gates.npz", gate=gate.detach().cpu().numpy(), epsilon=np.array(args.epsilon))
        if args.feedback_iterations > 0:
            feedback = refine_gpis_with_splat_feedback(
                model,
                splats,
                args.epsilon,
                iterations=args.feedback_iterations,
                pseudo_points_per_iteration=args.feedback_pseudo_points,
                min_gate=args.feedback_min_gate,
                pseudo_noise_std=args.feedback_pseudo_noise_std,
            )
            feedback_gate = feedback.feedback_gate
            save_model(
                str(out_dir / "feedback_gpis_model.npz"),
                feedback.model,
                metadata={
                    "shape": shape,
                    "scene": args.scene,
                    "feedback_iterations": args.feedback_iterations,
                    "feedback_pseudo_points": args.feedback_pseudo_points,
                    "feedback_min_gate": args.feedback_min_gate,
                },
            )
            save_feedback_trace(out_dir / "feedback_trace.csv", feedback.trace)
            np.savez_compressed(
                out_dir / "feedback_splat_gates.npz",
                base_gate=feedback.base_gate.detach().cpu().numpy(),
                feedback_gate=feedback.feedback_gate.detach().cpu().numpy(),
                selected_mask=feedback.selected_mask.detach().cpu().numpy(),
                epsilon=np.array(args.epsilon),
                iterations=np.array(args.feedback_iterations),
            )

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
        if feedback_gate is not None:
            feedback_render = render_splats(
                splats,
                image_size=args.image_size,
                bounds=bounds,
                view=view,
                gate=feedback_gate,
            )
            save_image(out_dir / f"render_feedback_{view}.png", feedback_render)

    config["render"] = {
        "image_size": args.image_size,
        "num_splats": int(splats.centers.shape[0]),
        "epsilon": args.epsilon,
        "use_gpis_gate": args.use_gpis_gate,
        "views": selected_views(args.view),
        "feedback_iterations": args.feedback_iterations,
        "feedback_pseudo_points": args.feedback_pseudo_points,
        "feedback_min_gate": args.feedback_min_gate,
        "feedback_pseudo_noise_std": args.feedback_pseudo_noise_std,
    }
    write_json(config_path, config)

    print(f"Wrote {splat_path}")
    for view in selected_views(args.view):
        print(f"Wrote {out_dir / f'render_reference_{view}.png'}")
        print(f"Wrote {out_dir / f'render_plain_{view}.png'}")
        if args.use_gpis_gate:
            print(f"Wrote {out_dir / f'render_gpis_{view}.png'}")
        if feedback_gate is not None:
            print(f"Wrote {out_dir / f'render_feedback_{view}.png'}")
    if feedback_gate is not None:
        print(f"Wrote {out_dir / 'feedback_gpis_model.npz'}")
        print(f"Wrote {out_dir / 'feedback_trace.csv'}")
        print(f"Wrote {out_dir / 'feedback_splat_gates.npz'}")


if __name__ == "__main__":
    main()
