"""Synthetic GPIS to uncertainty-gated splat rendering prototype."""

try:
    from gpis_splatting.colmap_camera_models import install_colmap_camera_model_patches

    install_colmap_camera_model_patches()
except Exception:
    # Keep package import lightweight/robust; individual real-data paths still
    # raise explicit errors when unsupported camera models are encountered.
    pass

__all__ = [
    "scenes",
    "gpis",
    "feedback",
    "splats",
    "renderer",
    "real_download",
    "real_pipeline",
    "real_workflow",
    "metrics",
    "gpis_3dgs_regularization",
    "gpis_3dgs_optimization",
    "gpis_3dgs_training_prior",
    "gpis_initialization",
    "calibrated_confidence_api",
    "depth_normal_supervision",
    "gsplat_adapter",
    "colmap_camera_models",
]
