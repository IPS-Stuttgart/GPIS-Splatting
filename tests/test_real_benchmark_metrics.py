from __future__ import annotations

import numpy as np

from gpis_splatting.real_benchmark import ssim_arrays


def test_ssim_arrays_returns_one_for_identical_images() -> None:
    image = np.linspace(0.0, 1.0, 32 * 32 * 3, dtype=np.float64).reshape(32, 32, 3)

    assert np.isclose(ssim_arrays(image, image), 1.0, atol=1e-12)


def test_ssim_arrays_matches_slow_local_gaussian_reference() -> None:
    rng = np.random.default_rng(7)
    target = rng.random((16, 17, 3), dtype=np.float64)
    prediction = np.clip(target + rng.normal(0.0, 0.05, size=target.shape), 0.0, 1.0)

    assert np.isclose(ssim_arrays(prediction, target), _slow_local_ssim_reference(prediction, target), atol=1e-12)


def test_ssim_arrays_handles_tiny_images() -> None:
    target = np.zeros((6, 8, 3), dtype=np.float64)
    prediction = np.full_like(target, 2.0 / 255.0)

    assert np.isfinite(ssim_arrays(prediction, target))


def _slow_local_ssim_reference(prediction: np.ndarray, target: np.ndarray) -> float:
    window_size = 11
    sigma = 1.5
    radius = window_size // 2
    kernel_1d = _gaussian_kernel(window_size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)[..., np.newaxis]
    prediction_pad = np.pad(prediction, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    target_pad = np.pad(target, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")

    height, width, channels = prediction.shape
    mu_x = np.empty((height, width, channels), dtype=np.float64)
    mu_y = np.empty_like(mu_x)
    mean_xx = np.empty_like(mu_x)
    mean_yy = np.empty_like(mu_x)
    mean_xy = np.empty_like(mu_x)
    for y in range(height):
        for x in range(width):
            pred_window = prediction_pad[y : y + window_size, x : x + window_size, :]
            target_window = target_pad[y : y + window_size, x : x + window_size, :]
            mu_x[y, x, :] = np.sum(kernel_2d * pred_window, axis=(0, 1))
            mu_y[y, x, :] = np.sum(kernel_2d * target_window, axis=(0, 1))
            mean_xx[y, x, :] = np.sum(kernel_2d * pred_window * pred_window, axis=(0, 1))
            mean_yy[y, x, :] = np.sum(kernel_2d * target_window * target_window, axis=(0, 1))
            mean_xy[y, x, :] = np.sum(kernel_2d * pred_window * target_window, axis=(0, 1))

    c1 = 0.01**2
    c2 = 0.03**2
    sigma_x_sq = np.maximum(mean_xx - mu_x * mu_x, 0.0)
    sigma_y_sq = np.maximum(mean_yy - mu_y * mu_y, 0.0)
    sigma_xy = mean_xy - mu_x * mu_y
    ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return float(np.clip(np.mean(ssim_map[radius:-radius, radius:-radius, :]), -1.0, 1.0))


def _gaussian_kernel(window_size: int, sigma: float) -> np.ndarray:
    offsets = np.arange(window_size, dtype=np.float64) - window_size // 2
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()
