from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .math_utils import clamp_positive, normal_cdf

Tensor = torch.Tensor


@dataclass
class GPISModel:
    x_train: Tensor
    y_train: Tensor
    alpha: Tensor
    chol: Tensor
    lengthscale: float
    variance: float
    noise_std: float
    mean_constant: float
    jitter: float = 1e-6
    observation_noise_std: Tensor | None = None

    @property
    def dtype(self) -> torch.dtype:
        return self.x_train.dtype

    @property
    def device(self) -> torch.device:
        return self.x_train.device


@dataclass
class GPISPrediction:
    mean: Tensor
    variance: Tensor
    gradient: Tensor

    @property
    def std(self) -> Tensor:
        return torch.sqrt(clamp_positive(self.variance))

    @property
    def grad_norm(self) -> Tensor:
        return torch.clamp(torch.linalg.norm(self.gradient, dim=-1), min=1e-6)

    @property
    def distance(self) -> Tensor:
        return self.mean / self.grad_norm

    @property
    def distance_std(self) -> Tensor:
        return self.std / self.grad_norm

    @property
    def inside_probability(self) -> Tensor:
        return normal_cdf(-self.mean / self.std)


def rbf_kernel(x1: Tensor, x2: Tensor, lengthscale: float, variance: float) -> Tensor:
    x1s = x1 / lengthscale
    x2s = x2 / lengthscale
    sqdist = torch.cdist(x1s, x2s).pow(2)
    return variance * torch.exp(-0.5 * sqdist)


def fit_dense_gpis(
    x_train: Tensor,
    y_train: Tensor,
    *,
    lengthscale: float = 0.34,
    variance: float = 1.0,
    noise_std: float = 0.035,
    observation_noise_std: Tensor | None = None,
    mean_constant: float | None = None,
    jitter: float = 1e-6,
) -> GPISModel:
    x_train = x_train.detach().to(dtype=torch.float64, device="cpu")
    y_train = y_train.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    if observation_noise_std is None:
        observation_noise = None
        noise_variance = torch.full_like(y_train, noise_std**2)
    else:
        observation_noise = observation_noise_std.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        if observation_noise.shape != y_train.shape:
            raise ValueError("observation_noise_std must have one value per training observation.")
        noise_variance = torch.clamp(observation_noise, min=1e-8).pow(2)

    if mean_constant is None:
        mean_constant = float(y_train.mean())

    kernel = rbf_kernel(x_train, x_train, lengthscale, variance)
    eye = torch.eye(x_train.shape[0], dtype=x_train.dtype, device=x_train.device)
    system = kernel + torch.diag(noise_variance) + jitter * eye
    chol, info = torch.linalg.cholesky_ex(system)
    if int(info.item()) != 0:
        raise RuntimeError(f"Cholesky factorization failed at leading minor {int(info.item())}.")
    centered_y = y_train - mean_constant
    alpha = torch.cholesky_solve(centered_y[:, None], chol).reshape(-1)
    return GPISModel(
        x_train,
        y_train,
        alpha,
        chol,
        lengthscale,
        variance,
        noise_std,
        mean_constant,
        jitter,
        observation_noise,
    )


def predict_gpis(model: GPISModel, x_query: Tensor, batch_size: int = 8192) -> GPISPrediction:
    x_query = x_query.detach().to(dtype=model.dtype, device=model.device)
    means: list[Tensor] = []
    variances: list[Tensor] = []
    gradients: list[Tensor] = []

    for start in range(0, x_query.shape[0], batch_size):
        x_batch = x_query[start : start + batch_size]
        k_x = rbf_kernel(x_batch, model.x_train, model.lengthscale, model.variance)
        mean = model.mean_constant + k_x @ model.alpha

        solved = torch.cholesky_solve(k_x.T, model.chol)
        variance = model.variance - torch.sum(k_x.T * solved, dim=0)
        variance = torch.clamp(variance, min=1e-12)

        diff = model.x_train[None, :, :] - x_batch[:, None, :]
        weighted = k_x * model.alpha[None, :]
        gradient = torch.sum(weighted[:, :, None] * diff, dim=1) / (model.lengthscale**2)

        means.append(mean)
        variances.append(variance)
        gradients.append(gradient)

    return GPISPrediction(
        mean=torch.cat(means, dim=0),
        variance=torch.cat(variances, dim=0),
        gradient=torch.cat(gradients, dim=0),
    )


def surface_band_probability(
    prediction: GPISPrediction,
    epsilon: float,
    *,
    min_std: float = 1e-4,
) -> Tensor:
    distance = prediction.distance
    distance_std = torch.clamp(prediction.distance_std, min=min_std)
    upper = (epsilon - distance) / distance_std
    lower = (-epsilon - distance) / distance_std
    return torch.clamp(normal_cdf(upper) - normal_cdf(lower), min=0.0, max=1.0)


def save_model(path: str, model: GPISModel, *, metadata: dict[str, object] | None = None) -> None:
    data = {
        "x_train": model.x_train.detach().cpu().numpy(),
        "y_train": model.y_train.detach().cpu().numpy(),
        "alpha": model.alpha.detach().cpu().numpy(),
        "chol": model.chol.detach().cpu().numpy(),
        "lengthscale": np.array(model.lengthscale),
        "variance": np.array(model.variance),
        "noise_std": np.array(model.noise_std),
        "mean_constant": np.array(model.mean_constant),
        "jitter": np.array(model.jitter),
    }
    if model.observation_noise_std is not None:
        data["observation_noise_std"] = model.observation_noise_std.detach().cpu().numpy()
    if metadata:
        for key, value in metadata.items():
            data[f"meta_{key}"] = np.array(value)
    np.savez_compressed(path, **data)


def load_model(path: str) -> tuple[GPISModel, dict[str, object]]:
    npz = np.load(path, allow_pickle=False)
    model = GPISModel(
        x_train=torch.from_numpy(npz["x_train"]).to(dtype=torch.float64),
        y_train=torch.from_numpy(npz["y_train"]).to(dtype=torch.float64),
        alpha=torch.from_numpy(npz["alpha"]).to(dtype=torch.float64),
        chol=torch.from_numpy(npz["chol"]).to(dtype=torch.float64),
        lengthscale=float(npz["lengthscale"]),
        variance=float(npz["variance"]),
        noise_std=float(npz["noise_std"]),
        mean_constant=float(npz["mean_constant"]) if "mean_constant" in npz.files else 0.0,
        jitter=float(npz["jitter"]),
        observation_noise_std=torch.from_numpy(npz["observation_noise_std"]).to(dtype=torch.float64)
        if "observation_noise_std" in npz.files
        else None,
    )
    metadata: dict[str, object] = {}
    for key in npz.files:
        if key.startswith("meta_"):
            value = npz[key]
            metadata[key.removeprefix("meta_")] = value.item() if value.shape == () else value
    return model, metadata
