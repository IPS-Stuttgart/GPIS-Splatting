from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch

from gpis_splatting.gpis import GPISModel, GPISPrediction, fit_dense_gpis, load_model, predict_gpis, rbf_kernel, save_model

Tensor = torch.Tensor
GPISBackendName = Literal["dense_exact", "local_exact"]


@runtime_checkable
class GPISBackend(Protocol):
    """Common interface for GPIS implementations.

    The dense Cholesky implementation remains the reference backend. Approximate or
    partitioned implementations should expose this interface so evaluation and future
    training-time regularization code can swap inference engines without changing
    downstream splat-confidence logic.
    """

    backend_name: str

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def device(self) -> torch.device: ...

    def predict(self, x_query: Tensor, *, batch_size: int = 8192) -> GPISPrediction: ...

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> None: ...


@dataclass(frozen=True)
class DenseExactGPISBackend:
    """Backend wrapper around the current exact dense GPIS implementation."""

    model: GPISModel
    backend_name: str = "dense_exact"

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    @property
    def device(self) -> torch.device:
        return self.model.device

    @classmethod
    def fit(
        cls,
        x_train: Tensor,
        y_train: Tensor,
        *,
        lengthscale: float = 0.34,
        variance: float = 1.0,
        noise_std: float = 0.035,
        observation_noise_std: Tensor | None = None,
        mean_constant: float | None = None,
        jitter: float = 1e-6,
    ) -> "DenseExactGPISBackend":
        return cls(
            fit_dense_gpis(
                x_train,
                y_train,
                lengthscale=lengthscale,
                variance=variance,
                noise_std=noise_std,
                observation_noise_std=observation_noise_std,
                mean_constant=mean_constant,
                jitter=jitter,
            )
        )

    def predict(self, x_query: Tensor, *, batch_size: int = 8192) -> GPISPrediction:
        return predict_gpis(self.model, x_query, batch_size=batch_size)

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> None:
        merged_metadata = {"backend": self.backend_name}
        if metadata:
            merged_metadata.update(metadata)
        save_model(str(path), self.model, metadata=merged_metadata)

    @classmethod
    def load(cls, path: str | Path) -> tuple["DenseExactGPISBackend", dict[str, object]]:
        model, metadata = load_model(str(path))
        metadata.setdefault("backend", "dense_exact")
        return cls(model), metadata


@dataclass(frozen=True)
class LocalExactGPISBackend:
    """Local exact GPIS backend with bounded per-query Cholesky systems.

    This backend stores all observations but avoids the global O(n^2) kernel matrix.
    Each query is predicted from its k nearest training observations through a small
    exact GP solve. It is an approximation to the dense posterior, but it provides a
    practical scaling baseline and exercises the backend abstraction before adding
    more sophisticated SKI/inducing/CUDA backends.
    """

    x_train: Tensor
    y_train: Tensor
    lengthscale: float = 0.34
    variance: float = 1.0
    noise_std: float = 0.035
    mean_constant: float = 0.0
    jitter: float = 1e-6
    observation_noise_std: Tensor | None = None
    num_neighbors: int = 64
    backend_name: str = "local_exact"

    @property
    def dtype(self) -> torch.dtype:
        return self.x_train.dtype

    @property
    def device(self) -> torch.device:
        return self.x_train.device

    @classmethod
    def fit(
        cls,
        x_train: Tensor,
        y_train: Tensor,
        *,
        lengthscale: float = 0.34,
        variance: float = 1.0,
        noise_std: float = 0.035,
        observation_noise_std: Tensor | None = None,
        mean_constant: float | None = None,
        jitter: float = 1e-6,
        num_neighbors: int = 64,
    ) -> "LocalExactGPISBackend":
        if num_neighbors < 1:
            raise ValueError("num_neighbors must be positive.")
        x_train = x_train.detach().to(dtype=torch.float64, device="cpu")
        y_train = y_train.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        validate_training_inputs(x_train, y_train)
        if observation_noise_std is None:
            observation_noise = None
        else:
            observation_noise = observation_noise_std.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
            if observation_noise.shape != y_train.shape:
                raise ValueError("observation_noise_std must have one value per training observation.")
        if mean_constant is None:
            mean_constant = float(y_train.mean())
        return cls(
            x_train=x_train,
            y_train=y_train,
            lengthscale=float(lengthscale),
            variance=float(variance),
            noise_std=float(noise_std),
            mean_constant=float(mean_constant),
            jitter=float(jitter),
            observation_noise_std=observation_noise,
            num_neighbors=int(num_neighbors),
        )

    def predict(self, x_query: Tensor, *, batch_size: int = 8192) -> GPISPrediction:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        x_query = x_query.detach().to(dtype=self.dtype, device=self.device)
        if x_query.numel() == 0:
            return GPISPrediction(
                mean=torch.empty((0,), dtype=self.dtype, device=self.device),
                variance=torch.empty((0,), dtype=self.dtype, device=self.device),
                gradient=torch.empty((0, self.x_train.shape[1]), dtype=self.dtype, device=self.device),
            )
        means: list[Tensor] = []
        variances: list[Tensor] = []
        gradients: list[Tensor] = []
        for start in range(0, x_query.shape[0], batch_size):
            batch = x_query[start : start + batch_size]
            neighbor_indices = nearest_training_indices(
                batch,
                self.x_train,
                num_neighbors=min(self.num_neighbors, int(self.x_train.shape[0])),
            )
            for query, indices in zip(batch, neighbor_indices, strict=True):
                mean, variance, gradient = self._predict_one(query, indices)
                means.append(mean)
                variances.append(variance)
                gradients.append(gradient)
        return GPISPrediction(mean=torch.stack(means), variance=torch.stack(variances), gradient=torch.stack(gradients))

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> None:
        data: dict[str, object] = {
            "backend_name": np.array(self.backend_name),
            "x_train": self.x_train.detach().cpu().numpy(),
            "y_train": self.y_train.detach().cpu().numpy(),
            "lengthscale": np.array(self.lengthscale),
            "variance": np.array(self.variance),
            "noise_std": np.array(self.noise_std),
            "mean_constant": np.array(self.mean_constant),
            "jitter": np.array(self.jitter),
            "num_neighbors": np.array(self.num_neighbors, dtype=np.int64),
        }
        if self.observation_noise_std is not None:
            data["observation_noise_std"] = self.observation_noise_std.detach().cpu().numpy()
        if metadata:
            for key, value in metadata.items():
                data[f"meta_{key}"] = np.array(value)
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str | Path) -> tuple["LocalExactGPISBackend", dict[str, object]]:
        npz = np.load(path, allow_pickle=False)
        model = cls(
            x_train=torch.from_numpy(npz["x_train"]).to(dtype=torch.float64),
            y_train=torch.from_numpy(npz["y_train"]).to(dtype=torch.float64),
            lengthscale=float(npz["lengthscale"]),
            variance=float(npz["variance"]),
            noise_std=float(npz["noise_std"]),
            mean_constant=float(npz["mean_constant"]),
            jitter=float(npz["jitter"]),
            observation_noise_std=torch.from_numpy(npz["observation_noise_std"]).to(dtype=torch.float64)
            if "observation_noise_std" in npz.files
            else None,
            num_neighbors=int(npz["num_neighbors"]),
        )
        metadata = metadata_from_npz(npz)
        metadata.setdefault("backend", "local_exact")
        return model, metadata

    def _predict_one(self, query: Tensor, indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x_local = self.x_train[indices]
        y_local = self.y_train[indices]
        if self.observation_noise_std is None:
            noise_variance = torch.full_like(y_local, self.noise_std**2)
        else:
            noise_variance = torch.clamp(self.observation_noise_std[indices], min=1e-8).pow(2)
        kernel = rbf_kernel(x_local, x_local, self.lengthscale, self.variance)
        eye = torch.eye(x_local.shape[0], dtype=self.dtype, device=self.device)
        system = kernel + torch.diag(noise_variance) + self.jitter * eye
        chol, info = torch.linalg.cholesky_ex(system)
        if int(info.item()) != 0:
            raise RuntimeError(f"Local Cholesky factorization failed at leading minor {int(info.item())}.")
        centered = y_local - self.mean_constant
        alpha = torch.cholesky_solve(centered[:, None], chol).reshape(-1)
        k_x = rbf_kernel(query[None, :], x_local, self.lengthscale, self.variance).reshape(-1)
        mean = self.mean_constant + torch.dot(k_x, alpha)
        solved = torch.cholesky_solve(k_x[:, None], chol).reshape(-1)
        variance = torch.clamp(torch.as_tensor(self.variance, dtype=self.dtype, device=self.device) - torch.dot(k_x, solved), min=1e-12)
        diff = x_local - query[None, :]
        gradient = torch.sum((k_x * alpha)[:, None] * diff, dim=0) / (self.lengthscale**2)
        return mean, variance, gradient


def fit_gpis_backend(
    backend: GPISBackendName,
    x_train: Tensor,
    y_train: Tensor,
    *,
    lengthscale: float = 0.34,
    variance: float = 1.0,
    noise_std: float = 0.035,
    observation_noise_std: Tensor | None = None,
    mean_constant: float | None = None,
    jitter: float = 1e-6,
    num_neighbors: int = 64,
) -> GPISBackend:
    if backend == "dense_exact":
        return DenseExactGPISBackend.fit(
            x_train,
            y_train,
            lengthscale=lengthscale,
            variance=variance,
            noise_std=noise_std,
            observation_noise_std=observation_noise_std,
            mean_constant=mean_constant,
            jitter=jitter,
        )
    if backend == "local_exact":
        return LocalExactGPISBackend.fit(
            x_train,
            y_train,
            lengthscale=lengthscale,
            variance=variance,
            noise_std=noise_std,
            observation_noise_std=observation_noise_std,
            mean_constant=mean_constant,
            jitter=jitter,
            num_neighbors=num_neighbors,
        )
    raise ValueError(f"Unknown GPIS backend '{backend}'. Expected one of: dense_exact, local_exact.")


def load_gpis_backend(path: str | Path) -> tuple[GPISBackend, dict[str, object]]:
    npz = np.load(path, allow_pickle=False)
    backend_name = read_backend_name(npz)
    npz.close()
    if backend_name in (None, "dense_exact"):
        return DenseExactGPISBackend.load(path)
    if backend_name == "local_exact":
        return LocalExactGPISBackend.load(path)
    raise ValueError(f"Unsupported GPIS backend in {path}: {backend_name!r}.")


def nearest_training_indices(x_query: Tensor, x_train: Tensor, *, num_neighbors: int) -> Tensor:
    if num_neighbors < 1:
        raise ValueError("num_neighbors must be positive.")
    if x_train.shape[0] == 0:
        raise ValueError("At least one training observation is required.")
    k = min(int(num_neighbors), int(x_train.shape[0]))
    distances = torch.cdist(x_query, x_train)
    return torch.topk(distances, k=k, dim=1, largest=False).indices


def validate_training_inputs(x_train: Tensor, y_train: Tensor) -> None:
    if x_train.ndim != 2:
        raise ValueError("x_train must have shape (n_observations, n_dims).")
    if y_train.ndim != 1:
        raise ValueError("y_train must have shape (n_observations,).")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must contain the same number of observations.")
    if x_train.shape[0] < 1:
        raise ValueError("At least one training observation is required.")


def read_backend_name(npz: np.lib.npyio.NpzFile) -> str | None:
    if "backend_name" in npz.files:
        return str(npz["backend_name"].item())
    if "meta_backend" in npz.files:
        return str(npz["meta_backend"].item())
    return None


def metadata_from_npz(npz: np.lib.npyio.NpzFile) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key in npz.files:
        if key.startswith("meta_"):
            value = npz[key]
            metadata[key.removeprefix("meta_")] = value.item() if value.shape == () else value
    return metadata
