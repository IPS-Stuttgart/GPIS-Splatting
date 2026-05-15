from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch

from gpis_splatting.gpis import GPISModel, GPISPrediction, fit_dense_gpis, load_model, predict_gpis, rbf_kernel, save_model

Tensor = torch.Tensor
GPISBackendName = Literal["dense_exact", "local_exact", "local_kdtree", "local_faiss", "inducing_points", "ard_inducing_points", "ski_grid", "multires_inducing"]
InducingSelectionName = Literal["farthest", "uniform", "first"]


@runtime_checkable
class GPISBackend(Protocol):
    backend_name: str

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def device(self) -> torch.device: ...

    def predict(self, x_query: Tensor, *, batch_size: int = 8192) -> GPISPrediction: ...

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> None: ...


@dataclass(frozen=True)
class DenseExactGPISBackend:
    model: GPISModel
    backend_name: str = "dense_exact"

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    @property
    def device(self) -> torch.device:
        return self.model.device

    @classmethod
    def fit(cls, x_train: Tensor, y_train: Tensor, *, lengthscale: float = 0.34, variance: float = 1.0, noise_std: float = 0.035, observation_noise_std: Tensor | None = None, mean_constant: float | None = None, jitter: float = 1e-6) -> "DenseExactGPISBackend":
        return cls(fit_dense_gpis(x_train, y_train, lengthscale=lengthscale, variance=variance, noise_std=noise_std, observation_noise_std=observation_noise_std, mean_constant=mean_constant, jitter=jitter))

    def predict(self, x_query: Tensor, *, batch_size: int = 8192) -> GPISPrediction:
        return predict_gpis(self.model, x_query, batch_size=batch_size)

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> None:
        meta = {"backend": self.backend_name}
        if metadata:
            meta.update(metadata)
        save_model(str(path), self.model, metadata=meta)

    @classmethod
    def load(cls, path: str | Path) -> tuple["DenseExactGPISBackend", dict[str, object]]:
        model, metadata = load_model(str(path))
        metadata.setdefault("backend", "dense_exact")
        return cls(model), metadata


@dataclass(frozen=True)
class LocalExactGPISBackend:
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
    def fit(cls, x_train: Tensor, y_train: Tensor, *, lengthscale: float = 0.34, variance: float = 1.0, noise_std: float = 0.035, observation_noise_std: Tensor | None = None, mean_constant: float | None = None, jitter: float = 1e-6, num_neighbors: int = 64, backend_name: str = "local_exact") -> "LocalExactGPISBackend":
        if num_neighbors < 1:
            raise ValueError("num_neighbors must be positive.")
        x_train = x_train.detach().to(dtype=torch.float64, device="cpu")
        y_train = y_train.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        validate_training_inputs(x_train, y_train)
        noise = normalize_observation_noise(observation_noise_std, y_train)
        return cls(x_train, y_train, float(lengthscale), float(variance), float(noise_std), float(y_train.mean() if mean_constant is None else mean_constant), float(jitter), noise, int(num_neighbors), backend_name)

    def _neighbor_indices(self, batch: Tensor) -> Tensor:
        return nearest_training_indices(batch, self.x_train, num_neighbors=min(self.num_neighbors, int(self.x_train.shape[0])))

    def predict(self, x_query: Tensor, *, batch_size: int = 8192) -> GPISPrediction:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        x_query = x_query.detach().to(dtype=self.dtype, device=self.device)
        if x_query.numel() == 0:
            return GPISPrediction(torch.empty((0,), dtype=self.dtype), torch.empty((0,), dtype=self.dtype), torch.empty((0, self.x_train.shape[1]), dtype=self.dtype))
        means: list[Tensor] = []
        variances: list[Tensor] = []
        gradients: list[Tensor] = []
        for start in range(0, x_query.shape[0], batch_size):
            batch = x_query[start : start + batch_size]
            for query, indices in zip(batch, self._neighbor_indices(batch), strict=True):
                mean, variance, gradient = self._predict_one(query, indices)
                means.append(mean)
                variances.append(variance)
                gradients.append(gradient)
        return GPISPrediction(torch.stack(means), torch.stack(variances), torch.stack(gradients))

    def _predict_one(self, query: Tensor, indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x_local = self.x_train[indices]
        y_local = self.y_train[indices]
        noise_var = torch.full_like(y_local, self.noise_std**2) if self.observation_noise_std is None else torch.clamp(self.observation_noise_std[indices], min=1e-8).pow(2)
        kernel = rbf_kernel(x_local, x_local, self.lengthscale, self.variance)
        chol = cholesky_with_jitter(kernel + torch.diag(noise_var), jitter=self.jitter, error_prefix="Local")
        alpha = torch.cholesky_solve((y_local - self.mean_constant)[:, None], chol).reshape(-1)
        k_x = rbf_kernel(query[None, :], x_local, self.lengthscale, self.variance).reshape(-1)
        solved = torch.cholesky_solve(k_x[:, None], chol).reshape(-1)
        mean = self.mean_constant + torch.dot(k_x, alpha)
        variance = torch.clamp(torch.as_tensor(self.variance, dtype=self.dtype) - torch.dot(k_x, solved), min=1e-12)
        gradient = torch.sum((k_x * alpha)[:, None] * (x_local - query[None, :]), dim=0) / (self.lengthscale**2)
        return mean, variance, gradient

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> None:
        data: dict[str, object] = {"backend_name": np.array(self.backend_name), "x_train": self.x_train.numpy(), "y_train": self.y_train.numpy(), "lengthscale": np.array(self.lengthscale), "variance": np.array(self.variance), "noise_std": np.array(self.noise_std), "mean_constant": np.array(self.mean_constant), "jitter": np.array(self.jitter), "num_neighbors": np.array(self.num_neighbors, dtype=np.int64)}
        if self.observation_noise_std is not None:
            data["observation_noise_std"] = self.observation_noise_std.numpy()
        if metadata:
            for key, value in metadata.items():
                data[f"meta_{key}"] = np.array(value)
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str | Path) -> tuple["LocalExactGPISBackend", dict[str, object]]:
        npz = np.load(path, allow_pickle=False)
        backend_name = read_backend_name(npz) or "local_exact"
        model = cls(torch.from_numpy(npz["x_train"]).to(dtype=torch.float64), torch.from_numpy(npz["y_train"]).to(dtype=torch.float64), float(npz["lengthscale"]), float(npz["variance"]), float(npz["noise_std"]), float(npz["mean_constant"]), float(npz["jitter"]), torch.from_numpy(npz["observation_noise_std"]).to(dtype=torch.float64) if "observation_noise_std" in npz.files else None, int(npz["num_neighbors"]), backend_name)
        metadata = metadata_from_npz(npz)
        metadata.setdefault("backend", backend_name)
        return model, metadata


@dataclass(frozen=True)
class KDTreeLocalExactGPISBackend(LocalExactGPISBackend):
    backend_name: str = "local_kdtree"

    @classmethod
    def fit(cls, x_train: Tensor, y_train: Tensor, **kwargs: object) -> "KDTreeLocalExactGPISBackend":
        kwargs.pop("leaf_size", None)
        base = LocalExactGPISBackend.fit(x_train, y_train, backend_name="local_kdtree", **kwargs)
        return cls(**vars(base))

    def _neighbor_indices(self, batch: Tensor) -> Tensor:
        try:
            from scipy.spatial import cKDTree  # type: ignore[import-not-found]
        except Exception:
            return super()._neighbor_indices(batch)
        _, idx = cKDTree(self.x_train.numpy()).query(batch.numpy(), k=min(self.num_neighbors, int(self.x_train.shape[0])))
        if idx.ndim == 1:
            idx = idx[:, None]
        return torch.as_tensor(idx, dtype=torch.long)


@dataclass(frozen=True)
class FaissLocalExactGPISBackend(LocalExactGPISBackend):
    backend_name: str = "local_faiss"

    @classmethod
    def fit(cls, x_train: Tensor, y_train: Tensor, **kwargs: object) -> "FaissLocalExactGPISBackend":
        base = LocalExactGPISBackend.fit(x_train, y_train, backend_name="local_faiss", **kwargs)
        return cls(**vars(base))

    def _neighbor_indices(self, batch: Tensor) -> Tensor:
        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("The local_faiss backend requires faiss-cpu.") from exc
        index = faiss.IndexFlatL2(int(self.x_train.shape[1]))
        index.add(self.x_train.numpy().astype(np.float32, copy=False))
        _, idx = index.search(batch.numpy().astype(np.float32, copy=False), min(self.num_neighbors, int(self.x_train.shape[0])))
        return torch.as_tensor(idx, dtype=torch.long)


@dataclass(frozen=True)
class InducingPointGPISBackend:
    inducing_points: Tensor
    weight_mean: Tensor
    weight_cov: Tensor
    chol_uu: Tensor
    lengthscale: float = 0.34
    variance: float = 1.0
    noise_std: float = 0.035
    mean_constant: float = 0.0
    jitter: float = 1e-6
    training_count: int = 0
    inducing_selection: InducingSelectionName = "farthest"
    ard_lengthscales: Tensor | None = None
    backend_name: str = "inducing_points"

    @property
    def dtype(self) -> torch.dtype:
        return self.inducing_points.dtype

    @property
    def device(self) -> torch.device:
        return self.inducing_points.device

    @property
    def num_inducing(self) -> int:
        return int(self.inducing_points.shape[0])

    @classmethod
    def fit(cls, x_train: Tensor, y_train: Tensor, *, lengthscale: float = 0.34, variance: float = 1.0, noise_std: float = 0.035, observation_noise_std: Tensor | None = None, mean_constant: float | None = None, jitter: float = 1e-6, num_inducing: int = 512, inducing_selection: InducingSelectionName = "farthest", fit_batch_size: int = 8192, ard_lengthscales: Tensor | None = None, inducing_points: Tensor | None = None, backend_name: str = "inducing_points") -> "InducingPointGPISBackend":
        x_train = x_train.detach().to(dtype=torch.float64, device="cpu")
        y_train = y_train.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        validate_training_inputs(x_train, y_train)
        if mean_constant is None:
            mean_constant = float(y_train.mean())
        if ard_lengthscales is not None:
            ard_lengthscales = normalize_ard_lengthscales(ard_lengthscales, x_train.shape[1])
        if inducing_points is None:
            scaled = x_train / ard_lengthscales[None, :] if ard_lengthscales is not None else x_train
            inducing_points = x_train[select_inducing_indices(scaled, num_inducing=num_inducing, method=inducing_selection)].contiguous()
        kernel = (lambda a, b: ard_rbf_kernel(a, b, ard_lengthscales, variance)) if ard_lengthscales is not None else (lambda a, b: rbf_kernel(a, b, lengthscale, variance))
        obs_noise = normalize_observation_noise(observation_noise_std, y_train)
        m = int(inducing_points.shape[0])
        chol_uu = cholesky_with_jitter(kernel(inducing_points, inducing_points), jitter=jitter, error_prefix="Inducing K_uu")
        precision = torch.eye(m, dtype=torch.float64)
        rhs = torch.zeros((m,), dtype=torch.float64)
        for start in range(0, x_train.shape[0], fit_batch_size):
            bx = x_train[start : start + fit_batch_size]
            by = y_train[start : start + fit_batch_size] - float(mean_constant)
            features = solve_lower_triangular_features(kernel(bx, inducing_points), chol_uu)
            proj_var = torch.sum(features * features, dim=1)
            noise_var = torch.full_like(by, noise_std**2) if obs_noise is None else torch.clamp(obs_noise[start : start + fit_batch_size], min=1e-8).pow(2)
            eff_noise = torch.clamp(noise_var + torch.clamp(torch.as_tensor(variance, dtype=torch.float64) - proj_var, min=0.0), min=1e-12)
            weighted = features / eff_noise[:, None]
            precision += features.T @ weighted
            rhs += weighted.T @ by
        post_chol = cholesky_with_jitter(precision, jitter=jitter, error_prefix="Inducing posterior")
        return cls(inducing_points, torch.cholesky_solve(rhs[:, None], post_chol).reshape(-1), torch.cholesky_inverse(post_chol), chol_uu, float(lengthscale), float(variance), float(noise_std), float(mean_constant), float(jitter), int(x_train.shape[0]), inducing_selection, ard_lengthscales, backend_name)

    def _kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        return ard_rbf_kernel(x1, x2, self.ard_lengthscales, self.variance) if self.ard_lengthscales is not None else rbf_kernel(x1, x2, self.lengthscale, self.variance)

    def _grad_scale(self) -> Tensor:
        return self.ard_lengthscales.pow(2) if self.ard_lengthscales is not None else torch.full((self.inducing_points.shape[1],), self.lengthscale**2, dtype=self.dtype)

    def predict(self, x_query: Tensor, *, batch_size: int = 8192) -> GPISPrediction:
        x_query = x_query.detach().to(dtype=self.dtype, device=self.device)
        means: list[Tensor] = []
        variances: list[Tensor] = []
        gradients: list[Tensor] = []
        kernel_alpha = torch.linalg.solve_triangular(self.chol_uu.T, self.weight_mean[:, None], upper=True).reshape(-1)
        for start in range(0, x_query.shape[0], batch_size):
            batch = x_query[start : start + batch_size]
            k_xu = self._kernel(batch, self.inducing_points)
            features = solve_lower_triangular_features(k_xu, self.chol_uu)
            mean = self.mean_constant + features @ self.weight_mean
            proj = torch.sum(features * features, dim=1)
            post = torch.sum((features @ self.weight_cov) * features, dim=1)
            var = torch.clamp(torch.clamp(torch.as_tensor(self.variance, dtype=self.dtype) - proj, min=0.0) + post, min=1e-12)
            grad = torch.sum((k_xu * kernel_alpha[None, :])[:, :, None] * (self.inducing_points[None, :, :] - batch[:, None, :]) / self._grad_scale()[None, None, :], dim=1)
            means.append(mean); variances.append(var); gradients.append(grad)
        return GPISPrediction(torch.cat(means), torch.cat(variances), torch.cat(gradients))

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> None:
        data: dict[str, object] = {"backend_name": np.array(self.backend_name), "inducing_points": self.inducing_points.numpy(), "weight_mean": self.weight_mean.numpy(), "weight_cov": self.weight_cov.numpy(), "chol_uu": self.chol_uu.numpy(), "lengthscale": np.array(self.lengthscale), "variance": np.array(self.variance), "noise_std": np.array(self.noise_std), "mean_constant": np.array(self.mean_constant), "jitter": np.array(self.jitter), "training_count": np.array(self.training_count, dtype=np.int64), "inducing_selection": np.array(self.inducing_selection)}
        if self.ard_lengthscales is not None:
            data["ard_lengthscales"] = self.ard_lengthscales.numpy()
        if metadata:
            for key, value in metadata.items():
                data[f"meta_{key}"] = np.array(value)
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str | Path) -> tuple["InducingPointGPISBackend", dict[str, object]]:
        npz = np.load(path, allow_pickle=False)
        backend_name = read_backend_name(npz) or "inducing_points"
        model = cls(torch.from_numpy(npz["inducing_points"]).to(dtype=torch.float64), torch.from_numpy(npz["weight_mean"]).to(dtype=torch.float64), torch.from_numpy(npz["weight_cov"]).to(dtype=torch.float64), torch.from_numpy(npz["chol_uu"]).to(dtype=torch.float64), float(npz["lengthscale"]), float(npz["variance"]), float(npz["noise_std"]), float(npz["mean_constant"]), float(npz["jitter"]), int(npz["training_count"]), str(npz["inducing_selection"].item()) if "inducing_selection" in npz.files else "farthest", torch.from_numpy(npz["ard_lengthscales"]).to(dtype=torch.float64) if "ard_lengthscales" in npz.files else None, backend_name)
        metadata = metadata_from_npz(npz); metadata.setdefault("backend", backend_name)
        return model, metadata


class ARDInducingPointGPISBackend(InducingPointGPISBackend):
    pass


class SKIGridGPISBackend(InducingPointGPISBackend):
    pass


@dataclass(frozen=True)
class MultiresInducingGPISBackend:
    levels: tuple[InducingPointGPISBackend, ...]
    backend_name: str = "multires_inducing"

    @property
    def dtype(self) -> torch.dtype:
        return self.levels[0].dtype

    @property
    def device(self) -> torch.device:
        return self.levels[0].device

    @property
    def num_inducing(self) -> int:
        return sum(level.num_inducing for level in self.levels)

    @property
    def training_count(self) -> int:
        return self.levels[0].training_count

    @classmethod
    def fit(cls, x_train: Tensor, y_train: Tensor, *, multires_levels: int = 3, multires_lengthscale_decay: float = 0.55, multires_inducing_growth: float = 1.5, num_inducing: int = 128, lengthscale: float = 0.34, **kwargs: object) -> "MultiresInducingGPISBackend":
        residual = y_train.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        levels: list[InducingPointGPISBackend] = []
        for level in range(multires_levels):
            backend = InducingPointGPISBackend.fit(x_train, residual, num_inducing=max(1, int(round(num_inducing * multires_inducing_growth**level))), lengthscale=lengthscale * multires_lengthscale_decay**level, backend_name="multires_level", **kwargs)
            levels.append(backend)
            residual = residual - backend.predict(x_train).mean.detach()
        return cls(tuple(levels))

    def predict(self, x_query: Tensor, *, batch_size: int = 8192) -> GPISPrediction:
        preds = [level.predict(x_query, batch_size=batch_size) for level in self.levels]
        return GPISPrediction(torch.stack([p.mean for p in preds]).sum(0), torch.stack([p.variance for p in preds]).sum(0), torch.stack([p.gradient for p in preds]).sum(0))

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> None:
        data: dict[str, object] = {"backend_name": np.array(self.backend_name), "num_levels": np.array(len(self.levels), dtype=np.int64)}
        for i, level in enumerate(self.levels):
            for name in ("inducing_points", "weight_mean", "weight_cov", "chol_uu"):
                data[f"level_{i}_{name}"] = getattr(level, name).numpy()
            for name in ("lengthscale", "variance", "noise_std", "mean_constant", "jitter", "training_count"):
                data[f"level_{i}_{name}"] = np.array(getattr(level, name))
        if metadata:
            for key, value in metadata.items():
                data[f"meta_{key}"] = np.array(value)
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str | Path) -> tuple["MultiresInducingGPISBackend", dict[str, object]]:
        npz = np.load(path, allow_pickle=False)
        levels = []
        for i in range(int(npz["num_levels"])):
            levels.append(InducingPointGPISBackend(torch.from_numpy(npz[f"level_{i}_inducing_points"]).to(dtype=torch.float64), torch.from_numpy(npz[f"level_{i}_weight_mean"]).to(dtype=torch.float64), torch.from_numpy(npz[f"level_{i}_weight_cov"]).to(dtype=torch.float64), torch.from_numpy(npz[f"level_{i}_chol_uu"]).to(dtype=torch.float64), float(npz[f"level_{i}_lengthscale"]), float(npz[f"level_{i}_variance"]), float(npz[f"level_{i}_noise_std"]), float(npz[f"level_{i}_mean_constant"]), float(npz[f"level_{i}_jitter"]), int(npz[f"level_{i}_training_count"])))
        metadata = metadata_from_npz(npz); metadata.setdefault("backend", "multires_inducing")
        return cls(tuple(levels)), metadata


def fit_gpis_backend(backend: GPISBackendName, x_train: Tensor, y_train: Tensor, *, lengthscale: float = 0.34, variance: float = 1.0, noise_std: float = 0.035, observation_noise_std: Tensor | None = None, mean_constant: float | None = None, jitter: float = 1e-6, num_neighbors: int = 64, num_inducing: int = 512, inducing_selection: InducingSelectionName = "farthest", fit_batch_size: int = 8192, leaf_size: int = 32, ard_lengthscales: Tensor | None = None, ski_grid_size: int = 8, ski_padding: float = 0.05, ski_max_grid_points: int = 2048, multires_levels: int = 3, multires_lengthscale_decay: float = 0.55, multires_inducing_growth: float = 1.5) -> GPISBackend:
    common = dict(lengthscale=lengthscale, variance=variance, noise_std=noise_std, observation_noise_std=observation_noise_std, mean_constant=mean_constant, jitter=jitter)
    if backend == "dense_exact":
        return DenseExactGPISBackend.fit(x_train, y_train, **common)
    if backend == "local_exact":
        return LocalExactGPISBackend.fit(x_train, y_train, num_neighbors=num_neighbors, **common)
    if backend == "local_kdtree":
        return KDTreeLocalExactGPISBackend.fit(x_train, y_train, num_neighbors=num_neighbors, leaf_size=leaf_size, **common)
    if backend == "local_faiss":
        return FaissLocalExactGPISBackend.fit(x_train, y_train, num_neighbors=num_neighbors, **common)
    if backend == "inducing_points":
        return InducingPointGPISBackend.fit(x_train, y_train, num_inducing=num_inducing, inducing_selection=inducing_selection, fit_batch_size=fit_batch_size, **common)
    if backend == "ard_inducing_points":
        if ard_lengthscales is None:
            spread = torch.clamp(torch.std(x_train.detach().to(dtype=torch.float64), dim=0), min=1e-3); ard_lengthscales = lengthscale * spread / torch.clamp(spread.mean(), min=1e-6)
        return ARDInducingPointGPISBackend.fit(x_train, y_train, num_inducing=num_inducing, inducing_selection=inducing_selection, fit_batch_size=fit_batch_size, ard_lengthscales=ard_lengthscales, backend_name="ard_inducing_points", **common)
    if backend == "ski_grid":
        grid = make_regular_grid(x_train.detach().to(dtype=torch.float64, device="cpu"), grid_size=ski_grid_size, padding=ski_padding, max_grid_points=ski_max_grid_points)
        return SKIGridGPISBackend.fit(x_train, y_train, inducing_points=grid, num_inducing=grid.shape[0], fit_batch_size=fit_batch_size, backend_name="ski_grid", **common)
    if backend == "multires_inducing":
        return MultiresInducingGPISBackend.fit(x_train, y_train, num_inducing=num_inducing, lengthscale=lengthscale, variance=variance, noise_std=noise_std, observation_noise_std=observation_noise_std, mean_constant=mean_constant, jitter=jitter, fit_batch_size=fit_batch_size, inducing_selection=inducing_selection, multires_levels=multires_levels, multires_lengthscale_decay=multires_lengthscale_decay, multires_inducing_growth=multires_inducing_growth)
    raise ValueError(f"Unknown GPIS backend '{backend}'.")


def load_gpis_backend(path: str | Path) -> tuple[GPISBackend, dict[str, object]]:
    npz = np.load(path, allow_pickle=False); name = read_backend_name(npz); npz.close()
    if name in (None, "dense_exact"):
        return DenseExactGPISBackend.load(path)
    if name in ("local_exact", "local_kdtree", "local_faiss"):
        return LocalExactGPISBackend.load(path)
    if name in ("inducing_points", "ard_inducing_points", "ski_grid", "multires_level"):
        return InducingPointGPISBackend.load(path)
    if name == "multires_inducing":
        return MultiresInducingGPISBackend.load(path)
    raise ValueError(f"Unsupported GPIS backend in {path}: {name!r}.")


def nearest_training_indices(x_query: Tensor, x_train: Tensor, *, num_neighbors: int) -> Tensor:
    if num_neighbors < 1:
        raise ValueError("num_neighbors must be positive.")
    if x_train.shape[0] == 0:
        raise ValueError("At least one training observation is required.")
    return torch.topk(torch.cdist(x_query, x_train), k=min(int(num_neighbors), int(x_train.shape[0])), dim=1, largest=False).indices


def select_inducing_indices(x_train: Tensor, *, num_inducing: int, method: InducingSelectionName = "farthest") -> Tensor:
    if num_inducing < 1:
        raise ValueError("num_inducing must be positive.")
    n = int(x_train.shape[0]); m = min(int(num_inducing), n)
    if method == "first":
        return torch.arange(m, dtype=torch.long, device=x_train.device)
    if method == "uniform":
        return torch.linspace(0, n - 1, steps=m, device=x_train.device).round().to(dtype=torch.long)
    if method != "farthest":
        raise ValueError("inducing_selection must be one of: farthest, uniform, first.")
    selected = torch.empty((m,), dtype=torch.long, device=x_train.device); selected[0] = int(torch.argmin(torch.sum((x_train - x_train.mean(0, keepdim=True)).pow(2), 1)).item())
    mind = torch.sum((x_train - x_train[int(selected[0])]).pow(2), dim=1); mind[selected[0]] = -1
    for i in range(1, m):
        selected[i] = int(torch.argmax(mind).item()); d = torch.sum((x_train - x_train[int(selected[i])]).pow(2), dim=1); mind = torch.minimum(mind, d); mind[selected[: i + 1]] = -1
    return selected


def validate_training_inputs(x_train: Tensor, y_train: Tensor) -> None:
    if x_train.ndim != 2 or y_train.ndim != 1 or x_train.shape[0] != y_train.shape[0] or x_train.shape[0] < 1:
        raise ValueError("Invalid GPIS training inputs.")


def normalize_observation_noise(observation_noise_std: Tensor | None, y_train: Tensor) -> Tensor | None:
    if observation_noise_std is None:
        return None
    noise = observation_noise_std.detach().reshape(-1).to(dtype=y_train.dtype, device=y_train.device)
    if noise.shape != y_train.shape:
        raise ValueError("observation_noise_std must have one value per training observation.")
    return noise


def cholesky_with_jitter(system: Tensor, *, jitter: float, error_prefix: str) -> Tensor:
    chol, info = torch.linalg.cholesky_ex(system + jitter * torch.eye(system.shape[0], dtype=system.dtype, device=system.device))
    if int(info.item()) != 0:
        raise RuntimeError(f"{error_prefix} Cholesky factorization failed at leading minor {int(info.item())}.")
    return chol


def solve_lower_triangular_features(k_xu: Tensor, chol_uu: Tensor) -> Tensor:
    return torch.linalg.solve_triangular(chol_uu, k_xu.T, upper=False).T


def ard_rbf_kernel(x1: Tensor, x2: Tensor, lengthscales: Tensor, variance: float) -> Tensor:
    return variance * torch.exp(-0.5 * torch.cdist(x1 / lengthscales[None, :], x2 / lengthscales[None, :]).pow(2))


def normalize_ard_lengthscales(lengthscales: Tensor, dims: int) -> Tensor:
    values = lengthscales.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    if values.shape != (dims,) or torch.any(values <= 0):
        raise ValueError("ard_lengthscales must contain one positive value per dimension.")
    return values


def make_regular_grid(x_train: Tensor, *, grid_size: int, padding: float, max_grid_points: int) -> Tensor:
    total = int(grid_size) ** int(x_train.shape[1])
    if grid_size < 2 or total > max_grid_points:
        raise ValueError("Invalid SKI grid size.")
    mins = x_train.min(0).values - float(padding); maxs = x_train.max(0).values + float(padding)
    axes = [torch.linspace(float(mins[d]), float(maxs[d]), steps=int(grid_size), dtype=x_train.dtype) for d in range(x_train.shape[1])]
    return torch.stack([axis.reshape(-1) for axis in torch.meshgrid(*axes, indexing="ij")], 1).contiguous()


def read_backend_name(npz: np.lib.npyio.NpzFile) -> str | None:
    return str(npz["backend_name"].item()) if "backend_name" in npz.files else (str(npz["meta_backend"].item()) if "meta_backend" in npz.files else None)


def metadata_from_npz(npz: np.lib.npyio.NpzFile) -> dict[str, object]:
    out: dict[str, object] = {}
    for key in npz.files:
        if key.startswith("meta_"):
            value = npz[key]; out[key.removeprefix("meta_")] = value.item() if value.shape == () else value
    return out
