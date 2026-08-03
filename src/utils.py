from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import math
from torch.distributions import Normal, StudentT
from torch.optim import LBFGS
from scipy.special import stdtrit


@dataclass(frozen=True)
class ExpandingWindowIterator:
    """Iterate over expanding-window train/test splits.

    For a series of length ``T``, initial input length ``S``, and forecast
    horizon ``h``, the ``idx``-th split is:

    - train: ``[0, S + idx)``
    - test: ``[S + idx, S + idx + h)``

    If ``dataset`` is provided, ``__getitem__`` and iteration return sliced
    ``(trainset, testset)`` pairs. Otherwise they return the corresponding
    ``(train_slice, test_slice)`` pair.
    """

    T: int
    S: int
    h: int
    dataset: Any | None = None
    var_h: bool = False

    def __post_init__(self) -> None:
        if self.dataset is not None:
            dataset_length = len(self.dataset)
            if dataset_length != self.T:
                raise ValueError(
                    f"dataset length ({dataset_length}) must match T ({self.T})."
                )

        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.S <= 0:
            raise ValueError("S must be positive.")
        if self.h <= 0:
            raise ValueError("h must be positive.")
        if self.S + self.h > self.T:
            raise ValueError("S + h must be less than or equal to T.")

    def __len__(self) -> int:
        if not self.var_h:
            return self.T - self.S - self.h + 1
        else:
            return self.T - self.S

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        if not isinstance(idx, int):
            raise TypeError("idx must be an integer.")

        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("expanding-window index out of range.")

        train_slice, test_slice = self.get_slices(idx)
        if self.dataset is None:
            return train_slice, test_slice
        return self.dataset[train_slice], self.dataset[test_slice]

    def collect_test(self) -> Any:
        """Return test observations for every forecast origin."""
        test_slices = [self.get_slices(idx)[1] for idx in range(len(self))]
        if self.dataset is None:
            return test_slices

        test_windows = [self.dataset[test_slice] for test_slice in test_slices]
        first_window = test_windows[0]
        if torch.is_tensor(first_window):
            return torch.stack(test_windows)
        if isinstance(first_window, np.ndarray):
            return np.stack(test_windows)
        return test_windows

    def get_slices(self, idx: int) -> tuple[slice, slice]:
        """Return the train/test slices for ``idx`` without slicing data."""
        if not isinstance(idx, int):
            raise TypeError("idx must be an integer.")

        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("expanding-window index out of range.")

        split = self.S + idx
        return slice(0, split), slice(split, min(split + self.h, self.T))


class FixedWindowIterator(ExpandingWindowIterator):
    def get_slices(self, idx):
        if not isinstance(idx, int):
            raise TypeError("idx must be an integer.")

        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("expanding-window index out of range.")
        split = self.S + idx
        return slice(idx, split), slice(split, split + self.h)


def expanding_window(
    T: int,
    S: int,
    h: int,
    dataset: Sequence[Any] | None = None,
) -> ExpandingWindowIterator:
    """Create an expanding-window iterator."""
    return ExpandingWindowIterator(T=T, S=S, h=h, dataset=dataset)


@dataclass(slots=["xi", "loc", "scale"])
class SkewNormal:

    xi: torch.Tensor | float
    loc: torch.Tensor | float = 0.0
    scale: torch.Tensor | float = 1.0

    def __post_init__(self):
        self.xi = torch.as_tensor(self.xi)
        self.loc = torch.as_tensor(self.loc, dtype=self.xi.dtype)
        self.scale = torch.as_tensor(self.scale, dtype=self.xi.dtype)
        if torch.any(self.scale < 0):
            raise ValueError("scale must be positive")

    def prob(self, x: torch.Tensor):
        return torch.exp(self.log_prob(x))

    def sample(
        self, size: torch.Size = torch.Size(), generator: torch.Generator | None = None
    ):
        size = torch.Size(size) + self.loc.shape
        unif = torch.rand(
            size,
            dtype=self.loc.dtype,
            device=self.loc.device,
            generator=generator,
        )
        eps = max(torch.finfo(unif.dtype).eps, 1e-12)
        unif = unif.clamp(eps, 1 - eps)
        return self.q(unif)

    @property
    def moments(self):
        xi = self.xi
        M1 = torch.sqrt(
            torch.as_tensor(2.0 / torch.pi, dtype=xi.dtype, device=xi.device)
        )
        M2 = torch.as_tensor(1.0, dtype=xi.dtype, device=xi.device)
        mu_xi = M1 * (xi - 1 / xi)
        var_xi = (M2 - M1**2) * (xi**2 + xi ** (-2)) + 2 * M1**2 - M2
        sigma_xi = torch.sqrt(var_xi)
        return mu_xi, sigma_xi

    def log_prob(
        self,
        x: torch.Tensor,
    ):
        z = (x - self.loc) / self.scale
        base_dist = Normal(loc=0.0, scale=1.0)
        mu_xi, sigma_xi = self.moments
        s = sigma_xi * z + mu_xi
        # Piecewise argument of the underlying symmetric Student-t density
        arg = torch.where(s < 0, self.xi * s, s / self.xi)
        log_norm = torch.log(
            torch.as_tensor(2.0, dtype=x.dtype, device=x.device)
        ) - torch.log(self.xi + 1 / self.xi)
        log_density = (
            -torch.log(self.scale)
            + torch.log(sigma_xi)
            + log_norm
            + base_dist.log_prob(arg)
        )
        return log_density

    def q(self, p: torch.Tensor):
        """
        Quantile function.

        p should be in (0, 1).
        """
        p = torch.as_tensor(p, dtype=self.loc.dtype, device=self.loc.device)

        eps = max(torch.finfo(p.dtype).eps, 1e-12)
        p = p.clamp(eps, 1 - eps)

        mu_xi, sigma_xi = self.moments

        threshold = 1 / (1 + self.xi**2)
        left_prob = p * (1 + self.xi**2) / 2
        right_prob = (p * (1 + self.xi**2) + self.xi**2 - 1) / (2 * self.xi**2)
        # Standard normal inverse CDF:
        # Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1)
        sqrt2 = torch.sqrt(torch.as_tensor(2.0, dtype=p.dtype, device=p.device))
        is_left = p < threshold
        branch_prob = torch.where(is_left, left_prob, right_prob)
        branch_prob = branch_prob.clamp(eps, 1 - eps)

        z = sqrt2 * torch.erfinv(2 * branch_prob - 1)
        arg = torch.where(is_left, z / self.xi, self.xi * z)

        return self.loc + self.scale * (arg - mu_xi) / sigma_xi


@dataclass(slots=["xi", "df", "loc", "scale"])
class SkewStudentT:

    xi: torch.Tensor | float
    df: torch.Tensor | float
    loc: torch.Tensor | float = 0.0
    scale: torch.Tensor | float = 1.0

    def __post_init__(self):
        self.xi = torch.as_tensor(self.xi)
        self.df = torch.as_tensor(self.df, dtype=self.xi.dtype, device=self.xi.device)
        self.loc = torch.as_tensor(self.loc, dtype=self.xi.dtype, device=self.xi.device)
        self.scale = torch.as_tensor(
            self.scale, dtype=self.xi.dtype, device=self.xi.device
        )
        if torch.any(self.xi <= 0):
            raise ValueError("xi must be positive")
        if torch.any(self.df <= 2):
            raise ValueError("df must be > 2 for finite variance")
        if torch.any(self.scale <= 0):
            raise ValueError("scale must be positive")

    def prob(self, x: torch.Tensor | float) -> torch.Tensor:
        return torch.exp(self.log_prob(x))

    def sample(
        self,
        size: int | tuple[int, ...] = (),
    ) -> torch.Tensor:
        if isinstance(size, int):
            sample_shape = (size,)
        else:
            sample_shape = tuple(size)

        magnitude = StudentT(self.df).sample(torch.Size(sample_shape)).abs()
        threshold = 1 / (1 + self.xi.square())
        is_left = torch.rand_like(magnitude) < threshold
        skew_t = torch.where(
            is_left,
            -magnitude / self.xi,
            magnitude * self.xi,
        )
        mu_xi, sigma_xi = self.moments
        return self.loc + self.scale * (skew_t - mu_xi) / sigma_xi

    @property
    def moments(self):
        xi = self.xi
        df = self.df
        half = torch.as_tensor(0.5, dtype=xi.dtype, device=xi.device)
        log_m1 = (
            math.log(2.0)
            + 0.5 * torch.log(df)
            - torch.log(df - 1)
            - torch.lgamma(half)
            - torch.lgamma(df / 2)
            + torch.lgamma((df + 1) / 2)
        )
        M1 = torch.exp(log_m1)
        M2 = df / (df - 2)
        mu_xi = M1 * (xi - 1 / xi)
        var_xi = (M2 - M1**2) * (xi**2 + xi ** (-2)) + 2 * M1**2 - M2
        var_xi = torch.clamp_min(
            var_xi,
            torch.finfo(xi.dtype).tiny,
        )
        return mu_xi, torch.sqrt(var_xi)

    def log_prob(
        self,
        x: torch.Tensor | float,
    ) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.xi.dtype, device=self.xi.device)
        df = self.df
        z = (x - self.loc) / self.scale
        mu_xi, sigma_xi = self.moments
        s = sigma_xi * z + mu_xi
        arg = torch.where(s < 0, self.xi * s, s / self.xi)
        log_norm = math.log(2.0) - torch.log(self.xi + 1 / self.xi)
        log_density = (
            -torch.log(self.scale)
            + torch.log(sigma_xi)
            + log_norm
            + StudentT(df).log_prob(arg)
        )
        return log_density

    def q(self, p: torch.Tensor | float) -> torch.Tensor:
        """Return quantiles, using SciPy only for the explicit inverse CDF."""
        p = torch.as_tensor(p, dtype=self.xi.dtype, device=self.xi.device)
        eps = max(torch.finfo(p.dtype).eps, 1e-12)
        p = p.clamp(eps, 1 - eps)
        p, xi, df = torch.broadcast_tensors(p, self.xi, self.df)
        threshold = 1 / (1 + xi.square())
        is_left = p < threshold
        arg = torch.empty_like(p)

        if torch.any(is_left):
            left_prob = p[is_left] * (1 + xi[is_left].square()) / 2
            left_prob = left_prob.clamp(eps, 1 - eps)
            left_quantile = stdtrit(
                df[is_left].detach().cpu().numpy(),
                left_prob.detach().cpu().numpy(),
            )
            arg[is_left] = (
                torch.as_tensor(left_quantile, dtype=p.dtype, device=p.device)
                / xi[is_left]
            )

        is_right = ~is_left
        if torch.any(is_right):
            right_prob = (
                p[is_right] * (1 + xi[is_right].square()) + xi[is_right].square() - 1
            ) / (2 * xi[is_right].square())
            right_prob = right_prob.clamp(eps, 1 - eps)
            right_quantile = stdtrit(
                df[is_right].detach().cpu().numpy(),
                right_prob.detach().cpu().numpy(),
            )
            arg[is_right] = xi[is_right] * torch.as_tensor(
                right_quantile, dtype=p.dtype, device=p.device
            )

        mu_xi, sigma_xi = self.moments
        return self.loc + self.scale * (arg - mu_xi) / sigma_xi


def mle_estimation_skewed_dist(samples: np.ndarray, max_iter: int = 20):
    dtype = torch.float64

    x = torch.as_tensor(samples, dtype=dtype)

    x_mean = x.mean()
    x_sd = x.std(correction=0)
    z = (x - x_mean) / x_sd
    raw_xi = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    raw_df = torch.tensor(0.0, dtype=dtype, requires_grad=True)

    optimizer = LBFGS(
        params=[raw_xi, raw_df],
        lr=0.5,
        line_search_fn="strong_wolfe",
    )

    DF_MIN = torch.tensor(2.05, dtype=dtype, device=x.device)
    DF_MAX = torch.tensor(200.0, dtype=dtype, device=x.device)
    XI_MAX = torch.tensor(10.0, dtype=dtype, device=x.device)

    def nll():
        xi = torch.exp(torch.log(XI_MAX) * torch.tanh(raw_xi))
        df = DF_MIN + (DF_MAX - DF_MIN) * torch.sigmoid(raw_df)
        dist = SkewStudentT(xi, df)
        return dist.log_prob(z).negative().sum()

    def closure():
        optimizer.zero_grad()
        loss = nll()
        loss.backward()
        return loss

    for _ in range(max_iter):
        optimizer.step(closure)
    xi = torch.exp(torch.log(XI_MAX) * torch.tanh(raw_xi.detach()))
    df = DF_MIN + (DF_MAX - DF_MIN) * torch.sigmoid(raw_df.detach())

    return SkewStudentT(xi.detach(), df.detach(), x_mean, x_sd)


def mle_estimation_skewed_normal(
    samples: np.ndarray, trace: bool = False, max_iter: int = 10
):
    dtype = torch.float64

    x = torch.as_tensor(samples, dtype=dtype)

    # plug-in standardization of the data
    x_mean = x.mean()
    x_sd = x.std(correction=0)
    z = (x - x_mean) / x_sd
    eta = torch.tensor(0.0, dtype=dtype, requires_grad=True)

    optimizer = LBFGS(
        params=[eta],
        lr=0.5,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def nll():
        xi = torch.exp(eta)
        dist = SkewNormal(xi)
        return dist.log_prob(z).negative().sum()

    def closure():
        optimizer.zero_grad()
        loss = nll()
        loss.backward()
        return loss

    for _ in range(max_iter):

        optimizer.step(closure)

        if trace:
            with torch.no_grad():
                xi = torch.exp(eta)
                current_loss = nll().item()
                mu_xi, sigma_xi = SkewNormal.moments(xi.detach(), df)
                print(f"NLL: {current_loss:.6f}")
                print(f"xi: {xi.item():.6f}")
                print(f"mu_xi: {mu_xi.item():.6f}")
                print(f"sigma_xi: {sigma_xi.item():.6f}")

    xi = torch.exp(eta.detach())
    # stablize the estimation of xi
    if torch.all(xi > 3):
        xi = torch.as_tensor(3.0, dtype=dtype)

    return SkewNormal(xi, x_mean, x_sd)
