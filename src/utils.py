from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from torch.distributions import StudentT
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
        return self.T - self.S - self.h + 1

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
        return slice(0, split), slice(split, split + self.h)


def expanding_window(
    T: int,
    S: int,
    h: int,
    dataset: Sequence[Any] | None = None,
) -> ExpandingWindowIterator:
    """Create an expanding-window iterator."""
    return ExpandingWindowIterator(T=T, S=S, h=h, dataset=dataset)


@dataclass(slots=["xi", "df", "loc", "scale"])
class SkewStudentT:

    xi: torch.Tensor | float
    df: float
    loc: torch.Tensor | float = 0.0
    scale: torch.Tensor | float = 1.0

    def __post_init__(self):
        self.xi = torch.as_tensor(self.xi)
        self.loc = torch.as_tensor(self.loc, dtype=self.xi.dtype)
        self.scale = torch.as_tensor(self.scale, dtype=self.xi.dtype)
        if torch.any(self.scale < 0):
            raise ValueError("scale must be positive")
        if self.df <= 2:
            raise ValueError("df must be > 2 for finite variance.")
        self.df = torch.as_tensor(self.df, dtype=self.xi.dtype)

    def prob(self, x: torch.Tensor):
        return torch.exp(self.log_prob(x))

    def sample(self, size: tuple, generator: torch.Generator | None = None):
        unif = torch.rand(
            size,
            dtype=self.loc.dtype,
            device=self.loc.device,
            generator=generator,
        )
        eps = max(torch.finfo(unif.dtype).eps, 1e-12)
        unif = unif.clamp(eps, 1 - eps)
        return self.q(unif)

    def moments(self):
        xi = self.xi
        df = self.df
        half = torch.as_tensor(0.5, dtype=xi.dtype)
        M1 = 2 * torch.sqrt(df) / ((df - 1) * torch_beta(half, df / 2))
        M2 = df / (df - 2)
        mu_xi = M1 * (xi - 1 / xi)
        var_xi = (M2 - M1**2) * (xi**2 + xi ** (-2)) + 2 * M1**2 - M2
        sigma_xi = torch.sqrt(var_xi)
        return mu_xi, sigma_xi

    def log_prob(
        self,
        x: torch.Tensor,
    ):
        df = self.df
        z = (x - self.loc) / self.scale
        base_dist = StudentT(df)
        mu_xi, sigma_xi = self.moments()
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

    def q(self, x: torch.Tensor):
        mu_xi, sigma_xi = self.moments()
        arg = torch.where(
            x < 1 / (1 + self.xi**2),
            stdtrit(self.df, x * (1 + self.xi**2) / 2) / self.xi,
            self.xi
            * stdtrit(
                self.df, (x * (1 + self.xi**2) + self.xi**2 - 1) / (2 * self.xi**2)
            ),
        )
        return self.loc + self.scale * (arg - mu_xi) / sigma_xi


def torch_beta(x, y):
    """
    Computes the Beta function element-wise: B(x, y) = Gamma(x)*Gamma(y) / Gamma(x+y)
    Supports autograd backpropagation out of the box.
    """
    # Compute in log-space to ensure numerical stability
    log_beta = torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)
    return torch.exp(log_beta)


def mle_estimation_skewed_dist(
    samples: np.ndarray, df: float, trace: bool = False, max_iter: int = 10
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
        dist = SkewStudentT(xi, df)
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
                mu_xi, sigma_xi = SkewStudentT.moments(xi.detach(), df)
                print(f"NLL: {current_loss:.6f}")
                print(f"xi: {xi.item():.6f}")
                print(f"mu_xi: {mu_xi.item():.6f}")
                print(f"sigma_xi: {sigma_xi.item():.6f}")

    xi = torch.exp(eta.detach())
    # stablize the estimation of xi
    if torch.all(xi > 3):
        xi = torch.as_tensor(3)

    return SkewStudentT(xi, df, x_mean, x_sd)
