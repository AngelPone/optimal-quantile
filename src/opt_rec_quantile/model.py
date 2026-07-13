import torch
from torch.utils.data import DataLoader, TensorDataset
import math
from opt_rec_quantile.loss import ApproxPinballLoss, approx_pinball_loss, pinball_loss
from typing import Callable
from torch.optim.lr_scheduler import ReduceLROnPlateau


class QOptRec:

    def __init__(
        self,
        A: torch.Tensor,
        alpha: list[float],
        beta: float,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=None,
    ):
        self.A = A
        self.alpha = alpha
        assert beta > 0
        self.beta = beta
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs

    @property
    def n(self) -> int:
        return self.A.shape[1] + self.A.shape[0]

    @property
    def m(self) -> int:
        return self.A.shape[1]

    @property
    def S(self) -> torch.Tensor:
        return torch.cat(
            [self.A, torch.eye(self.m, dtype=self.A.dtype, device=self.A.device)]
        )

    def _loss(
        self, y: torch.Tensor, y_pred: torch.Tensor, G: torch.Tensor, d: torch.Tensor
    ):
        loss = torch.tensor(0.0)
        S = self.S.to(dtype=G.dtype, device=G.device)
        bias = torch.einsum("mk,nm->nk", d, S)
        with torch.no_grad():
            rf = torch.einsum("tnj,kn->tkj", y_pred, S @ G)
        for idx, alpha in enumerate(self.alpha):
            q = ApproxPinballLoss.apply(S, rf, y_pred, alpha, self.beta)
            y = bias[:, idx][None, :] + q
            loss += approx_pinball_loss(y - q, self.beta, alpha)
        return loss

    def _pinball_loss(
        self, y: torch.Tensor, y_pred: torch.Tensor, G: torch.Tensor, d: torch.Tensor
    ):
        loss = torch.tensor(0.0)
        S = self.S.to(dtype=G.dtype, device=G.device)
        bias = torch.einsum("mk,nm->nk", d, S)
        rf = torch.einsum("tnj,kn->tkj", y_pred, self.S @ G)
        for idx, alpha in enumerate(self.alpha):
            q = ApproxPinballLoss.apply(S, rf, y_pred, alpha, self.beta)
            y = bias[:, idx][None, :] + q
            loss += pinball_loss(y - q, alpha)
        return loss

    @staticmethod
    def _as_training_tensor(value, y: torch.Tensor):
        if isinstance(value, torch.Tensor):
            return value.to(dtype=y.dtype, device=y.device)
        return torch.as_tensor(value, dtype=y.dtype, device=y.device)

    def train(
        self,
        y: torch.Tensor,
        sampling: Callable[[], torch.Tensor],
        G: torch.Tensor | str = "ols",
        generator: torch.Generator | None = None,
        max_iter: int = 200,
        lr_decay: float = 1.0,
        sampling_val: Callable[[], torch.Tensor] | None = None,
        y_val: torch.Tensor | None = None,
        batch_size: int | None = None,
    ):
        assert y.shape[1] == self.n
        if not 0 < lr_decay <= 1:
            raise ValueError("lr_decay must be in (0, 1].")
        if isinstance(G, str):
            if G == "ols":
                G = torch.linalg.solve(self.S.T @ self.S, self.S.T)
                G = torch.as_tensor(G).requires_grad_()
            elif G == "random":
                G = torch.rand(
                    (self.m, self.n),
                    requires_grad=True,
                    dtype=y.dtype,
                    device=y.device,
                    generator=generator,
                )
            else:
                raise ValueError(f"Initialization of using {G} is not supported")
        else:
            assert G.shape == (self.m, self.n), "invalid shape of G"
            G = torch.as_tensor(
                G,
                dtype=y.dtype,
                device=y.device,
            ).requires_grad_()
        d = torch.zeros(
            (self.m, len(self.alpha)),
            requires_grad=True,
            dtype=y.dtype,
            device=y.device,
        )
        optimizer_kwargs = self.optimizer_kwargs or {}
        optimizer = self.optimizer_cls(params=[G, d], **optimizer_kwargs)

        best_pinball_loss = float("inf")
        scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5)
        best_G = G.detach().clone()
        best_d = d.detach().clone()
        self.smooth_loss_history_ = []
        self.pinball_loss_history_ = []

        if sampling_val is not None:
            assert y_val is not None

        eval_y_pred = (
            self._as_training_tensor(sampling_val(), y)
            if sampling_val is not None
            else sampling()
        )
        y_val = y if sampling_val is None else y_val
        for step in range(max_iter):
            if batch_size is None:
                y_pred = self._as_training_tensor(sampling(), y)
                optimizer.zero_grad()
                loss = self._loss(y, y_pred, G, d)
                loss.backward()
                optimizer.step()
                loss_item = loss.detach().item()
            else:
                perm = torch.randperm(y.shape[0], generator=generator)
                loss_item = 0.0
                for start in range(0, y.shape[0], batch_size):
                    batch_indices = perm[start : start + batch_size]
                    batch_y = y[batch_indices]
                    batch_y_pred = sampling(batch_indices)
                    optimizer.zero_grad()
                    loss = self._loss(batch_y, batch_y_pred, G, d)
                    loss_item += loss.detach().item()
                    loss.backward()
                    optimizer.step()
                loss_item = loss_item / len(range(0, y.shape[0], batch_size))
            param_groups = getattr(optimizer, "param_groups", None)
            if param_groups is not None:
                if lr_decay < 1:
                    for param_group in param_groups:
                        param_group["lr"] *= lr_decay

            with torch.no_grad():
                current_pinball_loss = self._pinball_loss(
                    y_val, eval_y_pred, G, d
                ).item()
                self.smooth_loss_history_.append(loss_item)
                self.pinball_loss_history_.append(current_pinball_loss)
                if current_pinball_loss < best_pinball_loss:
                    best_pinball_loss = current_pinball_loss
                    best_G = G.detach().clone()
                    best_d = d.detach().clone()
            scheduler.step(current_pinball_loss)

            print(
                f"Step {step}: loss: {loss.detach():.4f}, "
                f"pinball_loss: {current_pinball_loss:.4f}"
            )

        self.best_pinball_loss_ = best_pinball_loss
        self.final_G_ = G.detach().clone()
        self.final_d_ = d.detach().clone()
        return best_G, best_d
