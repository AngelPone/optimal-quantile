import torch
import math
from opt_rec_quantile.loss import ApproxPinballLoss, approx_pinball_loss, pinball_loss
from typing import Callable
from torch.optim.lr_scheduler import ReduceLROnPlateau


class QOptRec:

    def __init__(
        self,
        A: torch.Tensor,
        alpha: float,
        beta: float,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=None,
    ):
        self.A = A
        assert 0 < alpha < 1
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
        q = self._quantile(y_pred, G, d)
        loss = approx_pinball_loss(y - q, self.beta, self.alpha)
        return loss

    def _quantile(self, y_pred: torch.Tensor, G: torch.Tensor, d: torch.Tensor):
        S = self.S.to(dtype=G.dtype, device=G.device)
        q = ApproxPinballLoss.apply(S, G, y_pred, self.alpha, self.beta)
        bias = d @ S.T
        return bias[None, :] + q

    def _pinball_loss(
        self, y: torch.Tensor, y_pred: torch.Tensor, G: torch.Tensor, d: torch.Tensor
    ):
        q = self._quantile(y_pred, G, d)
        return pinball_loss(y - q, self.alpha)

    @staticmethod
    def _as_training_tensor(value, y: torch.Tensor):
        if isinstance(value, torch.Tensor):
            return value.to(dtype=y.dtype, device=y.device)
        return torch.as_tensor(value, dtype=y.dtype, device=y.device)

    def train(
        self,
        y: torch.Tensor,
        sample: Callable[[], torch.Tensor],
        G: torch.Tensor | str = "ols",
        generator: torch.Generator | None = None,
        max_iter: int = 200,
        lr_decay: float = 1.0,
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
        d = torch.zeros((self.m), requires_grad=True, dtype=y.dtype, device=y.device)
        optimizer_kwargs = self.optimizer_kwargs or {}
        optimizer = self.optimizer_cls(params=[G, d], **optimizer_kwargs)

        best_pinball_loss = float("inf")
        scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5)
        best_G = G.detach().clone()
        best_d = d.detach().clone()
        self.smooth_loss_history_ = []
        self.pinball_loss_history_ = []

        eval_y_pred = self._as_training_tensor(sample(10000), y)
        for step in range(max_iter):
            optimizer.zero_grad()
            y_pred = self._as_training_tensor(sample(), y)
            loss = self._loss(y, y_pred, G, d)
            loss.backward()
            optimizer.step()
            param_groups = getattr(optimizer, "param_groups", None)
            if param_groups is not None:
                if lr_decay < 1:
                    for param_group in param_groups:
                        param_group["lr"] *= lr_decay

            with torch.no_grad():
                current_pinball_loss = self._pinball_loss(y, eval_y_pred, G, d).item()
                self.smooth_loss_history_.append(loss.detach().item())
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
