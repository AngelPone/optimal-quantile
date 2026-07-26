import torch
from opt_rec_quantile.loss import ApproxPinballLoss, approx_pinball_loss, pinball_loss
from typing import Callable
from torch.optim.lr_scheduler import ExponentialLR, ConstantLR, SequentialLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


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
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        G: torch.Tensor,
        d: torch.Tensor,
        weights: torch.Tensor | None,
    ):
        loss = torch.tensor(0.0, device=G.device, dtype=G.dtype)
        S = self.S.to(dtype=G.dtype, device=G.device)
        bias = torch.einsum("mk,nm->nk", d, S)
        with torch.no_grad():
            rf = torch.einsum("tnj,kn->tkj", y_pred, S @ G)
        for idx, alpha in enumerate(self.alpha):
            q = bias[:, idx][None, :] + ApproxPinballLoss.apply(
                S, G, rf, y_pred, alpha, self.beta
            )
            loss += approx_pinball_loss(y - q, self.beta, alpha, weights)
        return loss

    def _pinball_loss(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        G: torch.Tensor,
        d: torch.Tensor,
        weights: torch.Tensor | None = None,
    ):
        loss = torch.tensor(0.0, device=G.device, dtype=G.dtype)
        S = self.S.to(dtype=G.dtype, device=G.device)
        bias = torch.einsum("mk,nm->nk", d, S)
        rf = torch.einsum("tnj,kn->tkj", y_pred, self.S @ G)
        for idx, alpha in enumerate(self.alpha):
            q = bias[:, idx][None, :] + ApproxPinballLoss.apply(
                S, G, rf, y_pred, alpha, self.beta
            )
            loss += pinball_loss(y - q, alpha, weights)
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
        weights: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        max_iter: int = 200,
        lr_decay: float = 1.0,
        sampling_val: Callable[[], torch.Tensor] | None = None,
        y_val: torch.Tensor | None = None,
        batch_size: int | None = None,
        log_dir: Path | None = None,
    ):
        assert y.shape[1] == self.n
        if not 0 < lr_decay <= 1:
            raise ValueError("lr_decay must be in (0, 1].")

        writer = SummaryWriter(log_dir)

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
        optimizer = self.optimizer_cls(
            [
                {"params": (G,), "lr": optimizer_kwargs["lr"]},
                {"params": (d,), "lr": optimizer_kwargs["lr"] * 1e4},
            ]
        )

        best_pinball_loss = float("inf")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[optimizer_kwargs["lr"] * 10, optimizer_kwargs["lr"] * 1e5],
            steps_per_epoch=1,
            epochs=max_iter - 20,
            div_factor=10,
            final_div_factor=1e2,
        )

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
                loss = self._loss(y, y_pred, G, d, weights)
                loss.backward()
                optimizer.step()
                loss_item = loss.detach().item()
                grad_norm_G = torch.nn.utils.clip_grad_norm_(
                    G,
                    max_norm=float("inf"),
                )
                grad_norm_d = torch.nn.utils.clip_grad_norm_(
                    d,
                    max_norm=float("inf"),
                )
            else:
                perm = torch.randperm(y.shape[0], generator=generator)
                loss_item = 0.0
                train_num_samples = 0
                for start in range(0, y.shape[0], batch_size):
                    batch_indices = perm[start : start + batch_size]
                    batch_y = y[batch_indices]
                    batch_y_pred = sampling(batch_indices)
                    optimizer.zero_grad()
                    loss = self._loss(batch_y, batch_y_pred, G, d, weights)
                    loss.backward()
                    optimizer.step()
                    bs = batch_y.shape[0]
                    loss_item += loss.detach().item() * bs
                    train_num_samples += bs
                loss_item = loss_item / train_num_samples

            with torch.no_grad():
                val_pl = self._pinball_loss(y_val, eval_y_pred, G, d, weights).item()
                self.smooth_loss_history_.append(loss_item)
                self.pinball_loss_history_.append(val_pl)
                if val_pl < best_pinball_loss:
                    best_pinball_loss = val_pl
                    best_G = G.detach().clone()
                    best_d = d.detach().clone()
            if step > 19:
                scheduler.step()
                writer.add_scalar(
                    "Debug/Learning_rate_G", scheduler.get_last_lr()[0], step
                )
                writer.add_scalar(
                    "Debug/Learning_rate_d", scheduler.get_last_lr()[1], step
                )
            else:
                writer.add_scalar("Debug/Learning_rate_G", optimizer_kwargs["lr"], step)
                writer.add_scalar(
                    "Debug/Learning_rate_d", optimizer_kwargs["lr"] * 1e4, step
                )

            writer.add_scalar("Loss/train", loss_item, step)
            writer.add_scalar("Loss/Validation", val_pl, step)
            writer.add_scalar("Debug/grad_norm_G", grad_norm_G.item(), step)
            writer.add_scalar("Debug/grad_norm_d", grad_norm_d.item(), step)

        self.best_pinball_loss_ = best_pinball_loss
        self.final_G_ = G.detach().clone()
        self.final_d_ = d.detach().clone()

        writer.close()
        return best_G, best_d
