from pytask import task, Product

from tourism.config import (
    A,
    ALPHAs,
    SAMPLE_SIZE,
    TEST_WINDOWS,
    DEVICE,
    DTYPE,
    data_catalog,
    TRAIN_WINDOWS,
    BETA,
    BLD,
)
from utils import SkewStudentT
from opt_rec_quantile.model import QOptRec
from typing import Annotated, Any
import torch
import numpy as np
from torch.distributions import Normal
from forecopy import cscov, cstools

lr_set = {
    ("0.95", "skew"): 1e-5,
    ("0.95", "normal"): 1e-5,
}
MAX_ITER = 500
INIT = "shr"

for alpha in ALPHAs:

    for dist_idx, dist in enumerate(["normal", "skew"]):
        seed = 20260904 + int(alpha * 10000) + dist_idx
        LR = lr_set.get((str(alpha), dist), 1e-6)

        @task
        def task_perform_reconciliation(
            input_data: Annotated[np.ndarray, data_catalog["tourism"]],
            input_base: Annotated[list[dict[str, Any]], data_catalog["base"]],
            output: Annotated[dict, Product] = data_catalog[f"rf_{alpha}_{dist}"],
            alpha: float = [alpha],
            learning_rate: float = LR,
            dist: str = dist,
            seed: int = seed,
        ):
            y = input_base["true_y"][:-TEST_WINDOWS, :]
            train_slice = slice(y.shape[0] - 36, y.shape[0])
            # valid_slice = slice(TRAIN_WINDOWS, y.shape[0])
            point_f = input_base["mean"][:-TEST_WINDOWS, :]
            torch.manual_seed(seed)

            loc, scale, xi, df = (
                input_base["dist"]["mean"],
                input_base["dist"]["std"],
                input_base["dist"]["xi"],
                input_base["dist"]["df"],
            )

            def sampling(dist: str, indices, J: int = SAMPLE_SIZE):
                if dist == "skew":
                    smps = (
                        SkewStudentT(
                            xi[indices], df[indices], loc[indices], scale[indices]
                        )
                        .sample((J,))
                        .permute((1, 2, 0))
                    )
                elif dist == "normal":
                    smps = (
                        Normal(loc[indices], scale[indices])
                        .sample((J,))
                        .permute((1, 2, 0))
                    )
                else:
                    raise ValueError("Invalid distribution")
                samples = point_f[indices, :, None] + smps
                return samples

            mdl = QOptRec(
                A=A,
                alpha=alpha,
                beta=BETA,
                optimizer_cls=torch.optim.Adam,
                optimizer_kwargs={"lr": learning_rate},
            )

            params = cstools(A.cpu().numpy())

            weights = np.abs(np.diff(input_data, axis=0)).mean(axis=0)
            weights = torch.as_tensor(weights, device=DEVICE, dtype=DTYPE)
            W = cscov(params, input_base["resids"][-TEST_WINDOWS]).fit(comb=INIT)
            W = torch.as_tensor(np.linalg.inv(W), dtype=DTYPE, device=DEVICE)
            G_init = torch.linalg.solve(mdl.S.T @ W @ mdl.S, mdl.S.T @ W)
            G, d = mdl.train(
                y[train_slice],
                lambda: sampling(dist, train_slice),
                weights=weights,
                G=G_init,
                max_iter=MAX_ITER,
                # sampling_val=lambda: sampling(dist, valid_slice, SAMPLE_SIZE * 10),
                # y_val=y[valid_slice],
                log_dir=BLD / "tourism_logs" / f"alpha{int(alpha[0]*1000)}-dist{dist}",
            )
            output.save({"model": mdl, "result": (G, d)})


if __name__ == "__main__":
    task_perform_reconciliation(
        data_catalog["tourism"].load(),
        data_catalog["base"].load(),
        alpha=[0.2],
        dist="normal",
    )
