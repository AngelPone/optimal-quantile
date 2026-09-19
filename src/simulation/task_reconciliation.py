from pytask import task, Product

from simulation.config import (
    data_catalog,
    SAMPLE_SIZE,
    ALPHAs,
    SCENARIOS,
    A,
    WINDOW_S,
    TRAIN_WINDOWS,
    BETA,
    DTYPE,
    DEVICE,
    BLD,
    scenario_alpha_seed,
)
from utils import expanding_window, SkewStudentT
from opt_rec_quantile.model import QOptRec
from forecopy import cstools, cscov
from typing import Annotated, List, Any
import numpy as np
import torch

LR = {"S2": 1e-5, "S3": 5e-6, "S1": 1e-2, "S4": 1e-2}
INIT = {"S2": "shr", "S3": "shr", "S1": "shr", "S4": "shr"}

for scenario in ["S3"]:
    for alpha in ALPHAs:
        seed = scenario_alpha_seed(scenario, alpha, offset=20_000)
        lr = LR[scenario]

        @task
        def task_perform_reconciliation(
            input_data: Annotated[
                List[np.ndarray], data_catalog[f"simulation_{scenario}"]
            ],
            input_base: Annotated[
                List[dict[str, Any]], data_catalog[f"simulation_base_{scenario}"]
            ],
            output: Annotated[tuple[torch.Tensor], Product] = data_catalog[
                f"simulation_rf_{scenario}_{alpha}"
            ],
            scenario: str = scenario,
            alpha: float = alpha,
            seed: int = seed,
        ):
            y = torch.as_tensor(input_data["y"], dtype=DTYPE, device=DEVICE)
            weights = torch.diff(y, dim=0).abs().mean(dim=0)
            windows = expanding_window(y.shape[0], WINDOW_S, 1, y)
            test_windows = windows.collect_test().squeeze(1)

            train_slice = slice(0, TRAIN_WINDOWS)
            true_y = test_windows[train_slice, :]
            train_base = input_base[train_slice]
            mean = torch.stack([window["mean"] for window in train_base]).to(
                dtype=DTYPE, device=DEVICE
            )
            torch.manual_seed(seed)

            xi = torch.stack([window["dist"].xi for window in train_base])
            df = torch.stack([window["dist"].df for window in train_base])
            loc = torch.stack([window["dist"].loc for window in train_base])
            scale = torch.stack([window["dist"].scale for window in train_base])

            # sampling from the base forecast
            def sampling(slice, J=SAMPLE_SIZE):
                samples = (
                    SkewStudentT(xi[slice], df[slice], loc[slice], scale[slice])
                    .sample((J,))
                    .permute((1, 2, 0))
                )
                samples = mean[slice, :, None] + samples
                return samples

            mdl = QOptRec(
                A=A,
                alpha=[alpha],
                beta=BETA,
                optimizer_cls=torch.optim.Adam,
                optimizer_kwargs={"lr": lr},
            )

            params = cstools(A.cpu().numpy())
            W = cscov(params, input_base[TRAIN_WINDOWS]["resid"].T).fit(
                comb=INIT[scenario]
            )
            W = torch.linalg.inv(torch.as_tensor(W, dtype=DTYPE, device=DEVICE))
            G_init = torch.linalg.solve(mdl.S.T @ W @ mdl.S, mdl.S.T @ W)
            train_slice = slice(0, TRAIN_WINDOWS)
            # val_slice = slice(TRAIN_WINDOWS - 100, TRAIN_WINDOWS)
            G, d = mdl.train(
                true_y[train_slice],
                lambda: sampling(train_slice, SAMPLE_SIZE),
                G=G_init,
                max_iter=300,
                weights=weights,
                # y_val=true_y[val_slice],
                # sampling_val=lambda: sampling(val_slice, J=SAMPLE_SIZE * 10),
                log_dir=BLD / "simulation_logs" / f"{scenario}-{alpha}",
            )
            output.save({"model": mdl, "result": (G, d)})
