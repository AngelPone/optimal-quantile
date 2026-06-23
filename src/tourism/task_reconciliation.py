from pytask import task, Product

from tourism.config import (
    A,
    ALPHAs,
    SAMPLE_SIZE,
    TEST_WINDOWS,
    TRAIN_WINDOWS,
    WINDOW_S,
    data_catalog,
    tourism_alpha_seed,
)
from utils import expanding_window
from forecopy import cscov, cstools
from opt_rec_quantile.model import QOptRec
from typing import Annotated, Any
import numpy as np
import torch

for alpha in ALPHAs:
    seed = tourism_alpha_seed(alpha, offset=10_000)

    @task
    def task_perform_reconciliation(
        input_data: Annotated[np.ndarray, data_catalog["tourism"]],
        input_base: Annotated[list[dict[str, Any]], data_catalog["tourism_base"]],
        output: Annotated[dict, Product] = data_catalog[f"tourism_rf_{alpha}_skewt"],
        output_normal: Annotated[dict, Product] = data_catalog[
            f"tourism_rf_{alpha}_normal"
        ],
        alpha: float = alpha,
        seed: int = seed,
    ):
        y = torch.as_tensor(input_data, dtype=torch.float64)
        windows = expanding_window(y.shape[0], WINDOW_S, 1, y)
        test_windows = windows.collect_test().squeeze(1)
        expected_windows = TRAIN_WINDOWS + TEST_WINDOWS
        if len(input_base) != test_windows.shape[0]:
            raise ValueError(
                "Base forecast windows and true-y windows are not aligned: "
                f"{len(input_base)} != {test_windows.shape[0]}."
            )
        if len(input_base) != expected_windows:
            raise ValueError(
                "Configured train/test split does not match available windows: "
                f"{TRAIN_WINDOWS} + {TEST_WINDOWS} != {len(input_base)}."
            )

        train_slice = slice(0, TRAIN_WINDOWS)
        true_y = test_windows[train_slice, :]
        train_base = input_base[train_slice]
        mean = torch.stack([window["mean"] for window in train_base])
        generator = torch.Generator(device=true_y.device)
        generator.manual_seed(seed)

        def sampling_skewt():
            samples = torch.stack(
                [
                    torch.stack(
                        [
                            series.sample(SAMPLE_SIZE, generator=generator)
                            for series in window["skewt"]
                        ]
                    )
                    for window in train_base
                ]
            )
            samples = mean[:, :, None] + samples
            return samples

        def sampling_normal():
            samples = torch.stack(
                [
                    torch.stack(
                        [
                            series.sample(SAMPLE_SIZE, generator=generator)
                            for series in window["normal"]
                        ]
                    )
                    for window in train_base
                ]
            )
            samples = mean[:, :, None] + samples
            return samples

        mdl = QOptRec(
            A=A,
            alpha=alpha,
            beta=20.0,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.01},
        )

        # params = cstools(A.numpy())
        # W = cscov(params, train_base[0]["resid"].T).fit(comb="shr")
        # W = torch.linalg.inv(torch.as_tensor(W, dtype=torch.float64))
        # G_init = torch.linalg.solve(mdl.S.T @ W @ mdl.S, mdl.S.T @ W)
        G, d = mdl.train(
            true_y,
            sampling_skewt,
            generator=generator,
            max_iter=500,
            lr_decay=1,
        )

        output.save({"model": mdl, "result": (G, d)})

        mdl2 = QOptRec(
            A=A,
            alpha=alpha,
            beta=20.0,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.01},
        )
        G2, d2 = mdl2.train(
            true_y, sampling_normal, generator=generator, max_iter=500, lr_decay=1
        )
        output_normal.save({"model": mdl2, "result": (G2, d2)})
