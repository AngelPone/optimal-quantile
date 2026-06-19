from pytask import task, Product

from simulation.config import (
    data_catalog,
    SAMPLE_SIZE,
    ALPHAs,
    SCENARIOS,
    A,
    WINDOW_S,
    TRAIN_WINDOWS,
    TEST_WINDOWS,
    scenario_alpha_seed,
)
from utils import expanding_window
from opt_rec_quantile.model import QOptRec
from typing import Annotated, List, Any
from torch.optim import Adam
import numpy as np
import torch

for scenario in SCENARIOS:
    for alpha in ALPHAs:
        seed = scenario_alpha_seed(scenario, alpha, offset=10_000)

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
            alpha: float = alpha,
            seed: int = seed,
        ):
            y = torch.as_tensor(input_data[0], dtype=torch.float64)
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

            # sampling from the base forecast
            def sampling():
                samples = torch.stack(
                    [
                        torch.stack(
                            [
                                series.sample(SAMPLE_SIZE, generator=generator)
                                for series in window["dist"]
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
                optimizer_cls=Adam,
                optimizer_kwargs={"lr": 1e-2},
            )
            G, d = mdl.train(true_y, sampling, generator=generator, max_iter=200)
            output.save({"model": mdl, "result": (G, d)})
