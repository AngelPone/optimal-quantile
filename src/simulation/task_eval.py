import torch
from pytask import task, Product
from simulation.config import (
    ALPHAs,
    SCENARIOS,
    data_catalog,
    WINDOW_S,
    TEST_WINDOWS,
    BLD,
)
from utils import expanding_window
from typing import Annotated
from opt_rec_quantile.loss import pinball_loss
import pandas as pd
from pathlib import Path

for scenario in SCENARIOS:

    rf_samples = [
        data_catalog[f"simulation_samples_{scenario}_{alpha}"] for alpha in ALPHAs
    ]

    @task
    def task_evaluate(
        input_data: Annotated[dict, data_catalog[f"simulation_{scenario}"]],
        input: Annotated[dict, rf_samples],
        output: Annotated[Path, Product] = BLD
        / "tables"
        / f"simulation_acc_{scenario}.tex",
        scenario: str = scenario,
    ):
        y = input_data[0]
        true_y = expanding_window(y.shape[0], WINDOW_S, 1, y).collect_test()[
            -TEST_WINDOWS:, :
        ]
        true_y = torch.as_tensor(true_y, dtype=torch.float64)

        output_dict = {"method": [], "loss": [], "alpha": []}
        for alpha, rf_samples in zip(ALPHAs, input):
            for method, samples in rf_samples.items():
                q = torch.quantile(samples, alpha, dim=2)
                loss = pinball_loss(true_y - q, alpha=alpha).detach().item()
                output_dict["method"].append(method)
                output_dict["alpha"].append(alpha)
                output_dict["loss"].append(loss)
        df = pd.DataFrame(output_dict)
        df = df.pivot(index="method", columns="alpha", values="loss")
        df.columns = [f"{i:.2f}" for i in df.columns]
        output.write_text(df.to_latex(float_format="%.2f", label=" ", caption=scenario))
