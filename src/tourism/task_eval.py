import torch
from pytask import task, Product
from tourism.config import ALPHAs, BLD, S, TEST_WINDOWS, WINDOW_S, data_catalog
from utils import expanding_window
from typing import Annotated
from opt_rec_quantile.loss import pinball_loss
import pandas as pd
from pathlib import Path
import numpy as np

for dist in ["normal", "skewt"]:
    rf_samples = [data_catalog[f"tourism_samples_{alpha}_{dist}"] for alpha in ALPHAs]

    @task
    def task_evaluate(
        input_data: Annotated[np.ndarray, data_catalog["tourism"]],
        input_samples: Annotated[list[dict], rf_samples],
        output: Annotated[Path, Product] = BLD / "tables" / f"tourism_acc_{dist}.tex",
    ):
        true_y = (
            expanding_window(input_data.shape[0], WINDOW_S, 1, input_data)
            .collect_test()
            .squeeze()[-TEST_WINDOWS:, :]
        )
        true_y = torch.as_tensor(true_y, dtype=torch.float64)
        assert true_y.shape == (
            TEST_WINDOWS,
            S.shape[0],
        ), f"dim {true_y.shape} is not correct"
        output_dict = {"method": [], "loss": [], "alpha": []}
        for alpha, samples_by_method in zip(ALPHAs, input_samples):
            for method, samples in samples_by_method.items():
                q = torch.quantile(samples, alpha, dim=2)
                loss = (
                    pinball_loss(true_y - q, alpha=alpha).detach().item() * S.shape[0]
                )
                output_dict["method"].append(method)
                output_dict["alpha"].append(alpha)
                output_dict["loss"].append(loss)
        df = pd.DataFrame(output_dict)
        df = df.pivot(index="method", columns="alpha", values="loss")
        df.columns = [f"{i:.2f}" for i in df.columns]
        output.write_text(
            df.to_latex(float_format="%.2f", label=" ", caption="Tourism")
        )
