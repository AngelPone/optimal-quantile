from simulation.config import (
    ALPHAs,
    SCENARIOS,
    TEST_WINDOWS,
    data_catalog,
    OUTPUT_SAMPLE_SIZE,
    A,
    scenario_seed,
    WINDOW_S,
    LOG_PATH,
)
from utils import expanding_window
import torch

from pytask import task
from typing import Annotated, List
from forecopy import cscov, cstools
import numpy as np
import pandas as pd


def spl(x, alpha, weights):
    loss = torch.where(x < 0, -(1 - alpha) * x, alpha * x)
    return loss / weights


def benchmarks(A, resids):
    params = cstools(agg_mat=A.cpu().numpy())
    S = torch.concat([A, torch.eye(A.shape[1], dtype=A.dtype, device=A.device)])
    output = {}
    for m in ["ols", "wls", "shr", "sam"]:
        cov_mat = cscov(params, resids.T).fit(comb=m)
        cov_mat_inv = torch.as_tensor(
            np.linalg.inv(cov_mat), dtype=A.dtype, device=A.device
        )
        rec_mat = S @ torch.linalg.solve(S.T @ cov_mat_inv @ S, S.T @ cov_mat_inv)
        output[m] = rec_mat
    output["base"] = torch.eye(S.shape[0], dtype=A.dtype, device=A.device)
    return output


for scenario in SCENARIOS:
    seed = scenario_seed(scenario, offset=30_000)
    rf = [data_catalog[f"simulation_rf_{scenario}_{alpha}"] for alpha in ALPHAs]

    @task
    def task_collect(
        input_data: Annotated[List[np.ndarray], data_catalog[f"simulation_{scenario}"]],
        input_base: Annotated[List, data_catalog[f"simulation_base_{scenario}"]],
        input_rf: list = rf,
        scenario: str = scenario,
        seed: int = seed,
    ):
        torch.manual_seed(seed)
        mean = torch.stack(
            [window["mean"] for window in input_base[:-TEST_WINDOWS]],
        )
        samples = torch.stack(
            [
                window["dist"].sample(OUTPUT_SAMPLE_SIZE).permute((1, 0))
                for window in input_base[:-TEST_WINDOWS]
            ]
        )
        samples = mean[:, :, None] + samples

        S = torch.concat([A, torch.eye(6, dtype=samples.dtype, device=samples.device)])

        qs = {alpha: {} for alpha in ALPHAs}
        for alpha_idx, alpha in enumerate(ALPHAs):
            G, d = input_rf[alpha_idx]["result"]
            rf_samples = (
                torch.einsum("tnj,kn->tkj", samples, S @ G)
                + (S @ d[:, 0])[None, :, None]
            )
            qs[alpha]["QOpt"] = torch.quantile(rf_samples, alpha, dim=2)

        benchmarks_mat = benchmarks(A, input_base[-TEST_WINDOWS - 1]["resid"])
        for m, v in benchmarks_mat.items():
            rf_samples = torch.einsum("tnj,kn->tkj", samples, v)
            for alpha in ALPHAs:
                qs[alpha][m] = torch.quantile(rf_samples, alpha, dim=2)

        y = torch.as_tensor(input_data["y"], dtype=A.dtype, device=A.device)
        true_y = expanding_window(y.shape[0], WINDOW_S, 1, y).collect_test().squeeze()
        weights = torch.diff(y, dim=0).abs().mean(dim=0)
        true_y = true_y[:-TEST_WINDOWS, :]

        output_dict = {"method": [], "loss": [], "alpha": [], "window": [], "idx": []}

        for alpha in ALPHAs:
            for method, q in qs[alpha].items():
                loss = spl(true_y - q, alpha=alpha, weights=weights)
                for i in range(loss.shape[0]):
                    for j in range(loss.shape[1]):
                        output_dict["method"].append(method)
                        output_dict["alpha"].append(alpha)
                        output_dict["loss"].append(loss[i, j].item())
                        output_dict["window"].append(i)
                        output_dict["idx"].append(j)
        df = pd.DataFrame(output_dict)
        df.to_csv(LOG_PATH / f"simulation_{scenario}_insample.csv")
