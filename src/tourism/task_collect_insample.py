from tourism.config import (
    A,
    ALPHAs,
    OUTPUT_SAMPLE_SIZE,
    BLD,
    S,
    TEST_WINDOWS,
    DEVICE,
    DTYPE,
    data_catalog,
)
from utils import SkewStudentT
import torch
from torch.distributions import Normal

from pytask import task
from typing import Annotated, Any
from forecopy import cstools, cscov
import numpy as np
import pandas as pd


def style(df, methods):
    def format_value(x):
        if x == minimum:
            return rf"\textbf{{{x:.3f}}}"
        if second_minimum is not None and x == second_minimum:
            return rf"\textcolor{{red}}{{{x:.3f}}}"
        return f"{x:.3f}"

    for col in df.columns:
        unique_values = df[col].dropna().unique()
        unique_values.sort()
        minimum = unique_values[0]
        second_minimum = unique_values[1] if len(unique_values) > 1 else None
        df[col] = [format_value(value) for value in df[col]]

    target_rows = ["base", "QOpt($\\beta=1000$)"]
    latex = df.loc[methods].to_latex(escape=False)
    lines = latex.splitlines()
    new_lines = []

    for line in lines:
        new_lines.append(line)

        for target_row in target_rows:
            if line.strip().startswith(target_row):
                new_lines.append(r"\midrule")

    latex = "\n".join(new_lines)
    return latex


def spl(x, alpha, weights):
    loss = torch.where(x < 0, -(1 - alpha) * x, alpha * x)
    return loss / weights


def benchmarks(samples, A, resids, alphas):
    A = A.cpu().numpy()
    params = cstools(agg_mat=A)
    output = {}
    for method in ["ols", "sam", "wls", "shr"]:
        W = cscov(params=params, res=resids).fit(comb=method)
        W = torch.as_tensor(np.linalg.inv(W), dtype=DTYPE, device=DEVICE)
        g = torch.linalg.solve(S.T @ W @ S, S.T @ W)
        rf_smps = torch.einsum("tnj,kn->tkj", samples, S @ g)
        q = torch.quantile(rf_smps, alphas, dim=2)
        output[method] = q
    return output


for dist_idx, dist in enumerate(["normal", "skew"]):
    seed = 20260804 + dist_idx

    input_rf = {alpha: data_catalog[f"rf_{alpha}_{dist}"] for alpha in ALPHAs}

    @task
    def task_collect(
        input_data: Annotated[np.ndarray, data_catalog["tourism"]],
        input_base: Annotated[list[dict[str, Any]], data_catalog["base"]],
        input_rf: Annotated[dict, input_rf],
        dist: str = dist,
        seed: int = seed,
    ):
        torch.manual_seed(seed)

        test_slice = slice(0, input_base["mean"].shape[0] - TEST_WINDOWS)
        point_f = input_base["mean"][:-TEST_WINDOWS, :]
        loc, scale, xi, df = (
            input_base["dist"]["mean"],
            input_base["dist"]["std"],
            input_base["dist"]["xi"],
            input_base["dist"]["df"],
        )

        if dist == "skew":
            samples = (
                SkewStudentT(
                    xi[test_slice],
                    df[test_slice],
                    loc[test_slice],
                    scale[test_slice],
                )
                .sample((OUTPUT_SAMPLE_SIZE,))
                .permute((1, 2, 0))
            )
        elif dist == "normal":
            samples = (
                Normal(loc[test_slice], scale[test_slice])
                .sample((OUTPUT_SAMPLE_SIZE,))
                .permute((1, 2, 0))
            )

        samples = point_f[:, :, None] + samples
        alpha_vec = torch.as_tensor(ALPHAs, device=DEVICE, dtype=DTYPE)
        # benchmarks
        res = benchmarks(samples, A, input_base["resids"][-TEST_WINDOWS], alpha_vec)
        res["base"] = torch.quantile(samples, alpha_vec, dim=2)

        # QOpt
        qs = []
        for alpha in ALPHAs:
            G, d = input_rf[alpha]["result"]
            rf_samples = (
                torch.einsum("tnj,kn->tkj", samples, S @ G)
                + (S @ d[:, 0])[None, :, None]
            )
            qs.append(torch.quantile(rf_samples, q=alpha, dim=2))
        res[r"QOpt($\beta=100$)"] = torch.stack(qs)

        # evaluate
        true_y = input_base["true_y"][test_slice, :]
        weights = np.abs(np.diff(input_data, axis=0)).mean(axis=0)
        weights = torch.as_tensor(weights, device=DEVICE, dtype=DTYPE)

        output_df = {"method": [], "alpha": [], "loss": [], "window": [], "idx": []}
        for method, qs in res.items():
            for alpha_idx, alpha in enumerate(ALPHAs):
                loss = spl(true_y - qs[alpha_idx], alpha=alpha, weights=weights)
                for i in range(loss.shape[0]):
                    for j in range(loss.shape[1]):
                        output_df["method"].append(method)
                        output_df["alpha"].append(alpha)
                        output_df["loss"].append(loss[i, j].item())
                        output_df["window"].append(i)
                        output_df["idx"].append(j)
        output_df = pd.DataFrame(output_df)
        output_df.to_csv(BLD / "logs" / f"tourism_{dist}_insample.csv")
