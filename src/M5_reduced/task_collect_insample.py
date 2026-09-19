from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
from pytask import Product, task
from torch.distributions import Normal
from utils import SkewStudentT

from M5_reduced.config import (
    DEVICE,
    DTYPE,
    LOGGING_PATH,
    OUTPUT_SAMPLE_SIZE,
    TABLES_PATH,
    ALPHAs,
    BETAs,
    data_catalog,
    OUTSAMPLE_H,
    DISTS,
)


def mean_losses_over_batches(n_windows, batch_size, evaluate_batch):
    loss_sums = {}
    n_processed = 0
    expected_keys = None
    for start in range(0, n_windows, batch_size):
        stop = min(start + batch_size, n_windows)
        batch_losses = evaluate_batch(slice(start, stop))
        batch_keys = set(batch_losses)
        if expected_keys is None:
            expected_keys = batch_keys
        elif batch_keys != expected_keys:
            raise ValueError("evaluate_batch returned inconsistent loss keys")

        for key, loss in batch_losses.items():
            if loss.shape[0] != stop - start:
                raise ValueError("batch loss has an inconsistent first dimension")
            batch_sum = loss.sum(dim=0)
            loss_sums[key] = loss_sums.get(key, 0) + batch_sum
        n_processed += stop - start

    return {key: loss_sum / n_processed for key, loss_sum in loss_sums.items()}


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


for idx, dist in enumerate(DISTS):
    seed = 20260720 + idx
    rf = {
        alpha: {
            beta: {
                h: data_catalog[f"rf_{alpha}_{beta}_{dist}_outsample_h{h}"]
                for h in OUTSAMPLE_H
            }
            for beta in BETAs
        }
        for alpha in ALPHAs
    }

    @task
    @torch.no_grad()
    def task_collect(
        input_rf: Annotated[dict, rf],
        input_data: Annotated[dict, data_catalog["train_data"]],
        output_df: Annotated[Path, Product] = LOGGING_PATH
        / f"M5_ets_{dist}_insample.csv",
        dist: str = dist,
        seed: int = seed,
    ):
        torch.manual_seed(seed)

        def sampling(loc, scale, xi, df, mean, size: int = OUTPUT_SAMPLE_SIZE):
            if dist == "normal":
                smps = Normal(loc, scale).sample((size,)).permute((1, 2, 0))
            elif dist == "skew":
                smps = (
                    SkewStudentT(xi, df, loc, scale).sample((size,)).permute((1, 2, 0))
                )
            smps = smps.to(dtype=DTYPE, device=DEVICE)
            return mean[:, :, None] + smps

        A = input_data["A"]
        S = torch.concat([A, torch.eye(A.shape[1], dtype=A.dtype, device=A.device)])
        alpha_tensor = torch.tensor(ALPHAs, dtype=DTYPE, device=DEVICE)

        dfs = []
        for h in range(28):
            loc = input_data["out-of-sample"]["loc"][h]
            scale = input_data["out-of-sample"]["scale"][h]
            xi = input_data["out-of-sample"]["xi"][h]
            df = input_data["out-of-sample"]["df"][h]
            mean = input_data["mean"][h]
            true_y = input_data["y"][h]
            benchmark_matrices = {
                m: S @ input_data["out-of-sample"]["G"][h][m]
                for m in ["sam", "shr", "wls", "ols"]
            }
            qopt_transforms = {}
            for beta in BETAs:
                m = f"QOpt($\\beta={beta}$)"
                for alpha in ALPHAs:
                    g, d = input_rf[alpha][beta][h]["result"]
                    qopt_transforms[(m, alpha)] = (
                        S @ g,
                        (S @ d[:, 0])[None, :, None],
                    )

            def evaluate_batch(batch):
                smps = sampling(
                    loc[batch],
                    scale[batch],
                    xi[batch],
                    df[batch],
                    mean[batch],
                )

                qf = {"base": torch.quantile(smps, alpha_tensor, dim=2)}
                for m, matrix in benchmark_matrices.items():
                    rf_smps = torch.einsum("kn,TnJ->TkJ", matrix, smps)
                    qf[m] = torch.quantile(rf_smps, alpha_tensor, dim=2)

                for beta in BETAs:
                    m = f"QOpt($\\beta={beta}$)"
                    qs = []
                    for alpha in ALPHAs:
                        matrix, bias = qopt_transforms[(m, alpha)]
                        rf_smps = torch.einsum("kn,TnJ->TkJ", matrix, smps) + bias
                        qs.append(torch.quantile(rf_smps, alpha, dim=2))
                    qf[m] = torch.stack(qs)

                return {
                    (method, alpha): spl(
                        true_y[batch] - q[alpha_idx],
                        alpha,
                        input_data["weights"],
                    )
                    for alpha_idx, alpha in enumerate(ALPHAs)
                    for method, q in qf.items()
                }

            batch_size = 16
            n_windows = true_y.shape[0]
            for start in range(0, n_windows, 16):
                stop = min(start + batch_size, n_windows)
                batch_losses = evaluate_batch(slice(start, stop))
                for (method, alpha), loss in batch_losses.items():
                    loss_mat = pd.DataFrame(loss.cpu().numpy())
                    loss_mat.columns = [f"series{i+1}" for i in range(loss.shape[1])]
                    loss_mat["window_idx"] = range(start, stop)
                    loss_mat["method"] = method
                    loss_mat["alpha"] = alpha
                    loss_mat["h"] = h

                    dfs.append(loss_mat)

        df = pd.concat(dfs)
        df.to_csv(output_df)
        # df1 = (
        #     df.groupby(["method", "alpha"])
        #     .mean(numeric_only=True)["loss"]
        #     .reset_index()
        # )
        # methods = (
        #     ["base"]
        #     + [f"QOpt($\\beta={beta}$)" for beta in BETAs]
        #     + ["ols", "wls", "shr", "sam"]
        # )
        # df1 = df1.pivot(index="method", columns="alpha", values="loss")
        # df1.columns = [f"{i:.3f}" for i in df1.columns]
        # df1 = style(df1, methods)
        # output.write_text(df1)

        # df_ = (
        #     df.groupby(["h", "method", "alpha"])
        #     .mean(numeric_only=True)["loss"]
        #     .reset_index()
        # )
        # for h in OUTSAMPLE_H:
        #     output_path = TABLES_PATH / f"M5_ets_{dist}_h{h}_outsample.tex"
        #     dfh = df_[df_["h"] == h].pivot(
        #         index="method", columns="alpha", values="loss"
        #     )
        #     dfh.columns = [f"{i:.3f}" for i in dfh.columns]
        #     dfh = style(dfh, methods)
        #     output_path.write_text(dfh)

        # df = df.merge(input_data["m5_weights"], how="left", on="idx")
        # df_ = (
        #     df.groupby(["h", "method", "level"])
        #     .mean(numeric_only=True)["loss"]
        #     .reset_index()
        # )
        # for h in [0, 7, 14, 27]:
        #     output_path = TABLES_PATH / f"M5_ets_{dist}_h{h}_level.tex"
        #     dfh = df_[df_["h"] == h].pivot(
        #         index="method", columns="level", values="loss"
        #     )
        #     dfh = dfh[[f"level{i}" for i in range(1, 10)]]
        #     column_names = [
        #         "Total",
        #         "State",
        #         "Store",
        #         "Category",
        #         "Department",
        #         r"State\\Category",
        #         r"State\\Department",
        #         r"Store\\Category",
        #         r"Store\\Department",
        #     ]
        #     dfh.columns = [rf"{i}\\" + j for i, j in zip(range(1, 10), column_names)]
        #     dfh = style(dfh, methods)
        #     output_path.write_text(dfh)
