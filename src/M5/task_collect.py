import numpy as np
import pandas as pd
from forecopy import cstools, csrec
import torch
from typing import Annotated

from M5.config import ALPHAs, data_catalog, OUTPUT_SAMPLE_SIZE, TABLES_PATH, M5_PATH
from utils import mle_estimation_skewed_normal
from pytask import task, Product
from pathlib import Path
from opt_rec_quantile.loss import pinball_loss
from M5.data_prepare import LEVEL_SPECS, add_derived_keys, make_series_names


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

    return df.loc[methods].to_latex(escape=False)


def benchmarks(samples, A, resids):
    A = np.array(A)
    resids = np.array(resids)
    A = np.array(A)
    params = cstools(agg_mat=A)
    olss = []
    wlss = []
    shrs = []
    sams = []
    for i in range(samples.shape[0]):
        ols = csrec(np.array(samples[i, :, :].T), params=params, res=resids)
        olss.append(ols.T)
        wls = csrec(np.array(samples[i, :, :].T), params=params, res=resids, comb="wls")
        wlss.append(wls.T)
        shr = csrec(
            np.array(samples[i, :, :].T),
            params=params,
            res=resids,
            comb="shr",
        )
        shrs.append(shr.T)
        sam = csrec(
            np.array(samples[i, :, :].T),
            params=params,
            res=resids,
            comb="sam",
        )
        sams.append(sam.T)
    return {
        "ols": torch.as_tensor(np.stack(olss)),
        "wls": torch.as_tensor(np.stack(wlss)),
        "shr": torch.as_tensor(np.stack(shrs)),
        "sam": torch.as_tensor(np.stack(sams)),
    }


for idx, dist in enumerate(["normal"]):
    seed = 20260706 + int(idx * 2000)

    rf = {alpha: data_catalog[f"rf_{alpha}_{dist}"] for alpha in ALPHAs}

    @task
    def task_collect(
        base: Annotated[dict, data_catalog["base"]],
        input_rf: Annotated[dict, rf],
        output: Annotated[Path, Product] = TABLES_PATH / f"M5_acc_{dist}.tex",
        output_spl: Annotated[Path, Product] = TABLES_PATH / f"M5_spl_{dist}.tex",
        dist: str = dist,
        seed: int = seed,
    ):
        generator = torch.Generator()
        generator.manual_seed(seed)
        S = torch.as_tensor(base["S"], dtype=torch.float64)
        n, m = S.shape
        A = S[:-m, :]
        resids = torch.as_tensor(base["resids"], dtype=torch.float64)
        mean = torch.as_tensor(base["y_pred"], dtype=torch.float64)
        true_y = torch.as_tensor(base["y_future"], dtype=torch.float64)
        hist = torch.as_tensor(base["y_hist"], dtype=torch.float64)

        skewnormal = [mle_estimation_skewed_normal(resids[:, i]) for i in range(n)]

        def sampling(J=OUTPUT_SAMPLE_SIZE):
            T = true_y.shape[0]
            if dist == "normal":
                resids_mean = torch.mean(resids, dim=0).unsqueeze(-1).expand(T, n, J)
                resids_std = torch.std(resids, dim=0).unsqueeze(-1).expand(T, n, J)
                smp = torch.normal(resids_mean, resids_std, generator=generator)
                return smp + mean[:, :, None]
            elif dist == "skewnormal":
                smp = [skewnormal[i].sample((T, J)) for i in range(n)]
                smp = torch.stack(smp, dim=1)
                return smp + mean[:, :, None]
            return None

        samples = sampling()
        res = benchmarks(samples, A, resids)

        res["base"] = samples
        output_dict = {
            "method": [],
            "loss": [],
            "alpha": [],
            "spl": [],
            "idx": [],
            "name": [],
        }
        weights = pd.read_csv(M5_PATH / "weights_evaluation.csv").rename(
            columns={"Agg_Level_1": "item_id", "Agg_Level_2": "store_id"}
        )
        weights = add_derived_keys(weights)
        weights = weights[weights["Level_id"] == "Level12"]
        weights_df = {"name": [], "weights": [], "level": []}
        for level in LEVEL_SPECS:
            if level.group_keys:
                level_weights = (
                    weights.groupby([i for i in level.group_keys])["weight"]
                    .sum()
                    .reset_index()
                )
                weights_df["name"].extend(
                    make_series_names(level_weights, level.group_keys).tolist()
                )
                weights_df["level"].extend([level.M5_level] * level_weights.shape[0])
                weights_df["weights"].extend(level_weights["weight"].values.tolist())
            else:
                weights_df["name"].append("Total")
                weights_df["weights"].append(1)
                weights_df["level"].append("Level1")

        weights_df = pd.DataFrame(weights_df)
        for alpha in ALPHAs:
            G, d = input_rf[alpha]["result"]
            rf_samples = (
                torch.einsum("tnj,kn->tkj", samples, S @ G)
                + (S @ d[:, 0])[None, :, None]
            )
            res["QOpt"] = rf_samples
            for method, smp in res.items():
                q = torch.quantile(smp, alpha, dim=2)
                for i in range(n):
                    loss = (
                        pinball_loss(true_y[:, i] - q[:, i], alpha=alpha)
                        .detach()
                        .item()
                    )
                    spl = loss / hist[:, i].diff().abs().mean().item()

                    output_dict["method"].append(method)
                    output_dict["alpha"].append(alpha)
                    output_dict["loss"].append(loss)
                    output_dict["spl"].append(spl)
                    output_dict["idx"].append(i)
                    output_dict["name"].append(base["names"][i])
        df = pd.DataFrame(output_dict).merge(weights_df, on=["name"], how="left")

        df1 = (
            df.groupby(["alpha", "method"])["spl"]
            .mean()
            .reset_index()
            .pivot(index="method", columns="alpha", values="spl")
        )
        df2 = (
            df.groupby(["method", "level", "idx"])[["spl", "weights"]]
            .mean()
            .reset_index()
        )
        df2["spl"] = df2["spl"] * df2["weights"]
        df2 = (
            df2.groupby(["method", "level"])[["spl"]]
            .sum()
            .reset_index()
            .pivot(index="method", columns="level", values="spl")
            .sort_index(axis=1)
            .reset_index()
        )
        df1.columns = [f"{i:.3f}" for i in df1.columns]
        benchmark = pd.DataFrame(
            [
                [
                    "ARIMA",
                    0.158,
                    0.148,
                    0.163,
                    0.147,
                    0.167,
                    0.170,
                    0.202,
                    0.178,
                    0.201,
                ]
            ],
            columns=["method"] + [f"Level{i}" for i in range(1, 10)],
        )
        df2 = pd.concat([df2, benchmark], ignore_index=True)
        df2["Average"] = df2[[f"Level{level}" for level in range(1, 10)]].mean(axis=1)
        df2.set_index("method", inplace=True)
        methods = ["base", "QOpt", "ols", "wls", "shr", "sam"]
        output.write_text(style(df1, methods))
        output_spl.write_text(style(df2, methods))


if __name__ == "__main__":
    for idx, dist in enumerate(["normal"]):
        seed = 20260706 + int(idx * 2000)
        base = data_catalog["base"].load()
        rf = {alpha: data_catalog[f"rf_{alpha}_{dist}"].load() for alpha in ALPHAs}
        task_collect(
            base, rf, dist=dist, seed=seed, output=TABLES_PATH / f"M5_acc_{dist}.tex"
        )
