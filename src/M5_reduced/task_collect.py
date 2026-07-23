import numpy as np
import pickle as pkl
import pandas as pd
from forecopy import cstools, csrec
import torch
from typing import Annotated

from M5_reduced.config import (
    ALPHAs,
    BETAs,
    OUTPUT_SAMPLE_SIZE,
    TABLES_PATH,
    DATA_OUTPUT_PATH,
    BETAs,
    LOGGING_PATH,
    M5_PATH,
    data_catalog,
)
from M5_reduced.data_prepare import add_derived_keys, LEVEL_SPECS, make_series_names
from pytask import task, Product
from pathlib import Path
from opt_rec_quantile.loss import pinball_loss


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


def benchmarks(samples, A, resids, alpha):
    A = A.numpy()
    params = cstools(agg_mat=A)
    olss = []
    wlss = []
    shrs = []
    sams = []
    resids[np.abs(resids) > 10000] = resids.mean()
    resids_demean = resids - resids.mean(axis=0)[None, :]
    for i in range(samples.shape[0]):
        smps = samples[i, :, :].T.numpy()
        ols = csrec(smps, params=params, res=resids_demean, comb="ols")
        olss.append(ols.T)
        wls = csrec(smps, params=params, res=resids_demean, comb="wls")
        wlss.append(wls.T)
        shr = csrec(smps, params=params, res=resids_demean, comb="shr")
        shrs.append(shr.T)
        sam = csrec(smps, params=params, res=resids_demean, comb="sam")
        sams.append(sam.T)
    return [
        {
            "ols": torch.as_tensor(np.quantile(np.stack(olss), q=alpha, axis=2)),
            "wls": torch.as_tensor(np.quantile(np.stack(wlss), q=alpha, axis=2)),
            "shr": torch.as_tensor(np.quantile(np.stack(shrs), q=alpha, axis=2)),
            "sam": torch.as_tensor(np.quantile(np.stack(sams), q=alpha, axis=2)),
        }
        for alpha in alpha
    ]


for idx, dist in enumerate(["normal", "skewnormal"]):
    seed = 20260720 + idx
    rf = [
        {alpha: data_catalog[f"rf_{alpha}_{beta}_{dist}"] for alpha in ALPHAs}
        for beta in BETAs
    ]

    @task
    def task_collect(
        base: Annotated[dict, data_catalog["base"]],
        input_rf: Annotated[dict, rf],
        data_path: Path = DATA_OUTPUT_PATH,
        output: Annotated[Path, Product] = TABLES_PATH / f"M5_ets_{dist}.tex",
        output_spl: Annotated[Path, Product] = TABLES_PATH / f"M5_ets_spl_{dist}.tex",
        output_df: Annotated[Path, Product] = LOGGING_PATH / f"M5_ets_{dist}.csv",
        dist: str = dist,
        seed: int = seed,
    ):
        generator = torch.Generator()
        generator.manual_seed(seed)

        with open(data_path, "rb") as f:
            data = pkl.load(f)
            S = torch.as_tensor(data["S"], dtype=torch.float64)
            names = data["names"]
        n, m = S.shape

        A = S[:-m, :]

        def sampling(dist: str, smp_slice, size: int = OUTPUT_SAMPLE_SIZE):

            if dist == "normal":
                smp_loc = (
                    torch.stack(
                        [
                            torch.stack([base[i]["normal"][j].loc for j in range(n)])
                            for i in smp_slice
                        ]
                    )
                    .unsqueeze(-1)
                    .expand(len(smp_slice), n, size)
                )
                smp_scale = (
                    torch.stack(
                        [
                            torch.stack([base[i]["normal"][j].scale for j in range(n)])
                            for i in smp_slice
                        ]
                    )
                    .unsqueeze(-1)
                    .expand(len(smp_slice), n, size)
                )
                smps = torch.normal(smp_loc, smp_scale, generator=generator)
            elif dist == "skewnormal":
                smps = torch.stack(
                    [
                        torch.stack(
                            [
                                base[i][dist][j].sample(size, generator=generator)
                                for j in range(n)
                            ]
                        )
                        for i in smp_slice
                    ]
                )
            return mean[smp_slice, :, None] + smps

        test_slice = range(len(base) - 1, len(base))
        resids = base[-1]["resid"]
        output_dict = {
            "method": [],
            "loss": [],
            "alpha": [],
            "h": [],
            "series_code": [],
            "name": [],
        }

        hist = base[0]["hist"]
        hist = np.concat([hist, np.stack([i["future"][0,] for i in base[:-1]])])
        mod = [np.abs(np.diff(hist[:, i])).mean() for i in range(hist.shape[1])]

        for h in range(1, 29):
            mean = torch.stack([i["mean"][:, h - 1] for i in base])
            true_y = torch.as_tensor(
                np.stack([i["future"][h - 1, :] for i in base]),
                dtype=mean.dtype,
                device=mean.device,
            )
            smps = sampling(dist, test_slice)
            res = benchmarks(smps, A, resids, ALPHAs)

            for alpha_idx, alpha in enumerate(ALPHAs):
                res[alpha_idx]["base"] = torch.quantile(smps, q=alpha, dim=2)
                for idx, beta in enumerate(BETAs):
                    G, d = input_rf[idx][alpha]["result"]
                    d = d[:, 0]
                    rf_samples = (
                        torch.einsum("tnj,kn->tkj", smps, S @ G)
                        + (S @ d)[None, :, None]
                    )
                    q = torch.quantile(rf_samples, alpha, dim=2)
                    res[alpha_idx][f"QOpt($\\beta={beta}$)"] = q
                for method, q in res[alpha_idx].items():
                    loss = [
                        (
                            pinball_loss(true_y[test_slice, i] - q[0, i], alpha=alpha)
                            .detach()
                            .item()
                            / mod[i]
                        )
                        for i in range(q.numel())
                    ]
                    for i in range(len(loss)):
                        output_dict["method"].append(method)
                        output_dict["alpha"].append(alpha)
                        output_dict["loss"].append(loss[i])
                        output_dict["h"].append(h)
                        output_dict["series_code"].append(i)
                        output_dict["name"].append(names[i])
        df = pd.DataFrame(output_dict)
        df.to_csv(output_df)
        df1 = (
            df.groupby(["method", "alpha"])
            .mean(numeric_only=True)["loss"]
            .reset_index()
        )
        methods = (
            ["base"]
            + [f"QOpt($\\beta={beta}$)" for beta in BETAs]
            + ["ols", "wls", "shr", "sam"]
        )
        df1 = df1.pivot(index="method", columns="alpha", values="loss")
        df1.columns = [f"{i:.3f}" for i in df1.columns]
        df1 = style(df1, methods)
        output.write_text(df1)

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
                weights_df["level"].extend([level.key] * level_weights.shape[0])
                weights_df["weights"].extend(level_weights["weight"].values.tolist())
            else:
                weights_df["name"].append("Total")
                weights_df["weights"].append(1)
                weights_df["level"].append("level1")

        weights_df = pd.DataFrame(weights_df)
        df = df.merge(weights_df, on="name", how="left")
        df2 = (
            df.groupby(["method", "level", "series_code"])[["loss", "weights"]]
            .mean()
            .reset_index()
        )
        df2["spl"] = df2["loss"] * df2["weights"]
        df2 = (
            df2.groupby(["method", "level"])[["spl"]]
            .sum()
            .reset_index()
            .pivot(index="method", columns="level", values="spl")
            .sort_index(axis=1)
            .reset_index()
        )
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
            columns=["method"] + [f"level{i}" for i in range(1, 10)],
        )
        df2 = pd.concat([df2, benchmark], ignore_index=True)
        df2["Average"] = df2[[f"level{level}" for level in range(1, 10)]].mean(axis=1)
        df2.set_index("method", inplace=True)
        df2 = style(df2, ["ARIMA"] + methods)
        output_spl.write_text(df2)


if __name__ == "__main__":
    for idx, dist in enumerate(["normal", "skewnormal"]):
        seed = 20260720 + idx
        rf = [
            {
                alpha: data_catalog[f"rf_{alpha}_{beta}_{dist}"].load()
                for alpha in ALPHAs
            }
            for beta in BETAs
        ]
        base = data_catalog["base"].load()
        task_collect(base, rf, dist=dist, seed=seed)
