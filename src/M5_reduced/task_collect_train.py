import pickle as pkl
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import torch
from forecopy import cscov, cstools
from pytask import Product

from M5_reduced.config import DATA_OUTPUT_PATH, DEVICE, DTYPE, data_catalog, M5_PATH
from M5_reduced.data_prepare import add_derived_keys, make_series_names, LEVEL_SPECS
from utils import SkewNormal


def get_M5_weights():
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
    return weights_df


def task_collect_train(
    base: Annotated[list, data_catalog["base"]],
    data_path: Path = DATA_OUTPUT_PATH,
    output: Annotated[dict, Product] = data_catalog["train_data"],
    output_test: Annotated[dict, Product] = data_catalog["test_data"],
):
    mean = torch.stack([i["mean"][:, 0] for i in base[:-1]]).to(
        dtype=DTYPE, device=DEVICE
    )
    true_y = torch.as_tensor(
        np.stack([i["future"][0, :] for i in base[:-1]]),
        dtype=mean.dtype,
        device=mean.device,
    )
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    S = torch.as_tensor(data["S"], dtype=DTYPE, device=DEVICE)
    n, m = S.shape
    A = S[:-m, :]

    hist = base[0]["hist"]
    hist = np.concat([hist, np.stack([i["future"][0,] for i in base[:-1]])])
    weights = np.array(
        [np.abs(np.diff(hist[:, i])).mean() for i in range(hist.shape[1])]
    )
    weights = torch.as_tensor(weights, dtype=true_y.dtype, device=true_y.device)

    resids = base[-1]["resid"]
    resids[np.abs(resids) > 10000] = resids.mean()
    resids = resids - resids.mean(axis=0)
    params = cstools(agg_mat=A.cpu().numpy())
    W = cscov(params, res=resids).fit(comb="shr")
    W_sam = torch.as_tensor(
        np.linalg.inv(cscov(params, res=resids).fit(comb="sam")),
        dtype=DTYPE,
        device=DEVICE,
    )
    W_wls = torch.as_tensor(
        np.linalg.inv(cscov(params, res=resids).fit(comb="wls")),
        dtype=DTYPE,
        device=DEVICE,
    )
    W = torch.as_tensor(np.linalg.inv(W), dtype=DTYPE, device=DEVICE)
    shr_mat = torch.linalg.solve(S.T @ W @ S, S.T @ W)
    wls_mat = torch.linalg.solve(S.T @ W_wls @ S, S.T @ W_wls)
    normal_loc = torch.tensor(
        [[i["normal"][j].loc for j in range(n)] for i in base[:-1]]
    ).to(device=DEVICE, dtype=DTYPE)
    normal_scale = torch.tensor(
        [[i["normal"][j].scale for j in range(n)] for i in base[:-1]]
    ).to(device=DEVICE, dtype=DTYPE)

    sn_xi = torch.tensor(
        [[i["skewnormal"][j].xi for j in range(n)] for i in base[:-1]]
    ).to(device=DEVICE, dtype=DTYPE)

    output.save(
        {
            "A": A,
            "weights": weights,
            "y": true_y,
            "mean": mean,
            "G_shr": shr_mat,
            "G_wls": wls_mat,
            "normal": (normal_loc, normal_scale),
            "skewnormal": (sn_xi, normal_loc, normal_scale),
        }
    )

    normal_loc = torch.tensor([base[-1]["normal"][j].loc for j in range(n)]).to(
        device=DEVICE, dtype=DTYPE
    )
    normal_scale = torch.tensor([base[-1]["normal"][j].scale for j in range(n)]).to(
        device=DEVICE, dtype=DTYPE
    )

    snormal_xi = torch.tensor([base[-1]["skewnormal"][j].xi for j in range(n)]).to(
        device=DEVICE, dtype=DTYPE
    )

    ols_mat = torch.linalg.solve(S.T @ S, S.T)
    W = torch.diag(torch.diagonal(W))
    wls_mat = torch.linalg.solve(S.T @ W @ S, S.T @ W)
    sam_mat = torch.linalg.solve(S.T @ W_sam @ S, S.T @ W_sam)

    # compute M5 weights
    weights_df = get_M5_weights().loc[:, ["name", "weights", "level"]]
    m5_weights = (
        pd.DataFrame({"idx": range(n), "name": data["names"]})
        .merge(weights_df, on="name", how="left")
        .sort_values("idx")
        .loc[:, ["weights", "idx", "level"]]
    )
    true_y = torch.as_tensor(
        base[-1]["future"],
        dtype=mean.dtype,
        device=mean.device,
    )
    output_test.save(
        {
            "A": A,
            "weights": weights,
            "mean": base[-1]["mean"].to(dtype=DTYPE, device=DEVICE).T,
            "normal": torch.distributions.Normal(normal_loc, normal_scale),
            "skewnormal": SkewNormal(snormal_xi, normal_loc, normal_scale),
            "G": {"shr": shr_mat, "ols": ols_mat, "wls": wls_mat, "sam": sam_mat},
            "m5_weights": m5_weights,
            "y": true_y,
        }
    )


if __name__ == "__main__":
    task_collect_train(data_catalog["base"].load())
