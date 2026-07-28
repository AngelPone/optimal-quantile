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
from utils import mle_estimation_skewed_dist


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


def linear_rec_mat(A, resids):
    params = cstools(agg_mat=A)
    S = np.concat([A, np.identity(A.shape[1])])
    resids = resids - resids.mean(axis=0)
    out = {}
    for m in ["ols", "wls", "sam", "shr"]:
        W = np.linalg.inv(cscov(params, res=resids).fit(comb=m))
        rec_mat = np.linalg.solve(S.T @ W @ S, S.T @ W)
        out[m] = torch.as_tensor(rec_mat, dtype=DTYPE, device=DEVICE)
    return out


def task_collect_train(
    base: Annotated[list, data_catalog["base"]],
    data_path: Path = DATA_OUTPUT_PATH,
    output: Annotated[dict, Product] = data_catalog["train_data"],
    output_test: Annotated[dict, Product] = data_catalog["test_data"],
):
    mean = [
        torch.stack([i["mean"][:, h - 1] for i in base[:-h]]).to(
            dtype=DTYPE, device=DEVICE
        )
        for h in range(1, 29)
    ]
    true_y = [
        torch.as_tensor(
            np.stack([i["future"][h - 1, :] for i in base[:-h]]),
            dtype=DTYPE,
            device=DEVICE,
        )
        for h in range(1, 29)
    ]
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
    weights = torch.as_tensor(weights, dtype=DTYPE, device=DEVICE)

    def preprocess_resid(x):
        resids_lst = []
        for i in range(x.shape[1]):
            resid = x[:, i]
            # christmas extreme error
            resid[resid < resid.mean() - 4 * resid.std()] = resid.mean()
            resids_lst.append(resid)
        resids = np.stack(resids_lst, axis=1)
        return resids

    # insample resids for training reconciliation matrix
    resids = preprocess_resid(base[-1]["resid"])
    rec_mats = linear_rec_mat(A.cpu().numpy(), resids)

    xi = []
    loc = []
    scale = []
    df = []
    for b in base:
        resids = preprocess_resid(b["resid"])
        dist = [mle_estimation_skewed_dist(resids[:, i]) for i in range(n)]
        xi.append(torch.tensor([i.xi for i in dist]))
        df.append(torch.tensor([i.df for i in dist]))
        loc.append(resids.mean(axis=0))
        scale.append(resids.std(axis=0))
    i_xi = torch.stack(xi).to(dtype=DTYPE, device=DEVICE)
    i_df = torch.stack(df).to(dtype=DTYPE, device=DEVICE)
    i_loc = torch.as_tensor(np.stack(loc), device=DEVICE, dtype=DTYPE)
    i_scale = torch.as_tensor(np.stack(scale), dtype=DTYPE, device=DEVICE)

    out_of_sample_resids = [true_y[h] - mean[h] for h in range(28)]

    o_loc = []
    o_scale = []
    o_xi = []
    o_resid = []
    o_df = []
    for h in range(28):
        resids = preprocess_resid(out_of_sample_resids[h].cpu().numpy())
        o_resid.append(resids)
        dist = [mle_estimation_skewed_dist(resids[:, i]) for i in range(n)]
        o_xi.append(torch.tensor([i.xi for i in dist]))
        o_df.append(torch.tensor([i.df for i in dist]))
        o_loc.append(resids.mean(axis=0))
        o_scale.append(resids.std(axis=0))
    o_xi = torch.stack(o_xi).to(dtype=DTYPE, device=DEVICE)
    o_scale = torch.as_tensor(np.stack(o_scale), dtype=DTYPE, device=DEVICE)
    o_loc = torch.as_tensor(np.stack(o_loc), dtype=DTYPE, device=DEVICE)
    o_df = torch.stack(o_df).to(dtype=DTYPE, device=DEVICE)
    o_rec_mats = [linear_rec_mat(A.cpu().numpy(), o_resid[h]) for h in range(28)]

    output.save(
        {
            "A": A,
            "weights": weights,
            "y": true_y,
            "mean": mean,
            "in-sample": {
                "loc": i_loc[:-1, :],
                "scale": i_scale[:-1, :],
                "xi": i_xi[:-1, :],
                "df": i_df[:-1, :],
                "G": rec_mats,
            },
            "out-of-sample": {
                "resids": o_resid,
                "loc": o_loc,
                "scale": o_scale,
                "xi": o_xi,
                "df": o_df,
                "G": o_rec_mats,
            },
        }
    )

    # compute M5 weights
    weights_df = get_M5_weights().loc[:, ["name", "weights", "level"]]
    m5_weights = (
        pd.DataFrame({"idx": range(n), "name": data["names"]})
        .merge(weights_df, on="name", how="left")
        .sort_values("idx")
        .loc[:, ["weights", "idx", "level"]]
    )
    true_y = torch.as_tensor(base[-1]["future"], dtype=DTYPE, device=DEVICE)
    output_test.save(
        {
            "A": A,
            "weights": weights,
            "y": true_y,
            "mean": base[-1]["mean"].to(dtype=DTYPE, device=DEVICE).T,
            "in-sample": {
                "loc": i_loc[-1, :],
                "scale": i_scale[-1, :],
                "xi": i_xi[-1, :],
                "df": i_df[-1, :],
                "G": rec_mats,
            },
            "out-of-sample": {
                "resid": o_resid,
                "loc": o_loc,
                "scale": o_scale,
                "xi": o_xi,
                "df": o_df,
                "G": o_rec_mats,
            },
            "m5_weights": m5_weights,
        }
    )


if __name__ == "__main__":
    task_collect_train(data_catalog["base"].load())
