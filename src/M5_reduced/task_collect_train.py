import pickle as pkl
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import torch
from forecopy import cscov, cstools
from pytask import Product

from M5_reduced.config import (
    DATA_OUTPUT_PATH,
    DEVICE,
    DTYPE,
    data_catalog,
    M5_PATH,
    TEST_WINDOWS,
)
from M5_reduced.data_prepare import add_derived_keys, make_series_names, LEVEL_SPECS
from utils import mle_estimation_skewed_dist

MIN_OUTSAMPLE_RESIDUALS = 365


def _split_expanding_series(values, parameters, min_residuals, test_windows):
    """Split values and expanding estimates while preserving origin alignment."""
    expected_parameters = values.shape[0] - min_residuals
    if parameters.shape[0] != expected_parameters:
        raise ValueError(
            "parameter rows must equal value rows minus the initial residual history: "
            f"got {parameters.shape[0]} and {values.shape[0]} - {min_residuals}"
        )
    if test_windows <= 0 or test_windows > parameters.shape[0]:
        raise ValueError(
            "test_windows must be positive and no larger than parameter rows"
        )

    return (
        values[min_residuals:-test_windows],
        values[-test_windows:],
        parameters[:-test_windows],
        parameters[-test_windows:],
    )


def _preprocess_resid(x):
    """Replace extreme negative residuals without mutating the input array."""
    residual_columns = []
    for series_idx in range(x.shape[1]):
        residual = x[:, series_idx].copy()
        mean = residual.mean()
        std = residual.std()
        residual[residual < mean - 4 * std] = mean
        residual_columns.append(residual)
    return np.stack(residual_columns, axis=1)


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
    # prepare data and summing matrix
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    S = torch.as_tensor(data["S"], dtype=DTYPE, device=DEVICE)
    n, m = S.shape
    A = S[:-m, :]

    # compute M5 weights
    weights_df = get_M5_weights().loc[:, ["name", "weights", "level"]]
    m5_weights = (
        pd.DataFrame({"idx": range(n), "name": data["names"]})
        .merge(weights_df, on="name", how="left")
        .sort_values("idx")
        .loc[:, ["weights", "idx", "level"]]
    )

    weights = np.stack(
        [np.abs(np.diff(data["data"][name][1].values)).mean() for name in data["names"]]
    )
    weights = torch.as_tensor(weights, dtype=DTYPE, device=DEVICE)

    all_data = np.stack(
        [data["data"][name][1].values for name in data["names"]], axis=1
    )

    mean = []
    true_y = []
    for h in range(28):
        origins = base[: len(base) - h]
        mean.append(
            torch.as_tensor(
                np.stack([origin["mean"][h, :] for origin in origins]),
                dtype=DTYPE,
                device=DEVICE,
            )
        )
        true_y.append(
            torch.as_tensor(
                np.stack(
                    [all_data[origin["slice"][1].start + h, :] for origin in origins]
                ),
                dtype=DTYPE,
                device=DEVICE,
            )
        )

    out_of_sample_resids = [true_y[h] - mean[h] for h in range(28)]

    o_loc = []
    o_scale = []
    o_xi = []
    o_df = []

    for h in range(28):
        T_t = out_of_sample_resids[h].shape[0]
        xi_, df_, loc_, scale_ = [], [], [], []
        for origin_idx in range(MIN_OUTSAMPLE_RESIDUALS, T_t):
            resids = _preprocess_resid(
                out_of_sample_resids[h][:origin_idx, :].cpu().numpy()
            )
            if (origin_idx - MIN_OUTSAMPLE_RESIDUALS) % 14 == 0:
                dist = [
                    mle_estimation_skewed_dist(resids[:, series_idx])
                    for series_idx in range(n)
                ]
                xi = torch.tensor(
                    [fitted.xi for fitted in dist], dtype=DTYPE, device=DEVICE
                )
                df = torch.tensor(
                    [fitted.df for fitted in dist], dtype=DTYPE, device=DEVICE
                )
            xi_.append(xi)
            df_.append(df)
            loc_.append(
                torch.as_tensor(resids.mean(axis=0), dtype=DTYPE, device=DEVICE)
            )
            scale_.append(
                torch.as_tensor(resids.std(axis=0), dtype=DTYPE, device=DEVICE)
            )
        o_xi.append(torch.stack(xi_))
        o_df.append(torch.stack(df_))
        o_scale.append(torch.stack(scale_))
        o_loc.append(torch.stack(loc_))
    o_rec_mats = [
        linear_rec_mat(
            A.cpu().numpy(),
            r[:-TEST_WINDOWS, :].detach().cpu().numpy(),
        )
        for r in out_of_sample_resids
    ]

    train_y, test_y = [], []
    train_mean, test_mean = [], []
    train_loc, test_loc = [], []
    train_scale, test_scale = [], []
    train_xi, test_xi = [], []
    train_df, test_df = [], []
    for h in range(28):
        y_train, y_test, loc_train, loc_test = _split_expanding_series(
            true_y[h], o_loc[h], MIN_OUTSAMPLE_RESIDUALS, TEST_WINDOWS
        )
        mean_train, mean_test, scale_train, scale_test = _split_expanding_series(
            mean[h], o_scale[h], MIN_OUTSAMPLE_RESIDUALS, TEST_WINDOWS
        )
        _, _, xi_train, xi_test = _split_expanding_series(
            mean[h], o_xi[h], MIN_OUTSAMPLE_RESIDUALS, TEST_WINDOWS
        )
        _, _, df_train, df_test = _split_expanding_series(
            mean[h], o_df[h], MIN_OUTSAMPLE_RESIDUALS, TEST_WINDOWS
        )
        train_y.append(y_train)
        test_y.append(y_test)
        train_mean.append(mean_train)
        test_mean.append(mean_test)
        train_loc.append(loc_train)
        test_loc.append(loc_test)
        train_scale.append(scale_train)
        test_scale.append(scale_test)
        train_xi.append(xi_train)
        test_xi.append(xi_test)
        train_df.append(df_train)
        test_df.append(df_test)

    output.save(
        {
            "A": A,
            "weights": weights,
            "y": train_y,
            "mean": train_mean,
            "out-of-sample": {
                "loc": train_loc,
                "scale": train_scale,
                "xi": train_xi,
                "df": train_df,
                "G": o_rec_mats,
            },
        }
    )

    output_test.save(
        {
            "A": A,
            "weights": weights,
            "y": test_y,
            "mean": test_mean,
            "out-of-sample": {
                "loc": test_loc,
                "scale": test_scale,
                "xi": test_xi,
                "df": test_df,
                "G": o_rec_mats,
            },
            "m5_weights": m5_weights,
        }
    )


if __name__ == "__main__":
    task_collect_train(data_catalog["base"].load())
