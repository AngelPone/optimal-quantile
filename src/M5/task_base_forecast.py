from __future__ import annotations

import pickle
from pathlib import Path
from typing import Annotated, Any, Iterable, Mapping
from itertools import product
import numpy as np
import pandas as pd
from pytask import Product, task
from lightgbm import LGBMRegressor

from M5.config import OUTPUT_PATH, data_catalog, VALIDATION_DAYS

LEVEL_KEYS = tuple(f"level{i}" for i in range(1, 10))
POINT_MODEL_PARAMS: dict[str, Any] = {
    "boosting_type": "gbdt",
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "subsample": 0.6,
    "subsample_freq": 1,
    "feature_fraction": 0.6,
    "max_bin": 50,
    "n_estimators": 1000,
    "boost_from_average": False,
    "verbose": -1,
    "num_threads": 1,
    "n_jobs": 2,
    "random_state": 42,
}

DEFAULT_TUNING_GRID = [
    {"max_depth": depth, "learning_rate": lr}
    for (depth, lr) in product(range(3, 11), [0.5, 0.1, 0.05, 0.001, 0.005, 0.01])
]


def safe_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return 1e-8
    scale = float(np.std(values, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        return 1e-8
    return scale


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def split_last_validation_days(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    validation_days: int = 28,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    if "d_num" not in X_train:
        raise ValueError(
            "X_train must contain a 'd_num' column for chronological split."
        )
    if validation_days <= 0:
        raise ValueError("validation_days must be positive.")
    unique_days = np.sort(pd.unique(X_train["d_num"]))
    if unique_days.size < 2:
        raise ValueError("Need at least two training days to hold out validation data.")
    n_validation_days = min(validation_days, unique_days.size - 1)
    validation_day_set = set(unique_days[-n_validation_days:])
    valid_mask = X_train["d_num"].isin(validation_day_set).to_numpy()
    train_mask = ~valid_mask

    return (
        X_train.loc[train_mask].reset_index(drop=True),
        np.asarray(y_train, dtype=np.float64)[train_mask],
        X_train.loc[valid_mask].reset_index(drop=True),
        np.asarray(y_train, dtype=np.float64)[valid_mask],
    )


def reshape_panel_values(
    X: pd.DataFrame,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "series_code" not in X or "d_num" not in X:
        raise ValueError("X must contain 'series_code' and 'd_num' columns.")
    frame = pd.DataFrame(
        {
            "series_code": X["series_code"].to_numpy(dtype=np.float64),
            "d_num": X["d_num"].to_numpy(dtype=np.float64),
            "value": np.asarray(values, dtype=np.float64),
        }
    )
    matrix = (
        frame.pivot(index="series_code", columns="d_num", values="value")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if matrix.isna().any().any():
        raise ValueError("Cannot reshape incomplete series/time panel.")
    return (
        matrix.to_numpy(dtype=np.float64),
        matrix.index.to_numpy(dtype=np.float64),
        matrix.columns.to_numpy(dtype=np.float64),
    )


def candidate_params(
    base_params: Mapping[str, Any] | None,
    param_grid: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    base = dict(POINT_MODEL_PARAMS if base_params is None else base_params)
    overrides = DEFAULT_TUNING_GRID if param_grid is None else tuple(param_grid)
    out = []
    for override in overrides:
        params = base.copy()
        params.update(override)
        out.append(params)
    return out


def tune_level_model(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_params: Mapping[str, Any] | None = None,
    param_grid: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    tuning = []
    best: dict[str, Any] | None = None
    for params in candidate_params(base_params, param_grid):
        model = LGBMRegressor(**params)
        model.fit(
            X_fit.drop(columns=["d_num"]),
            y_fit,
            eval_set=[(X_val.drop(columns=["d_num"]), y_val)],
            eval_metric="rmse",
        )
        pred = np.asarray(
            model.predict(X_val.drop(columns=["d_num"])), dtype=np.float64
        )
        score = rmse(y_val, pred)
        record = {"params": params, "rmse": score}
        tuning.append(record)
        if best is None or score < best["rmse"]:
            best = {"params": params, "rmse": score, "model": model, "pred": pred}

    return {
        "best_params": best["params"],
        "best_rmse": best["rmse"],
        "validation_pred": best["pred"],
        "tuning": tuning,
    }


def fit_level_forecast(
    level_data: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
    validation_days: int = 28,
    base_params: Mapping[str, Any] | None = None,
    param_grid: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    X_train, y_train, X_test, y_test = level_data
    X_fit, y_fit, X_val, y_val = split_last_validation_days(
        X_train, y_train, validation_days=validation_days
    )

    # tune the model
    tuned = tune_level_model(
        X_fit,
        y_fit,
        X_val,
        y_val,
        base_params=base_params,
        param_grid=param_grid,
    )

    # use ten-fold to produce out-of-sample residuals
    perm = np.random.permutation(range(0, X_train.shape[0]))
    val_len = perm.shape[0] // 5
    resids = np.empty(y_train.shape[0], dtype=np.float64)
    for i in range(0, perm.shape[0], val_len):
        val_idx = perm[i : (i + val_len)]
        train_idx = np.concatenate([perm[:i], perm[(i + val_len) :]])
        X_insample = X_train.iloc[train_idx].drop(columns=["d_num"])
        X_outsample = X_train.iloc[val_idx].drop(columns=["d_num"])
        y_insample = y_train[train_idx]
        y_outsample = y_train[val_idx]

        model = LGBMRegressor(**tuned["best_params"])
        model.fit(X_insample, y_insample)

        pred = np.asarray(model.predict(X_outsample), dtype=np.float64)
        resids[val_idx] = y_outsample - pred

    # use all data to produce out-of-sample forecasts on final 28-days
    # test data
    test_model = LGBMRegressor(**tuned["best_params"])
    test_model.fit(X_train.drop(columns=["d_num"]), y_train)
    test_pred = np.asarray(
        test_model.predict(X_test.drop(columns=["d_num"])), dtype=np.float64
    )

    # reshape the output to T*n_k
    resids, _, _ = reshape_panel_values(X_train, resids)
    future, _, _ = reshape_panel_values(X_test, y_test)
    hist, _, _ = reshape_panel_values(X_train, y_train)
    test_yhat, _, _ = reshape_panel_values(X_test, test_pred)

    return {
        "best_params": tuned["best_params"],
        "tuning": tuned["tuning"],
        "resids": resids,
        "y_hist": hist,
        "y_future": future,
        "y_pred": test_yhat,
    }


def fit_m5_base_forecasts(
    prepared: Mapping[str, Any],
    *,
    levels: Iterable[str] = LEVEL_KEYS,
    validation_days: int = 28,
    base_params: Mapping[str, Any] | None = None,
    param_grid: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "S": prepared["S"],
        "names": prepared["names"],
        "levels": prepared["levels"],
        "resids": [],
        "y_hist": [],
        "y_future": [],
        "y_pred": [],
        "models": [],
    }
    for level in levels:
        if level not in prepared:
            continue

        fit = fit_level_forecast(
            prepared[level],
            validation_days=validation_days,
            base_params=base_params,
            param_grid=param_grid,
        )
        for key in ["resids", "y_hist", "y_future", "y_pred"]:
            output[key].append(fit[key])
        output["models"].append(
            {"tuning": fit["tuning"], "best_params": fit["best_params"]}
        )
    for key in ["resids", "y_hist", "y_future", "y_pred"]:
        output[key] = np.concatenate(output[key]).T
    return output


@task
def task_base_forecast(
    prepared_path: Path = OUTPUT_PATH / "prepared.pkl",
    output_path: Annotated[Path, Product] = data_catalog["base"],
) -> None:
    with Path(prepared_path).open("rb") as file:
        prepared = pickle.load(file)
    output = fit_m5_base_forecasts(prepared, validation_days=VALIDATION_DAYS)
    output_path.save(output)


if __name__ == "__main__":
    task_base_forecast()
