from __future__ import annotations

import argparse
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from M5.config import DATA_OUTPUT_PATH

DEFAULT_LAGS = tuple(range(28, 43))
DEFAULT_ROLLING_WINDOWS = (7, 14, 30, 60, 180)
TARGET = "sales"
DAY_RE = re.compile(r"^d_(\d+)$")


@dataclass(frozen=True)
class LevelSpec:
    key: str
    label: str
    group_keys: tuple[str, ...]
    M5_level: str = None


LEVEL_SPECS = (
    LevelSpec("level1", "Total", (), "Level1"),
    LevelSpec("level2", "Category", ("cat_id",), "Level4"),
    LevelSpec("level3", "Department", ("dept_id",), "Level5"),
    LevelSpec("level4", "State", ("state_id",), "Level2"),
    LevelSpec("level5", "Department State", ("state_id", "dept_id"), "Level7"),
    LevelSpec("level6", "Category State", ("state_id", "cat_id"), "Level6"),
    LevelSpec("level7", "Store", ("store_id",), "Level3"),
    LevelSpec("level8", "Category Store", ("store_id", "cat_id"), "Level8"),
    LevelSpec("level9", "Department Store", ("store_id", "dept_id"), "Level9"),
)
BASE_LEVEL = LEVEL_SPECS[-1]


def day_number(column: str) -> int:
    match = DAY_RE.match(str(column))
    if match is None:
        raise ValueError(f"Not an M5 day column: {column!r}")
    return int(match.group(1))


def day_columns(df: pd.DataFrame) -> list[str]:
    columns = [col for col in df.columns if DAY_RE.match(str(col))]
    return sorted(columns, key=day_number)


def add_derived_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dept_id" not in out and "item_id" in out:
        out["dept_id"] = out["item_id"].str.rsplit("_", n=1).str[0]
    if "cat_id" not in out and "dept_id" in out:
        out["cat_id"] = out["dept_id"].str.rsplit("_", n=1).str[0]
    if "state_id" not in out and "store_id" in out:
        out["state_id"] = out["store_id"].str.rsplit("_", n=1).str[0]
    return out


def make_series_names(df: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series:
    if not keys:
        return pd.Series(["Total"] * len(df), index=df.index)
    return df.loc[:, list(keys)].astype(str).agg("-".join, axis=1)


def aggregate_sales(sales: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    sales = add_derived_keys(sales)
    days = day_columns(sales)
    if not days:
        raise ValueError("Sales data has no d_* columns.")

    if keys:
        grouped = (
            sales.groupby(list(keys), sort=True, observed=True)[days]
            .sum()
            .reset_index()
        )
    else:
        grouped = pd.DataFrame([{day: sales[day].sum() for day in days}])

    grouped.insert(0, "series_id", make_series_names(grouped, keys).to_numpy())
    return grouped


def build_composition_features(
    base_nodes: pd.DataFrame, keys: tuple[str, ...]
) -> pd.DataFrame:
    base = add_derived_keys(base_nodes)
    count_columns = {
        "bottom_series_count": ("base_series_id", "nunique"),
        "store_count": ("store_id", "nunique"),
        "dept_count": ("dept_id", "nunique"),
    }

    if keys:
        features = base.groupby(list(keys), sort=True, observed=True).agg(
            **count_columns
        )
        features = features.reset_index()
    else:
        features = pd.DataFrame(
            {
                "bottom_series_count": [base["base_series_id"].nunique()],
                "store_count": [base["store_id"].nunique()],
                "dept_count": [base["dept_id"].nunique()],
            }
        )
    return features


def build_hierarchy(
    train: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str], pd.DataFrame]:
    base = aggregate_sales(train, BASE_LEVEL.group_keys)
    base_nodes = base[["series_id", *BASE_LEVEL.group_keys]].rename(
        columns={"series_id": "base_series_id"}
    )
    base_nodes = add_derived_keys(base_nodes)

    rows = []
    names: list[str] = []
    levels: list[str] = []
    for spec in LEVEL_SPECS:
        nodes = aggregate_sales(train, spec.group_keys)
        nodes = add_derived_keys(nodes)
        for _, node in nodes.iterrows():
            mask = np.ones(len(base_nodes), dtype=bool)
            for key in spec.group_keys:
                mask &= base_nodes[key].to_numpy() == node[key]
            rows.append(mask.astype(np.int8))
            names.append(str(node["series_id"]))
            levels.append(spec.label)

    return np.vstack(rows).astype(np.int8), names, levels, base_nodes


def calendar_features(calendar: pd.DataFrame) -> pd.DataFrame:
    cal = calendar.copy()
    if "d" not in cal:
        raise ValueError("Calendar data must contain a 'd' column.")
    cal["d_num"] = cal["d"].map(day_number).astype(np.int16)

    if "date" in cal:
        date = pd.to_datetime(cal["date"], errors="coerce")
        cal["tm_d"] = date.dt.day.fillna(0).astype(np.int16)
        cal["tm_w"] = date.dt.isocalendar().week.fillna(0).astype(np.int16)
        cal["tm_m"] = date.dt.month.fillna(0).astype(np.int16)
        cal["tm_y"] = date.dt.year.fillna(0).astype(np.int16)
        cal["tm_dw"] = date.dt.dayofweek.fillna(0).astype(np.int16)
        cal["tm_w_end"] = (cal["tm_dw"] >= 5).astype(np.int8)
        cal = cal.drop(columns=["date"])

    cal = cal.drop(columns=[col for col in ["d"] if col in cal])
    object_cols = cal.select_dtypes(include=["object", "category"]).columns
    for col in object_cols:
        cal[col] = pd.Categorical(cal[col].fillna("missing")).codes.astype(np.int16)

    for col in cal.columns:
        if col == "d_num":
            continue
        cal[col] = pd.to_numeric(cal[col], errors="coerce")

    return cal.drop_duplicates("d_num").reset_index(drop=True)


def price_features(sell_prices: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    prices = add_derived_keys(sell_prices)
    if not {"wm_yr_wk", "sell_price"}.issubset(prices.columns):
        raise ValueError("Price data must contain 'wm_yr_wk' and 'sell_price'.")

    group_keys = [*keys, "wm_yr_wk"] if keys else ["wm_yr_wk"]
    features = (
        prices.groupby(group_keys, sort=True, observed=True)["sell_price"]
        .agg(["mean", "std", "min", "max", "nunique", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "sell_price_mean",
                "std": "sell_price_std",
                "min": "sell_price_min",
                "max": "sell_price_max",
                "nunique": "sell_price_nunique",
                "count": "active_price_count",
            }
        )
    )
    features["sell_price_std"] = features["sell_price_std"].fillna(0.0)
    return features


def melt_sales(wide: pd.DataFrame, keys: tuple[str, ...], split: str) -> pd.DataFrame:
    days = day_columns(wide)
    id_vars = ["series_id", *keys]
    long = wide[id_vars + days].melt(
        id_vars=id_vars,
        var_name="d",
        value_name=TARGET,
    )
    long["d_num"] = long["d"].map(day_number).astype(np.int16)
    long["split"] = split
    return long.drop(columns=["d"])


def add_time_series_features(
    panel: pd.DataFrame,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
) -> pd.DataFrame:
    out = panel.sort_values(["series_id", "d_num"]).reset_index(drop=True)
    grouped_sales = out.groupby("series_id", sort=False)[TARGET]
    lag_values = tuple(sorted(set(int(lag) for lag in lags)))
    if not lag_values:
        raise ValueError("At least one lag is required.")

    for lag in lag_values:
        out[f"sales_lag_{lag}"] = grouped_sales.transform(
            lambda series: series.shift(lag)
        )

    rolling_shift = min(lag_values)
    rolling_values = tuple(sorted(set(int(window) for window in rolling_windows)))
    for window in rolling_values:
        shifted = grouped_sales.transform(lambda series: series.shift(rolling_shift))
        out[f"rolling_mean_{window}"] = shifted.groupby(out["series_id"]).transform(
            lambda series: series.rolling(window, min_periods=1).mean()
        )
        out[f"rolling_std_{window}"] = shifted.groupby(out["series_id"]).transform(
            lambda series: series.rolling(window, min_periods=2).std()
        )

    return out


def add_level_features(
    train_wide: pd.DataFrame,
    test_wide: pd.DataFrame,
    spec: LevelSpec,
    calendar: pd.DataFrame,
    sell_prices: pd.DataFrame,
    base_nodes: pd.DataFrame,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
    level_code: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    panel = pd.concat(
        [
            melt_sales(train_wide, spec.group_keys, "train"),
            melt_sales(test_wide, spec.group_keys, "test"),
        ],
        ignore_index=True,
    )
    panel = add_time_series_features(panel, lags, rolling_windows)
    panel = panel.merge(calendar_features(calendar), on="d_num", how="left")

    merge_keys = [*spec.group_keys, "wm_yr_wk"] if spec.group_keys else ["wm_yr_wk"]
    panel = panel.merge(
        price_features(sell_prices, spec.group_keys), on=merge_keys, how="left"
    )
    composition = build_composition_features(base_nodes, spec.group_keys)
    if spec.group_keys:
        panel = panel.merge(composition, on=list(spec.group_keys), how="left")
    else:
        for col in composition.columns:
            panel[col] = composition[col].iloc[0]

    panel["series_code"] = pd.Categorical(panel["series_id"]).codes.astype(np.int16)
    for key in spec.group_keys:
        panel[f"{key}_code"] = pd.Categorical(panel[key]).codes.astype(np.int16)

    required_lags = [f"sales_lag_{int(lag)}" for lag in sorted(set(lags))]
    panel = panel.dropna(subset=required_lags).reset_index(drop=True)

    drop_columns = {
        TARGET,
        "split",
        "series_id",
        *spec.group_keys,
        "weekday",
        "wday",
        "wm_yr_wk",
        "month",
        "year",
    }

    feature_columns = [
        col
        for col in panel.columns
        if col not in drop_columns and np.issubdtype(panel[col].dtype, np.number)
    ]

    cat_cols = ["tm_d", "tm_w", "tm_m", "tm_y", "tm_dw", "tm_w_end", "series_code"]
    for col in cat_cols:
        panel[col] = panel[col].astype("category")

    train_panel = panel[panel["split"] == "train"].reset_index(drop=True)
    test_panel = panel[panel["split"] == "test"].reset_index(drop=True)
    X_train = train_panel[feature_columns].replace([np.inf, -np.inf], np.nan)
    X_test = test_panel[feature_columns].replace([np.inf, -np.inf], np.nan)

    missing_columns = sorted(
        set(X_train.columns[X_train.isna().any()]).union(
            X_test.columns[X_test.isna().any()]
        )
    )
    for col in missing_columns:
        X_train[f"{col}_missing"] = X_train[col].isna().astype(np.int8)
        X_test[f"{col}_missing"] = X_test[col].isna().astype(np.int8)

    y_train = train_panel[TARGET].to_numpy(dtype=np.float64)
    y_test = test_panel[TARGET].to_numpy(dtype=np.float64)
    return X_train, y_train, X_test, y_test


def prepare_m5_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sell_prices: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    lags: Iterable[int] = DEFAULT_LAGS,
    rolling_windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
) -> dict[str, object]:
    train = add_derived_keys(train)
    test = add_derived_keys(test)
    S, names, levels, base_nodes = build_hierarchy(train)
    prepared: dict[str, object] = {"S": S, "names": names, "levels": levels}

    for level_code, spec in enumerate(LEVEL_SPECS, start=1):
        train_wide = aggregate_sales(train, spec.group_keys)
        test_wide = aggregate_sales(test, spec.group_keys)
        prepared[spec.key] = add_level_features(
            train_wide,
            test_wide,
            spec,
            calendar,
            sell_prices,
            base_nodes,
            lags,
            rolling_windows,
            level_code,
        )

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            pickle.dump(prepared, file, protocol=pickle.HIGHEST_PROTOCOL)

    return prepared


def prepare_m5_data_from_files(
    data_dir: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    test = pd.read_csv(data_dir / "sales_test_evaluation.csv")
    sell_prices = pd.read_csv(data_dir / "sell_prices.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv")
    weights = pd.read_csv(data_dir / "weights_evaluation.csv")
    return prepare_m5_data(
        train,
        test,
        sell_prices,
        calendar,
        output_path=output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare M5 aggregate-level feature data."
    )
    parser.add_argument("--data-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_m5_data_from_files(
        args.data_dir,
        DATA_OUTPUT_PATH,
    )
    print(f"Saved M5 feature data to {DATA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
