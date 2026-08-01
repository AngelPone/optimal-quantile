from utils import ExpandingWindowIterator, mle_estimation_skewed_normal
from typing import Annotated
from M5_reduced.config import (
    data_catalog,
    WINDOW_S,
    FORECAST_HORIZON,
    DATA_OUTPUT_PATH,
)
from pytask import Product
from statsforecast.models import AutoETS
import numpy as np
import torch
from pathlib import Path
import pickle as pkl


def task_base_forecast(
    data_path: Path = DATA_OUTPUT_PATH,
    node: Annotated[list, Product] = data_catalog["base"],
):
    with open(data_path, "rb") as f:
        data = pkl.load(f)

    T = data["data"]["Total"][1].shape[0]
    output = []
    for train_slice, test_slice in ExpandingWindowIterator(
        T, WINDOW_S, FORECAST_HORIZON, var_h=True
    ):
        mean = []
        resids = []
        futures = []
        hist = []
        for series in data["names"]:
            train = data["data"][series][1].values[train_slice]
            hist.append(train)
            futures.append(data["data"][series][1].values[test_slice])
            mdl = AutoETS(season_length=7)
            mdl.fit(train)
            horizon = test_slice.stop - test_slice.start
            fcasts = mdl.predict(h=horizon)["mean"]
            resid = train - mdl.predict_in_sample()["fitted"]
            mean.append(fcasts[:, None])
            resids.append(resid[:, None])
        output.append(
            {
                "mean": np.concat(mean, axis=1),
                "resid": np.concat(resids, axis=1),
                "slice": (train_slice, test_slice),
            }
        )

    node.save(output)


if __name__ == "__main__":
    task_base_forecast()
