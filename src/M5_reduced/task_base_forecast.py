from utils import FixedWindowIterator, mle_estimation_skewed_normal
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
    for train_slice, test_slice in FixedWindowIterator(T, WINDOW_S, FORECAST_HORIZON):
        mean = []
        resids = []
        skewnormal = []
        normals = []
        futures = []
        hist = []
        for series in data["names"]:
            train = data["data"][series][1].values[train_slice]
            hist.append(train)
            futures.append(data["data"][series][1].values[test_slice])
            X_train = data["data"][series][0].values[train_slice]
            X_pred = data["data"][series][0].values[test_slice]
            mdl = AutoETS(season_length=7)
            mdl.fit(train, X_train)
            fcasts = mdl.predict(h=FORECAST_HORIZON, X=X_pred)["mean"]
            resid = mdl.model_["residuals"]
            dist = mle_estimation_skewed_normal(resid)
            normal = torch.distributions.Normal(resid.mean(), resid.std())
            mean.append(fcasts)
            resids.append(resid)
            skewnormal.append(dist)
            normals.append(normal)
        output.append(
            {
                "mean": torch.as_tensor(mean, dtype=torch.float64),
                "normal": normals,
                "skewnormal": skewnormal,
                "resid": np.stack(resids, axis=1),
                "hist": np.stack(hist, axis=1),
                "future": np.stack(futures, axis=1),
            }
        )

    node.save(output)


if __name__ == "__main__":
    task_base_forecast()
