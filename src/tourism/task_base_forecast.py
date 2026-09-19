from utils import ExpandingWindowIterator, mle_estimation_skewed_dist
from typing import Annotated
from tourism.config import data_catalog, WINDOW_S, DTYPE, DEVICE
from pytask import Product
from statsforecast.models import AutoARIMA
import numpy as np
import torch


def task_base_forecast(
    input_data: Annotated[np.ndarray, data_catalog["tourism"]],
    node: Annotated[dict, Product] = data_catalog["base"],
):
    output = {
        "mean": [],
        "dist": {"xi": [], "df": [], "mean": [], "std": []},
        "resids": [],
    }
    output["true_y"] = torch.as_tensor(
        np.concat(
            [
                input_data[test_slice, :]
                for _, test_slice in ExpandingWindowIterator(
                    input_data.shape[0], WINDOW_S, 1
                )
            ]
        ),
        device=DEVICE,
        dtype=DTYPE,
    )
    for train_slice, _ in ExpandingWindowIterator(input_data.shape[0], WINDOW_S, 1):
        mean_f = []
        resids = []
        xi = []
        df = []
        train = input_data[train_slice, :]
        for i in range(input_data.shape[1]):
            mdl = AutoARIMA(season_length=12)
            mdl.fit(train[:, i])
            fcasts = mdl.predict(h=1)["mean"]
            resid = train[:, i] - mdl.predict_in_sample()["fitted"]
            dist = mle_estimation_skewed_dist(resid)
            xi.append(dist.xi)
            df.append(dist.df)
            mean_f.append(fcasts)
            resids.append(resid)
        mean_f = torch.as_tensor(np.concat(mean_f), device=DEVICE, dtype=DTYPE)
        resids = np.stack(resids).T

        output["dist"]["mean"].append(
            torch.as_tensor(resids.mean(axis=0), device=DEVICE, dtype=DTYPE)
        )
        output["dist"]["std"].append(
            torch.as_tensor(resids.std(axis=0), device=DEVICE, dtype=DTYPE)
        )
        output["dist"]["xi"].append(torch.as_tensor(xi, device=DEVICE, dtype=DTYPE))
        output["dist"]["df"].append(torch.as_tensor(df, device=DEVICE, dtype=DTYPE))
        output["mean"].append(mean_f)
        output["resids"].append(resids)

    for key in output["dist"].keys():
        output["dist"][key] = torch.stack(output["dist"][key])

    output["mean"] = torch.stack(output["mean"])

    node.save(output)


if __name__ == "__main__":
    task_base_forecast(data_catalog["tourism"].load())
