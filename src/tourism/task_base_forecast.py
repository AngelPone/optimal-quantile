from utils import expanding_window, mle_estimation_skewed_dist
from typing import Annotated
from tourism.config import data_catalog, DF, WINDOW_S
from pytask import task, Product
from statsforecast.arima import AutoARIMA
import numpy as np
import torch


@task
def task_base_forecast(
    input_data: Annotated[np.ndarray, data_catalog["tourism"]],
    node: Annotated[list, Product] = data_catalog["tourism_base"],
):
    output = []
    for train, _ in expanding_window(input_data.shape[0], WINDOW_S, 1, input_data):
        mean = []
        resids = []
        skewt = []
        normals = []
        for i in range(input_data.shape[1]):
            mdl = AutoARIMA()
            mdl.fit(train[:, i])
            fcasts = mdl.predict(h=1)["mean"]
            resid = train[:, i] - mdl.predict_in_sample()["fitted"]
            dist = mle_estimation_skewed_dist(resid, DF)
            normal = torch.distributions.Normal(torch.mean(resid), torch.std(resid))
            mean.append(fcasts)
            resids.append(resid)
            skewt.append(dist)
            normals.append(normal)
        mean = np.concat(mean)
        resids = np.stack(resids)
        output.append(
            {
                "mean": torch.as_tensor(mean, dtype=torch.float64),
                "normal": normals,
                "skewt": skewt,
                "resid": resids,
            }
        )
    node.save(output)
