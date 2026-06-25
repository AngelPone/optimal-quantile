from utils import expanding_window, mle_estimation_skewed_dist
from typing import Annotated
from simulation.config import data_catalog, DF, WINDOW_S, SCENARIOS
from pytask import task, Product
from statsforecast.models import ARIMA
import numpy as np
import torch

for scenario in SCENARIOS:

    @task
    def task_base_forecast(
        input: Annotated[np.ndarray, data_catalog[f"simulation_{scenario}"]],
        node: Annotated[list, Product] = data_catalog[f"simulation_base_{scenario}"],
    ):
        input = input["y"]
        output = []
        for train, _ in expanding_window(input.shape[0], WINDOW_S, 1, input):
            mean = []
            resids = []
            dists = []
            for i in range(input.shape[1]):
                mdl = ARIMA(order=(2, 0, 0))
                mdl.fit(train[:, i])
                fcasts = mdl.predict(h=1)["mean"]
                resid = train[:, i] - mdl.predict_in_sample()["fitted"]
                dist = mle_estimation_skewed_dist(resid, DF)
                mean.append(fcasts)
                resids.append(resid)
                dists.append(dist)
            mean = np.concat(mean)
            resids = np.stack(resids)
            output.append(
                {
                    "mean": torch.as_tensor(mean, dtype=torch.float64),
                    "dist": dists,
                    "resid": resids,
                }
            )
        node.save(output)
