from utils import expanding_window, mle_estimation_skewed_dist, SkewStudentT
from typing import Annotated
from simulation.config import data_catalog, WINDOW_S, SCENARIOS, DEVICE, DTYPE
from pytask import task, Product
from statsforecast.models import ARIMA, AutoETS
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
                mdl = AutoETS()
                mdl.fit(train[:, i])
                fcasts = mdl.predict(h=1)["mean"]
                resid = train[:, i] - mdl.predict_in_sample()["fitted"]
                dist = mle_estimation_skewed_dist(resid)
                mean.append(fcasts)
                resids.append(resid)
                dists.append(dist)
            mean = np.concat(mean)
            resids = np.stack(resids)
            dists = SkewStudentT(
                torch.stack([i.xi for i in dists]).to(dtype=DTYPE, device=DEVICE),
                torch.stack([i.df for i in dists]).to(dtype=DTYPE, device=DEVICE),
                torch.stack([i.loc for i in dists]).to(dtype=DTYPE, device=DEVICE),
                torch.stack([i.scale for i in dists]).to(dtype=DTYPE, device=DEVICE),
            )
            output.append(
                {
                    "mean": torch.as_tensor(mean, dtype=DTYPE, device=DEVICE),
                    "dist": dists,
                    "resid": resids,
                }
            )
        node.save(output)
