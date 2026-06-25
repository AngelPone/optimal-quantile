from simulation.config import (
    ALPHAs,
    SCENARIOS,
    TEST_WINDOWS,
    data_catalog,
    OUTPUT_SAMPLE_SIZE,
    A,
    scenario_seed,
)
import torch

from pytask import task, Product
from typing import Annotated
from forecopy import csrec, cstools
import numpy as np


def benchmarks(samples, A, resids):
    A = np.array(A)
    params = cstools(agg_mat=A)
    olss = []
    wlss = []
    shrs = []
    sams = []
    for i in range(len(resids)):
        ols = csrec(np.array(samples[i, :, :].T), params=params, res=resids[0].T)
        olss.append(ols.T)
        wls = csrec(
            np.array(samples[i, :, :].T), params=params, res=resids[0].T, comb="wls"
        )
        wlss.append(wls.T)
        shr = csrec(
            np.array(samples[i, :, :].T),
            params=params,
            res=resids[0].T,
            comb="shr",
        )
        shrs.append(shr.T)
        sam = csrec(
            np.array(samples[i, :, :].T),
            params=params,
            res=resids[0].T,
            comb="sam",
        )
        sams.append(sam.T)
    return {
        "ols": torch.as_tensor(np.stack(olss)),
        "wls": torch.as_tensor(np.stack(wlss)),
        "shr": torch.as_tensor(np.stack(shrs)),
        "sam": torch.as_tensor(np.stack(sams)),
    }


for scenario in SCENARIOS:
    for alpha in ALPHAs:
        seed = scenario_seed(scenario, offset=20_000)

        @task
        def task_collect(
            input_base: Annotated[list, data_catalog[f"simulation_base_{scenario}"]],
            input_rf: Annotated[
                dict, data_catalog[f"simulation_rf_{scenario}_{alpha}"]
            ],
            output: Annotated[dict, Product] = data_catalog[
                f"simulation_samples_{scenario}_{alpha}"
            ],
            seed: int = seed,
        ):
            generator = torch.Generator()
            generator.manual_seed(seed)
            mean = torch.stack(
                [window["mean"] for window in input_base[-TEST_WINDOWS:]],
            )
            samples = torch.stack(
                [
                    torch.stack(
                        [
                            series.sample(OUTPUT_SAMPLE_SIZE, generator=generator)
                            for series in window["dist"]
                        ]
                    )
                    for window in input_base[-TEST_WINDOWS:]
                ]
            )
            samples = mean[:, :, None] + samples

            G, d = input_rf["result"]

            S = torch.concat([A, torch.eye(6)]).to(mean.dtype)
            rf_samples = (
                torch.einsum("tnj,kn->tkj", samples, S @ G) + (S @ d)[None, :, None]
            )

            res = benchmarks(
                samples, A, [i["resid"] for i in input_base[-TEST_WINDOWS:]]
            )
            res["base"] = samples
            res["QOpt"] = rf_samples
            output.save(res)
