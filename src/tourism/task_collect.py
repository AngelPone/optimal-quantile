from tourism.config import (
    A,
    ALPHAs,
    OUTPUT_SAMPLE_SIZE,
    S,
    TEST_WINDOWS,
    data_catalog,
    tourism_seed,
)
import torch

from pytask import task, Product
from typing import Annotated, Any
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
        ols = csrec(np.array(samples[i, :, :].T), params=params, res=resids[i].T)
        olss.append(ols.T)
        wls = csrec(
            np.array(samples[i, :, :].T), params=params, res=resids[i].T, comb="wls"
        )
        wlss.append(wls.T)
        shr = csrec(
            np.array(samples[i, :, :].T),
            params=params,
            res=resids[i].T,
            comb="shr",
        )
        shrs.append(shr.T)
        sam = csrec(
            np.array(samples[i, :, :].T),
            params=params,
            res=resids[i].T,
            comb="sam",
        )
        sams.append(sam.T)
    return {
        "ols": torch.as_tensor(np.stack(olss)),
        "wls": torch.as_tensor(np.stack(wlss)),
        "shr": torch.as_tensor(np.stack(shrs)),
        "sam": torch.as_tensor(np.stack(sams)),
    }


for alpha in ALPHAs:
    for dist in ["normal", "skewt"]:
        seed = tourism_seed(offset=20_000)

        @task
        def task_collect(
            input_base: Annotated[list[dict[str, Any]], data_catalog["tourism_base"]],
            input_rf: Annotated[dict, data_catalog[f"tourism_rf_{alpha}_{dist}"]],
            output: Annotated[dict, Product] = data_catalog[
                f"tourism_samples_{alpha}_{dist}"
            ],
            dist: str = dist,
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
                            for series in window[dist]
                        ]
                    )
                    for window in input_base[-TEST_WINDOWS:]
                ]
            )
            samples = mean[:, :, None] + samples

            G, d = input_rf["result"]

            rf_samples = (
                torch.einsum("tnj,kn->tkj", samples, S @ G) + (S @ d)[None, :, None]
            )

            res = benchmarks(
                samples, A, [i["resid"] for i in input_base[-TEST_WINDOWS:]]
            )
            res["base"] = samples
            res["QOpt"] = rf_samples
            output.save(res)
