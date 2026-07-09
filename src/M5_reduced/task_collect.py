import numpy as np
import pickle as pkl
import pandas as pd
from forecopy import cstools, csrec
import torch
from typing import Annotated

from M5_reduced.config import (
    ALPHAs,
    data_catalog,
    OUTPUT_SAMPLE_SIZE,
    TABLES_PATH,
    DATA_OUTPUT_PATH,
)
from pytask import task, Product
from pathlib import Path
from opt_rec_quantile.loss import pinball_loss


def benchmarks(samples, A, resids):
    A = np.array(A)
    resids = np.array(resids)
    A = np.array(A)
    params = cstools(agg_mat=A)
    olss = []
    wlss = []
    shrs = []
    sams = []
    for i in range(samples.shape[0]):
        ols = csrec(np.array(samples[i, :, :].T), params=params, res=resids)
        olss.append(ols.T)
        wls = csrec(np.array(samples[i, :, :].T), params=params, res=resids, comb="wls")
        wlss.append(wls.T)
        shr = csrec(
            np.array(samples[i, :, :].T),
            params=params,
            res=resids,
            comb="shr",
        )
        shrs.append(shr.T)
        sam = csrec(
            np.array(samples[i, :, :].T),
            params=params,
            res=resids,
            comb="sam",
        )
        sams.append(sam.T)
    return {
        "ols": torch.as_tensor(np.stack(olss)),
        "wls": torch.as_tensor(np.stack(wlss)),
        "shr": torch.as_tensor(np.stack(shrs)),
        "sam": torch.as_tensor(np.stack(sams)),
    }


for idx, dist in enumerate(["normal", "skewnormal"]):
    seed = 20260706 + int(idx * 2000)

    rf = {alpha: data_catalog[f"rf_{alpha}_{dist}"] for alpha in ALPHAs}

    @task
    def task_collect(
        base: Annotated[dict, data_catalog["base"]],
        input_rf: Annotated[dict, rf],
        data_path: Path = DATA_OUTPUT_PATH,
        output: Annotated[Path, Product] = TABLES_PATH / f"M5_acc_{dist}.tex",
        dist: str = dist,
        seed: int = seed,
    ):
        generator = torch.Generator()
        generator.manual_seed(seed)

        with open(data_path, "rb") as f:
            S = torch.as_tensor(pkl.load(f)["S"], dtype=torch.float64)
        n, m = S.shape

        A = S[:-m, :]

        def sampling(dist: str, smp_slice, size: int = OUTPUT_SAMPLE_SIZE):

            if dist == "normal":
                smp_loc = (
                    torch.stack(
                        [
                            torch.stack([base[i]["normal"][j].loc for j in range(n)])
                            for i in smp_slice
                        ]
                    )
                    .unsqueeze(-1)
                    .expand(len(smp_slice), n, size)
                )
                smp_scale = (
                    torch.stack(
                        [
                            torch.stack([base[i]["normal"][j].scale for j in range(n)])
                            for i in smp_slice
                        ]
                    )
                    .unsqueeze(-1)
                    .expand(len(smp_slice), n, size)
                )
                smps = torch.normal(smp_loc, smp_scale, generator=generator)
            elif dist == "skewnormal":
                smps = torch.stack(
                    [
                        torch.stack(
                            [
                                base[i][dist][j].sample(size, generator=generator)
                                for j in range(n)
                            ]
                        )
                        for i in smp_slice
                    ]
                )
            return mean[smp_slice, :, None] + smps

        test_slice = range(len(base) - 1, len(base))
        resids = base[-1]["resid"]
        output_dict = {"method": [], "loss": [], "alpha": [], "h": []}
        for h in range(1, 29):
            mean = torch.stack([i["mean"][:, h - 1] for i in base])
            true_y = torch.as_tensor(
                np.stack([i["future"][h - 1, :] for i in base]),
                dtype=mean.dtype,
                device=mean.device,
            )
            smps = sampling(dist, test_slice)
            res = benchmarks(smps, A, resids)

            res["base"] = smps

            for alpha in ALPHAs:
                G, d = input_rf[alpha]["result"]
                rf_samples = (
                    torch.einsum("tnj,kn->tkj", smps, S @ G)
                    + (S @ d * 1000)[None, :, None]
                )
                res["QOpt"] = rf_samples
                for method, smp in res.items():
                    q = torch.quantile(smp, alpha, dim=2)
                    loss = (
                        pinball_loss(true_y[test_slice] - q, alpha=alpha)
                        .detach()
                        .item()
                    )
                    output_dict["method"].append(method)
                    output_dict["alpha"].append(alpha)
                    output_dict["loss"].append(loss)
                    output_dict["h"].append(h)
        df = pd.DataFrame(output_dict)
        df = df.groupby(["method", "alpha"]).mean()["loss"].reset_index()
        df = df.pivot(index="method", columns="alpha", values="loss")
        df.columns = [f"{i:.2f}" for i in df.columns]
        output.write_text(df.to_latex(float_format="%.2f", label=" ", caption="M5"))


if __name__ == "__main__":
    base = data_catalog["base"].load()
    rf = {alpha: data_catalog[f"rf_{alpha}_{dist}"].load() for alpha in ALPHAs}
    task_collect(base, rf, dist="normal")
