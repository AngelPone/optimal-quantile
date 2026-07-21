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
    BETAs,
)
from pytask import task, Product
from pathlib import Path
from opt_rec_quantile.loss import pinball_loss


def benchmarks(samples, A, resids, alpha):
    A = A.numpy()
    params = cstools(agg_mat=A)
    olss = []
    wlss = []
    shrs = []
    sams = []
    resids[np.abs(resids) > 10000] = resids.mean()
    resids_demean = resids - resids.mean(axis=0)[None, :]
    for i in range(samples.shape[0]):
        smps = samples[i, :, :].T.numpy()
        ols = csrec(smps, params=params, res=resids_demean, comb="ols")
        olss.append(ols.T)
        wls = csrec(smps, params=params, res=resids_demean, comb="wls")
        wlss.append(wls.T)
        shr = csrec(smps, params=params, res=resids_demean, comb="shr")
        shrs.append(shr.T)
        sam = csrec(smps, params=params, res=resids_demean, comb="sam")
        sams.append(sam.T)
    return [
        {
            "ols": torch.as_tensor(np.quantile(np.stack(olss), q=alpha, axis=2)),
            "wls": torch.as_tensor(np.quantile(np.stack(wlss), q=alpha, axis=2)),
            "shr": torch.as_tensor(np.quantile(np.stack(shrs), q=alpha, axis=2)),
            "sam": torch.as_tensor(np.quantile(np.stack(sams), q=alpha, axis=2)),
        }
        for alpha in alpha
    ]


for idx, dist in enumerate(["normal", "skewnormal"]):
    seed = 20260720 + idx
    rf = [
        {alpha: data_catalog[f"rf_{alpha}_{beta}_{dist}"] for alpha in ALPHAs}
        for beta in BETAs
    ]

    @task
    def task_collect(
        base: Annotated[dict, data_catalog["base"]],
        input_rf: Annotated[dict, rf],
        data_path: Path = DATA_OUTPUT_PATH,
        output: Annotated[Path, Product] = TABLES_PATH / f"M5_ets_{dist}.tex",
        output_df: Annotated[Path, Product] = TABLES_PATH / f"M5_ets_{dist}.csv",
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
        output_dict = {
            "method": [],
            "loss": [],
            "alpha": [],
            "h": [],
            "series_code": [],
        }

        hist = base[0]["hist"]
        hist = np.concat([hist, np.stack([i["future"][0,] for i in base[:-1]])])
        mod = [np.abs(np.diff(hist[:, i])).mean() for i in range(hist.shape[1])]

        for h in range(1, 29):
            mean = torch.stack([i["mean"][:, h - 1] for i in base])
            true_y = torch.as_tensor(
                np.stack([i["future"][h - 1, :] for i in base]),
                dtype=mean.dtype,
                device=mean.device,
            )
            smps = sampling(dist, test_slice)
            res = benchmarks(smps, A, resids, ALPHAs)

            for alpha_idx, alpha in enumerate(ALPHAs):
                res[alpha_idx]["base"] = torch.quantile(smps, q=alpha, dim=2)
                for idx, beta in enumerate(BETAs):
                    G, d = input_rf[idx][alpha]["result"]
                    d = d[:, 0]
                    rf_samples = (
                        torch.einsum("tnj,kn->tkj", smps, S @ G)
                        + (S @ d * 1000)[None, :, None]
                    )
                    q = torch.quantile(rf_samples, alpha, dim=2)
                    res[alpha_idx][f"QOpt($\\beta={beta}$)"] = q
                for method, q in res[alpha_idx].items():
                    loss = [
                        (
                            pinball_loss(true_y[test_slice, i] - q[0, i], alpha=alpha)
                            .detach()
                            .item()
                            / mod[i]
                        )
                        for i in range(q.numel())
                    ]
                    for i in range(len(loss)):
                        output_dict["method"].append(method)
                        output_dict["alpha"].append(alpha)
                        output_dict["loss"].append(loss[i])
                        output_dict["h"].append(h)
                        output_dict["series_code"].append(i)
        df = pd.DataFrame(output_dict)
        df.to_csv(output_df)
        df = df.groupby(["method", "alpha"]).mean()["loss"].reset_index()
        df = df.pivot(index="method", columns="alpha", values="loss")
        df.columns = [f"{i:.3f}" for i in df.columns]
        output.write_text(df.to_latex(float_format="%.3f", label=" ", caption="M5"))


if __name__ == "__main__":
    for idx, dist in enumerate(["normal", "skewnormal"]):
        seed = 20260720 + idx
        rf = [
            {
                alpha: data_catalog[f"rf_{alpha}_{beta}_{dist}"].load()
                for alpha in ALPHAs
            }
            for beta in BETAs
        ]
        base = data_catalog["base"].load()
        task_collect(
            base,
            rf,
            dist=dist,
            seed=seed,
            output=TABLES_PATH / f"M5_ets_{dist}.tex",
        )
