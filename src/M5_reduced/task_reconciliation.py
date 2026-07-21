from pytask import task, Product
from typing import Annotated
from M5_reduced.config import (
    data_catalog,
    ALPHAs,
    DATA_OUTPUT_PATH,
    BETAs,
)
from forecopy import cscov, cstools
from opt_rec_quantile.model import QOptRec

import torch
import numpy as np
import pickle as pkl
from pathlib import Path

LR = 0.001
SAMPLE_SIZE = 100

for alpha in ALPHAs:
    for idx, dist in enumerate(["skewnormal", "normal"]):
        for beta in BETAs:
            seed = 20260706 + int(alpha * 1000) + beta * 10000 + idx

            @task
            def task_perform_reconciliation(
                base: Annotated[dict, data_catalog["base"]],
                data_path: Path = DATA_OUTPUT_PATH,
                output: Annotated[dict, Product] = data_catalog[
                    f"rf_{alpha}_{beta}_{dist}"
                ],
                alpha: float = [alpha],
                beta: int = beta,
                seed: int = seed,
            ) -> None:

                with open(data_path, "rb") as f:
                    data = pkl.load(f)

                generator = torch.Generator()
                generator.manual_seed(seed)

                S = data["S"]
                m = S.shape[1]
                A = torch.as_tensor(S[:-m, :], dtype=torch.float64)

                h = 1
                # for h in range(1, FORECAST_HORIZON + 1):
                all_slice = range(len(base) - h - 56 * 7, len(base) - h)
                mean = torch.stack([i["mean"][:, h - 1] for i in base])
                true_y = torch.as_tensor(
                    np.stack([i["future"][h - 1, :] for i in base]),
                    dtype=mean.dtype,
                    device=mean.device,
                )
                n = mean.shape[1]

                def sampling(dist: str, smp_slice, size: int = SAMPLE_SIZE):

                    if dist == "normal":
                        smp_loc = (
                            torch.stack(
                                [
                                    torch.stack(
                                        [base[i]["normal"][j].loc for j in range(n)]
                                    )
                                    for i in smp_slice
                                ]
                            )
                            .unsqueeze(-1)
                            .expand(len(smp_slice), n, size)
                        )
                        smp_scale = (
                            torch.stack(
                                [
                                    torch.stack(
                                        [base[i]["normal"][j].scale for j in range(n)]
                                    )
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
                                        base[i][dist][j].sample(
                                            size, generator=generator
                                        )
                                        for j in range(n)
                                    ]
                                )
                                for i in smp_slice
                            ]
                        )
                    return (mean[smp_slice, :, None] + smps) / 1000

                train_slice = all_slice[:-28]
                val_slice = all_slice[-28:]
                model_normal = QOptRec(
                    A, alpha=alpha, beta=beta, optimizer_kwargs={"lr": LR}
                )

                def select_source(source_indices, local_indices):
                    if local_indices is None:
                        return source_indices
                    return [source_indices[int(i)] for i in local_indices]

                def train_sampling(local_indices=None):
                    source_indices = select_source(train_slice, local_indices)
                    return sampling(dist, source_indices)

                def val_sampling(local_indices=None):
                    source_indices = select_source(val_slice, local_indices)
                    return sampling(dist, source_indices, 1000)

                resids = base[-1]["resid"]
                resids = resids - resids.mean(axis=0)
                with open(data_path, "rb") as f:
                    S = torch.as_tensor(pkl.load(f)["S"], dtype=torch.float64)
                n, m = S.shape
                A = S[:-m, :]
                params = cstools(agg_mat=A)
                W = cscov(params, res=resids).fit(comb="shr")
                W = torch.as_tensor(np.linalg.inv(W), dtype=torch.float64)
                shr_mat = torch.linalg.solve(S.T @ W @ S, S.T @ W)

                G, d = model_normal.train(
                    true_y[train_slice] / 1000,
                    train_sampling,
                    G=shr_mat,
                    generator=generator,
                    sampling_val=val_sampling,
                    y_val=true_y[val_slice] / 1000,
                )
                output.save({"mdl": model_normal, "result": (G, d)})

                return

        if __name__ == "__main__":
            task_perform_reconciliation(base=data_catalog["base"].load(), seed=seed)
