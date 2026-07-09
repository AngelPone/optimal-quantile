from pytask import task, Product
from typing import Annotated
from M5_reduced.config import (
    data_catalog,
    ALPHAs,
    SAMPLE_SIZE,
    LR,
    DATA_OUTPUT_PATH,
    BETA,
)
from opt_rec_quantile.model import QOptRec

import torch
import numpy as np
import pickle as pkl
from pathlib import Path

for idx, alpha in enumerate(ALPHAs):

    seed = 20260706 + idx * 10

    @task
    def task_perform_reconciliation(
        base: Annotated[dict, data_catalog["base"]],
        data_path: Path = DATA_OUTPUT_PATH,
        output_skew: Annotated[dict, Product] = data_catalog[f"rf_{alpha}_skewnormal"],
        output_normal: Annotated[dict, Product] = data_catalog[f"rf_{alpha}_normal"],
        alpha: float = alpha,
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
        all_slice = range(0, len(base) - h)
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
            return (mean[smp_slice, :, None] + smps) / 1000

        train_slice = all_slice[: -28 * 2]
        val_slice = all_slice[-28 * 2 :]
        model_normal = QOptRec(A, alpha=alpha, beta=BETA, optimizer_kwargs={"lr": LR})

        def select_source(source_indices, local_indices):
            if local_indices is None:
                return source_indices
            return [source_indices[int(i)] for i in local_indices]

        def train_sampling(local_indices=None):
            source_indices = select_source(train_slice, local_indices)
            return sampling("normal", source_indices)

        def val_sampling(local_indices=None):
            source_indices = select_source(val_slice, local_indices)
            return sampling("normal", source_indices, 1000)

        G, d = model_normal.train(
            true_y[train_slice] / 1000,
            train_sampling,
            generator=generator,
            sampling_val=val_sampling,
            y_val=true_y[val_slice] / 1000,
            batch_size=256,
        )
        output_normal.save({"mdl": model_normal, "result": (G, d)})

        def train_sampling(local_indices=None):
            source_indices = select_source(train_slice, local_indices)
            return sampling("skewnormal", source_indices)

        def val_sampling(local_indices=None):
            source_indices = select_source(val_slice, local_indices)
            return sampling("skewnormal", source_indices, 1000)

        model_skew = QOptRec(A, alpha=alpha, beta=BETA, optimizer_kwargs={"lr": LR})
        G, d = model_skew.train(
            true_y[train_slice] / 1000,
            train_sampling,
            generator=generator,
            sampling_val=val_sampling,
            y_val=true_y[val_slice] / 1000,
            batch_size=256,
        )
        output_skew.save({"mdl": model_skew, "result": (G, d)})
        return
