from pytask import task, Product
from typing import Annotated
from M5.config import data_catalog, ALPHAs, SAMPLE_SIZE, LR
from opt_rec_quantile.model import QOptRec
from utils import mle_estimation_skewed_normal

import torch

for idx, alpha in enumerate(ALPHAs):
    for idx_dist, dist in enumerate(["normal", "skewnormal"]):
        seed = 20260806 + idx * 10 + idx_dist

        @task
        def task_perform_reconciliation(
            base: Annotated[dict, data_catalog["base"]],
            output: Annotated[dict, Product] = data_catalog[f"rf_{alpha}_{dist}"],
            dist: str = dist,
            alpha: float = [alpha],
            seed: int = seed,
        ) -> None:

            resids = torch.as_tensor(base["resids"], dtype=torch.float64)
            y_hist = torch.as_tensor(base["y_hist"], dtype=torch.float64)  # true
            y_pred = y_hist - resids  # mean

            S = torch.as_tensor(base["S"], dtype=torch.float64)
            n, m = S.shape
            A = S[:-m, :]
            assert torch.allclose(y_hist[:, -m:] @ A.T, y_hist[:, :-m])

            generator = torch.Generator(device=y_hist.device)
            generator.manual_seed(seed)

            skewnormal = [mle_estimation_skewed_normal(resids[:, i]) for i in range(n)]

            def sampling(slice, J=SAMPLE_SIZE):
                T = len(slice)
                if dist == "normal":
                    resids_mean = (
                        torch.mean(resids, dim=0).unsqueeze(-1).expand(T, n, J)
                    )
                    resids_std = torch.std(resids, dim=0).unsqueeze(-1).expand(T, n, J)
                    smp = torch.normal(resids_mean, resids_std, generator=generator)
                    return smp + y_pred[slice, :, None]
                elif dist == "skewnormal":
                    smp = [skewnormal[i].sample((T, J)) for i in range(n)]
                    smp = torch.stack(smp, dim=1)
                    return smp + y_pred[slice, :, None]
                return None

            all_slice = range(resids.shape[0])
            train_slice = all_slice[-52 * 7 * 2 - 28 : -28]
            val_slice = all_slice[-28:]

            def select_source(source_indices, local_indices):
                if local_indices is None:
                    return source_indices
                return [source_indices[int(i)] for i in local_indices]

            def train_sampling(local_indices=None):
                source_indices = select_source(train_slice, local_indices)
                return sampling(source_indices)

            def val_sampling(local_indices=None):
                source_indices = select_source(val_slice, local_indices)
                return sampling(source_indices, 1000)

            model = QOptRec(A, alpha=alpha, beta=100, optimizer_kwargs={"lr": LR})
            G, d = model.train(
                y_hist[train_slice],
                train_sampling,
                generator=generator,
                batch_size=52 * 7,
                sampling_val=val_sampling,
                y_val=y_hist[val_slice],
                max_iter=100,
            )
            output.save({"mdl": model, "result": (G, d)})
            return


if __name__ == "__main__":
    task_perform_reconciliation(
        data_catalog["base"].load(), dist="normal", alpha=[0.05]
    )
