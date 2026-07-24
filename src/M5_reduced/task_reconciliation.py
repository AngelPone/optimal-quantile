from pytask import task, Product
from typing import Annotated
from M5_reduced.config import (
    data_catalog,
    ALPHAs,
    BETAs,
    LOGGING_PATH,
)
from opt_rec_quantile.model import QOptRec
from utils import SkewNormal

import torch
from torch.distributions import Normal

LR = 0.0001
SAMPLE_SIZE = {
    alpha: (3000 if alpha in [0.005, 0.025, 0.975, 0.995] else 500) for alpha in ALPHAs
}
VAL_SAMPLE_SIZE = 5000
MAX_ITER = 300
VERSION = 20260729

for alpha in ALPHAs:
    for idx, dist in enumerate(["normal"]):
        for beta in BETAs:
            seed = VERSION + int(alpha * 1000) + beta * 10000 + idx

            @task
            def task_perform_reconciliation(
                train_data: Annotated[dict, data_catalog["train_data"]],
                output: Annotated[dict, Product] = data_catalog[
                    f"rf_{alpha}_{beta}_{dist}"
                ],
                alpha: float = [alpha],
                beta: int = beta,
                dist: str = dist,
                seed: int = seed,
            ) -> None:

                torch.manual_seed(seed)

                all_slice = range(train_data["mean"].shape[0])
                normal_loc, normal_scale = train_data["normal"]
                sn_xi, sn_loc, sn_scale = train_data["skewnormal"]

                def sampling(dist: str, smp_slice, size: int = SAMPLE_SIZE[alpha[0]]):

                    if dist == "normal":
                        smps = (
                            Normal(normal_loc[smp_slice], normal_scale[smp_slice])
                            .sample((size,))
                            .permute((1, 2, 0))
                        )
                    elif dist == "skewnormal":
                        smps = (
                            SkewNormal(
                                sn_xi[smp_slice], sn_loc[smp_slice], sn_scale[smp_slice]
                            )
                            .sample((size,))
                            .permute((1, 2, 0))
                        )
                    return train_data["mean"][smp_slice, :, None] + smps

                train_slice = all_slice[:-28]
                val_slice = all_slice[-28 * 3 :]
                model_normal = QOptRec(
                    train_data["A"],
                    alpha=alpha,
                    beta=beta,
                    optimizer_kwargs={"lr": LR},
                    optimizer_cls=torch.optim.Adam,
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
                    return sampling(dist, source_indices, VAL_SAMPLE_SIZE)

                G, d = model_normal.train(
                    train_data["y"][train_slice],
                    train_sampling,
                    G=train_data["G_shr"],
                    weights=train_data["weights"],
                    sampling_val=val_sampling,
                    y_val=train_data["y"][val_slice],
                    max_iter=MAX_ITER,
                    log_dir=LOGGING_PATH
                    / f"alpha{int(alpha[0]*1000)}"
                    / f"beta{beta}"
                    / f"lr{int(LR*10000)}_{VERSION}",
                )
                output.save({"mdl": model_normal, "result": (G, d)})


if __name__ == "__main__":
    task_perform_reconciliation(
        data_catalog["train_data"].load(),
        alpha=[0.005],
        beta=100,
        seed=42,
        dist="normal",
    )
