# from pytask import task, Product
# from typing import Annotated
# from M5_reduced.config import (
#     data_catalog,
#     ALPHAs,
#     BETAs,
#     LOGGING_PATH,
#     VERSION,
#     SAMPLE_SIZE,
#     VAL_SAMPLE_SIZE,
#     LR,
#     DISTS,
#     MAX_ITER,
#     INIT,
# )
# from opt_rec_quantile.model import QOptRec
# from utils import SkewStudentT

# import torch
# from torch.distributions import Normal

# for alpha in ALPHAs:
#     for idx, dist in enumerate(DISTS):
#         for beta in BETAs:
#             seed = VERSION + int(alpha * 1000) + beta * 10000 + idx

#             @task
#             def task_perform_reconciliation(
#                 train_data: Annotated[dict, data_catalog["train_data"]],
#                 alpha: float = [alpha],
#                 beta: int = beta,
#                 dist: str = dist,
#                 output: Annotated[dict, Product] = data_catalog[
#                     f"rf_{alpha}_{beta}_{dist}_insample"
#                 ],
#                 seed: int = seed,
#             ) -> None:

#                 torch.manual_seed(seed)

#                 mean = train_data["mean"][0]
#                 T, n = mean.shape
#                 all_slice = range(T)
#                 loc = train_data["in-sample"]["loc"]
#                 scale = train_data["in-sample"]["scale"]
#                 xi = train_data["in-sample"]["xi"]
#                 df = train_data["in-sample"]["df"]

#                 def sampling(dist: str, smp_slice, size: int = SAMPLE_SIZE[alpha[0]]):
#                     if dist == "normal":
#                         smps = (
#                             Normal(loc[smp_slice], scale[smp_slice])
#                             .sample((size,))
#                             .permute((1, 2, 0))
#                         )
#                     elif dist == "skew":
#                         smps = (
#                             SkewStudentT(
#                                 xi[smp_slice],
#                                 df[smp_slice],
#                                 loc[smp_slice],
#                                 scale[smp_slice],
#                             )
#                             .sample((size,))
#                             .permute((1, 2, 0))
#                         )
#                     else:
#                         raise ValueError(f"{dist} not supported")
#                     return mean[smp_slice, :, None] + smps

#                 train_slice = all_slice[: -28 * 3]
#                 val_slice = all_slice[-28 * 3 :]
#                 model_normal = QOptRec(
#                     train_data["A"],
#                     alpha=alpha,
#                     beta=beta,
#                     optimizer_kwargs={"lr": LR},
#                     optimizer_cls=torch.optim.Adam,
#                 )

#                 def select_source(source_indices, local_indices):
#                     if local_indices is None:
#                         return source_indices
#                     return [source_indices[int(i)] for i in local_indices]

#                 def train_sampling(local_indices=None):
#                     source_indices = select_source(train_slice, local_indices)
#                     return sampling(dist, source_indices)

#                 def val_sampling(local_indices=None):
#                     source_indices = select_source(val_slice, local_indices)
#                     return sampling(dist, source_indices, VAL_SAMPLE_SIZE)

#                 G_init = train_data["in-sample"]["G"][INIT]
#                 G, d = model_normal.train(
#                     train_data["y"][0][train_slice],
#                     train_sampling,
#                     G=G_init,
#                     weights=train_data["weights"],
#                     sampling_val=val_sampling,
#                     y_val=train_data["y"][0][val_slice],
#                     max_iter=MAX_ITER,
#                     log_dir=LOGGING_PATH
#                     / f"{VERSION}"
#                     / f"insample-{dist}-alpha{int(alpha[0]*1000)}"
#                     / f"beta{beta}",
#                 )
#                 output.save({"mdl": model_normal, "result": (G, d)})


# if __name__ == "__main__":
#     alpha = 0.165
#     beta = 100
#     dist = "skew"
#     idx = 0 if dist == "skew" else 1
#     seed = VERSION + int(alpha * 1000) + beta * 10000
#     task_perform_reconciliation(
#         data_catalog["train_data"].load(),
#         alpha=[alpha],
#         beta=100,
#         seed=seed,
#         dist=dist,
#     )
