# from pathlib import Path
# from typing import Annotated

# import pandas as pd
# import torch
# from pytask import Product, task
# from torch.utils.tensorboard import SummaryWriter
# from torch.distributions import Normal
# from utils import SkewStudentT

# from M5_reduced.config import (
#     DEVICE,
#     DTYPE,
#     LOGGING_PATH,
#     OUTPUT_SAMPLE_SIZE,
#     TABLES_PATH,
#     ALPHAs,
#     BETAs,
#     data_catalog,
#     VERSION,
#     DISTS,
# )


# def style(df, methods):
#     def format_value(x):
#         if x == minimum:
#             return rf"\textbf{{{x:.3f}}}"
#         if second_minimum is not None and x == second_minimum:
#             return rf"\textcolor{{red}}{{{x:.3f}}}"
#         return f"{x:.3f}"

#     for col in df.columns:
#         unique_values = df[col].dropna().unique()
#         unique_values.sort()
#         minimum = unique_values[0]
#         second_minimum = unique_values[1] if len(unique_values) > 1 else None
#         df[col] = [format_value(value) for value in df[col]]

#     target_rows = ["base", "QOpt($\\beta=1000$)"]
#     latex = df.loc[methods].to_latex(escape=False)
#     lines = latex.splitlines()
#     new_lines = []

#     for line in lines:
#         new_lines.append(line)

#         for target_row in target_rows:
#             if line.strip().startswith(target_row):
#                 new_lines.append(r"\midrule")

#     latex = "\n".join(new_lines)
#     return latex


# arima_benchmark = pd.DataFrame(
#     [
#         [
#             "ARIMA",
#             0.158,
#             0.148,
#             0.163,
#             0.147,
#             0.167,
#             0.170,
#             0.202,
#             0.178,
#             0.201,
#         ]
#     ],
#     columns=["method"] + [f"level{i}" for i in range(1, 10)],
# )

# for idx, dist in enumerate(DISTS):
#     seed = 20260720 + idx
#     rf = {
#         alpha: {
#             beta: data_catalog[f"rf_{alpha}_{beta}_{dist}_insample"] for beta in BETAs
#         }
#         for alpha in ALPHAs
#     }

#     @task
#     def task_collect(
#         input_rf: Annotated[dict, rf],
#         input_data: Annotated[dict, data_catalog["test_data"]],
#         output: Annotated[Path, Product] = TABLES_PATH / f"M5_ets_{dist}_insample.tex",
#         output_spl: Annotated[Path, Product] = TABLES_PATH
#         / f"M5_ets_spl_{dist}_insample.tex",
#         output_df: Annotated[Path, Product] = LOGGING_PATH
#         / f"M5_ets_{dist}_insample.csv",
#         dist: str = dist,
#         seed: int = seed,
#     ):
#         torch.manual_seed(seed)

#         T, n = input_data["mean"].shape
#         loc = input_data["in-sample"]["loc"].expand((T, n))
#         scale = input_data["in-sample"]["scale"].expand((T, n))
#         xi = input_data["in-sample"]["xi"].expand((T, n))
#         df = input_data["in-sample"]["df"].expand((T, n))

#         def sampling(dist: str, size: int = OUTPUT_SAMPLE_SIZE):
#             if dist == "normal":
#                 smps = Normal(loc, scale).sample((size,)).permute((1, 2, 0))
#             elif dist == "skew":
#                 smps = (
#                     SkewStudentT(xi, df, loc, scale).sample((size,)).permute((1, 2, 0))
#                 )
#             smps = smps.to(dtype=DTYPE, device=DEVICE)
#             return input_data["mean"][:, :, None] + smps

#         smps = sampling(dist)
#         A = input_data["A"]
#         S = torch.concat([A, torch.eye(A.shape[1], dtype=A.dtype, device=A.device)])

#         qf = {}
#         for alpha in ALPHAs:
#             qf[alpha] = {"base": torch.quantile(smps, alpha, dim=2)}

#         G_items = input_data["in-sample"]["G"]
#         for m, g in G_items.items():
#             rf_smps = torch.einsum("kn,TnJ->TkJ", S @ g, smps)
#             for alpha in ALPHAs:
#                 qf[alpha][m] = torch.quantile(rf_smps, alpha, dim=2)

#         for beta in BETAs:
#             for alpha in ALPHAs:
#                 g, d = input_rf[alpha][beta]["result"]
#                 rf_smps = (
#                     torch.einsum("kn,TnJ->TkJ", S @ g, smps)
#                     + (S @ d[:, 0])[None, :, None]
#                 )
#                 m = f"QOpt($\\beta={beta}$)"
#                 qf[alpha][m] = torch.quantile(rf_smps, alpha, dim=2)

#         def spl(x, alpha, weights):
#             loss = torch.where(x < 0, -(1 - alpha) * x, alpha * x)
#             return loss / weights

#         true_y = input_data["y"]
#         dfs = []
#         for alpha in ALPHAs:
#             for method, q in qf[alpha].items():
#                 loss = spl(true_y - q, alpha, input_data["weights"]).cpu().numpy()
#                 df = pd.DataFrame(loss.T)
#                 df.columns = [i for i in range(1, 29)]
#                 df["method"] = method
#                 df["alpha"] = alpha
#                 df["idx"] = range(S.shape[0])
#                 dfs.append(df)
#         dfs = pd.concat(dfs)
#         df = pd.melt(
#             dfs,
#             id_vars=["method", "alpha", "idx"],
#             value_vars=range(1, 29),
#             value_name="loss",
#             var_name="h",
#         )

#         df.to_csv(output_df)
#         df1 = (
#             df.groupby(["method", "alpha"])
#             .mean(numeric_only=True)["loss"]
#             .reset_index()
#         )
#         methods = (
#             ["base"]
#             + [f"QOpt($\\beta={beta}$)" for beta in BETAs]
#             + ["ols", "wls", "shr", "sam"]
#         )
#         df1 = df1.pivot(index="method", columns="alpha", values="loss")
#         df1.columns = [f"{i:.3f}" for i in df1.columns]
#         df1 = style(df1, methods)
#         output.write_text(df1)

#         df2 = (
#             df.merge(input_data["m5_weights"], on="idx", how="left")
#             .groupby(["method", "level", "idx"])[["loss", "weights"]]
#             .mean()
#             .reset_index()
#         )
#         df2["spl"] = df2["loss"] * df2["weights"]
#         df2 = (
#             df2.groupby(["method", "level"])[["spl"]]
#             .sum()
#             .reset_index()
#             .pivot(index="method", columns="level", values="spl")
#             .sort_index(axis=1)
#             .reset_index()
#         )
#         df2 = pd.concat([df2, arima_benchmark], ignore_index=True)
#         df2["Average"] = df2[[f"level{level}" for level in range(1, 10)]].mean(axis=1)
#         df2.set_index("method", inplace=True)
#         df2 = style(df2, ["ARIMA"] + methods)
#         output_spl.write_text(df2)

#         methods_n = (
#             ["base"] + [f"QOpt_{beta}" for beta in BETAs] + ["ols", "wls", "shr", "sam"]
#         )

#         df_alpha = df.groupby(["h", "method", "alpha"]).mean(numeric_only=True)["loss"]
#         df_m = df.groupby(["method", "alpha"]).mean(numeric_only=True)["loss"]
#         for idx, m in enumerate(methods):
#             writer = SummaryWriter(
#                 LOGGING_PATH
#                 / str(VERSION)
#                 / f"insample-{dist}-metrics"
#                 / methods_n[idx]
#             )
#             for idx, alpha in enumerate(ALPHAs):
#                 for h_ in range(1, 29):
#                     writer.add_scalar(
#                         f"test/by-h-alpha{int(alpha*1000)}",
#                         df_alpha[(h_, m, alpha)],
#                         h_ - 1,
#                     )
#                 writer.add_scalar("test/by-alpha", float(df_m[(m, alpha)]), idx)
#             writer.close()


# if __name__ == "__main__":
#     for idx, dist in enumerate(["skew", "normal"]):
#         seed = 20260720 + idx
#         rf = {
#             alpha: {
#                 beta: data_catalog[f"rf_{alpha}_{beta}_{dist}_insample"].load()
#                 for beta in BETAs
#             }
#             for alpha in ALPHAs
#         }
#         input_data = data_catalog["test_data"].load()
#         task_collect(rf, input_data, dist=dist, seed=seed)
