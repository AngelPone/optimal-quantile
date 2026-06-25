import numpy as np
from scipy.stats import wishart, norm
from sstudentt import SST

from typing import Annotated
from pytask import Product, task
from simulation.config import data_catalog, DF, SCENARIOS, SKEWNESS_POS, SKEWNESS_NEG


def cov_to_cor(mat):
    sigma = np.sqrt(np.diag(mat)).reshape((-1, mat.shape[0]))
    cor = sigma.T @ sigma
    return mat / cor


def generate_stationary_ar2(n=1, margin=1e-4, rng=None):

    kappa1 = rng.uniform(-1 + margin, 1 - margin, size=n)
    kappa2 = rng.uniform(-1 + margin, 1 - margin, size=n)

    phi2 = kappa2
    phi1 = kappa1 * (1 - kappa2)

    return np.column_stack([phi1, phi2])


scenario_id = 0
for skewness in ["positive", "negative"]:
    for dependence in ["positive", "negative"]:
        scenario_id += 1
        scenario = SCENARIOS[scenario_id - 1]

        @task
        def task_draw_samples(
            node: Annotated[dict, Product] = data_catalog[f"simulation_{scenario}"],
            seed: int = 142 + scenario_id,
            dependence: str = dependence,
            skewness: str = skewness,
        ) -> None:
            rng = np.random.default_rng(seed)
            T = 20000
            m = 6
            S = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
            S = np.concat([S, np.identity(6)])
            wishart_scale_pos = np.full((m, m), 0.8)
            np.fill_diagonal(wishart_scale_pos, 5)
            wishart_scale_neg = np.full((m, m), -0.8)
            np.fill_diagonal(wishart_scale_neg, 5)
            mean = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
            out = None
            while True:
                if dependence == "positive":
                    cov_mat = wishart.rvs(100, wishart_scale_pos, random_state=rng)
                else:
                    cov_mat = wishart.rvs(100, wishart_scale_neg, random_state=rng)
                if dependence == "positive" and not np.all(cov_mat > 0):
                    continue
                elif dependence == "negative" and not np.all(
                    cov_mat[~np.eye(6, dtype=bool)] < 0
                ):
                    continue
                noise = rng.multivariate_normal(mean, cov_to_cor(cov_mat), size=T)
                noise_cdf = norm.cdf(noise)
                if skewness == "positive":
                    dist = SST(0.0, 10, SKEWNESS_POS, DF)
                    skew_noise = dist.q(noise_cdf)
                else:
                    dist = SST(0.0, 10, SKEWNESS_NEG, DF)
                    skew_noise = dist.q(noise_cdf)
                y = np.zeros((T + 2, 6), dtype=np.float64)
                # generate AR parameters
                ar = generate_stationary_ar2(6, rng=rng)
                ar1 = ar[:, 0]
                ar2 = ar[:, 1]
                for i in range(T):
                    y[i + 2, :] = ar1 * y[i + 1,] + ar2 * y[i,] + skew_noise[i,]
                y = y @ S.T
                out = y[-1000:, :]
                break
            node.save({"cov": cov_mat, "y": out, "ar": ar})
