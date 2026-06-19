import numpy as np
from scipy.stats import wishart, norm
from sstudentt import SST

from typing import Annotated
from pytask import Product, task
from simulation.config import data_catalog, DF, SCENARIOS


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


def random_negative_gram(n=6, low=-0.20, high=-0.02, margin=0.1, rng=None):
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            val = rng.uniform(low, high)
            G[i, j] = val
            G[j, i] = val

    diag = np.sum(np.abs(G), axis=1) + margin
    np.fill_diagonal(G, diag)

    # G = A @ A.T
    A = np.linalg.cholesky(G)
    return A


scenario_id = 0
for dependence in ["positive", "negative"]:
    for skewness in ["positive", "negative"]:
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
            N = 50
            T = 20000
            S = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
            S = np.concat([S, np.identity(6)])
            A_pos = rng.normal(1, size=(6, 6))
            A_neg = random_negative_gram(6, rng=rng)
            wishart_scale_pos = A_pos @ A_pos.T + 0.01 * np.identity(6)
            wishart_scale_neg = A_neg @ A_neg.T + 0.01 * np.identity(6)
            mean = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
            out = []
            while len(out) < N:
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
                    dist = SST(0.0, 1.0, 5.0, DF)
                    skew_noise = dist.q(noise_cdf)
                else:
                    dist = SST(0.0, 1.0, 0.2, DF)
                    skew_noise = dist.q(noise_cdf)
                y = np.zeros((T + 2, 6), dtype=np.float64)
                # generate AR parameters
                ar = generate_stationary_ar2(6, rng=rng)
                ar1 = ar[:, 0]
                ar2 = ar[:, 1]
                for i in range(T):
                    y[i + 2, :] = ar1 * y[i + 1,] + ar2 * y[i,] + skew_noise[i,]
                y = y @ S.T
                out.append(y[-1000:, :])
            node.save(out)
