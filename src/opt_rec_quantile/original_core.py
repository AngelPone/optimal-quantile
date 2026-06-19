"""Original notebook-style quantile reconciliation calculations.

These functions intentionally keep the NumPy logic from
``Simulation/Codes/Opt_For_Rec_GD.ipynb`` explicit. They are small wrappers
around the original calculation steps so tests can compare the reimplementation
against the notebook formulas without relying on PyTorch autograd.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LossGradientResult:
    y_tilde: np.ndarray
    q_tilde: np.ndarray
    loss: float
    gradients_d: np.ndarray
    gradients_G: np.ndarray


def quantile(x: np.ndarray, alpha: float) -> np.ndarray:
    idx = np.ceil(len(x[0, 0, :]) * alpha) - 1
    idx = idx.astype(int)
    x_tmp = np.sort(x)
    return x_tmp[:, :, idx]


def pinball_loss(x: np.ndarray, z: np.ndarray, alpha: float) -> np.ndarray:
    return np.maximum(alpha * (x - z), (1 - alpha) * (z - x))


def f(x: np.ndarray, z: np.ndarray, beta: float) -> np.ndarray:
    f_tmp = np.exp(beta * (x - z))
    return f_tmp


def y_tilde_var(
    x: np.ndarray,
    d_tmp: np.ndarray,
    G_tmp: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    m = S.shape[1]
    y_til = np.empty((x.shape[0], x.shape[1], x.shape[2]))
    for i in range(x.shape[0]):
        for t in range(x.shape[1]):
            for j in range(x.shape[2]):
                y_til[i, t, j] = (
                    (S @ (d_tmp + G_tmp @ x[:, t, j]).reshape(m, 1))
                    .reshape(x.shape[0], 1)[i]
                    .item()
                )
    return y_til


def f_U(data_tmp: np.ndarray, q: np.ndarray, alpha: float, beta: float) -> float:
    return sum(
        sum((alpha - 1) * (data_tmp - q) + np.log(1 + f(data_tmp, q, beta)) / beta)
    )


def f_U_ex(data_tmp: np.ndarray, q: np.ndarray, alpha: float) -> float:
    return sum(sum(pinball_loss(data_tmp, q, alpha)))


def par_y_tilde_d(x: np.ndarray, S: np.ndarray) -> np.ndarray:
    m = S.shape[1]
    par_d = np.empty((x.shape[0], x.shape[1], x.shape[2], m))
    for i in range(x.shape[0]):
        for k in range(m):
            par_d[i, :, :, k] = S[i, k]
    return par_d


def par_y_tilde_G(x: np.ndarray, S: np.ndarray) -> np.ndarray:
    m = S.shape[1]
    par_G = np.empty((x.shape[0], x.shape[1], x.shape[2], m, x.shape[0]))
    for i in range(x.shape[0]):
        for k in range(m):
            for ell in range(x.shape[0]):
                par_G[i, :, :, k, ell] = x[ell, :, :] * S[i, k]
    return par_G


def par2_f_L_y_z(x: np.ndarray, z: np.ndarray, beta: float) -> np.ndarray:
    return -beta * f(x, z, beta) / ((1 + f(x, z, beta)) ** 2)


def par2_f_L_z2(x: np.ndarray, z: np.ndarray, beta: float) -> np.ndarray:
    return sum(beta * f(x, z, beta) / ((1 + f(x, z, beta)) ** 2))


def par_q_tilde(y_tild: np.ndarray, q: np.ndarray, beta: float) -> np.ndarray:
    par_q_tilde_tmp = np.zeros(
        (
            y_tild.shape[0],
            y_tild.shape[1],
            y_tild.shape[0],
            y_tild.shape[1],
            y_tild.shape[2],
        )
    )
    for i in range(y_tild.shape[0]):
        for t in range(y_tild.shape[1]):
            par_sum = par2_f_L_z2(y_tild[i, t, :], q[i, t], beta)
            for j in range(y_tild.shape[2]):
                par_q_tilde_tmp[i, t, i, t, j] = -(
                    par2_f_L_y_z(y_tild[i, t, j], q[i, t], beta) / par_sum
                )
    return par_q_tilde_tmp


def par_F_U(
    data_tmp: np.ndarray,
    q: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    return (1 / (1 + f(data_tmp, q, beta)) - alpha).reshape(1, q.shape[0], q.shape[1])


def grad_f_U_com(
    data_tmp: np.ndarray,
    x: np.ndarray,
    q: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    return par_F_U(data_tmp, q, alpha, beta).reshape(
        (1, x.shape[0] * x.shape[1])
    ) @ par_q_tilde(x, q, beta).reshape(
        (x.shape[0] * x.shape[1], x.shape[0] * x.shape[1] * x.shape[2])
    )


def grad_f_U_d_par(x: np.ndarray, S: np.ndarray) -> np.ndarray:
    m = S.shape[1]
    return par_y_tilde_d(x, S).reshape((x.shape[0] * x.shape[1] * x.shape[2], m))


def grad_f_U_G_par(x: np.ndarray, S: np.ndarray) -> np.ndarray:
    m = S.shape[1]
    return par_y_tilde_G(x, S).reshape(
        (x.shape[0] * x.shape[1] * x.shape[2], m * x.shape[0])
    )


def grad_f_U_d(
    data1_tmp: np.ndarray,
    data2_tmp: np.ndarray,
    x: np.ndarray,
    q: np.ndarray,
    alpha: float,
    beta: float,
    S: np.ndarray,
) -> np.ndarray:
    m = S.shape[1]
    return (
        par_F_U(data1_tmp, q, alpha, beta).reshape((1, x.shape[0] * x.shape[1]))
        @ par_q_tilde(x, q, beta).reshape(
            (x.shape[0] * x.shape[1], x.shape[0] * x.shape[1] * x.shape[2])
        )
        @ par_y_tilde_d(data2_tmp, S).reshape((x.shape[0] * x.shape[1] * x.shape[2], m))
    ).reshape(m)


def grad_f_U_G(
    data1_tmp: np.ndarray,
    data2_tmp: np.ndarray,
    x: np.ndarray,
    q: np.ndarray,
    alpha: float,
    beta: float,
    S: np.ndarray,
) -> np.ndarray:
    m = S.shape[1]
    return (
        par_F_U(data1_tmp, q, alpha, beta).reshape((1, x.shape[0] * x.shape[1]))
        @ par_q_tilde(x, q, beta).reshape(
            (x.shape[0] * x.shape[1], x.shape[0] * x.shape[1] * x.shape[2])
        )
        @ par_y_tilde_G(data2_tmp, S).reshape(
            (x.shape[0] * x.shape[1] * x.shape[2], m * x.shape[0])
        )
    ).reshape((m, x.shape[0]))


def quantile_gradient_wrt_G(
    y_tild: np.ndarray,
    y_hat: np.ndarray,
    q: np.ndarray,
    beta: float,
    S: np.ndarray,
) -> np.ndarray:
    m = S.shape[1]
    return (
        par_q_tilde(y_tild, q, beta).reshape(
            (
                y_tild.shape[0] * y_tild.shape[1],
                y_tild.shape[0] * y_tild.shape[1] * y_tild.shape[2],
            )
        )
        @ grad_f_U_G_par(y_hat, S)
    ).reshape((y_tild.shape[0], y_tild.shape[1], m, y_tild.shape[0]))


def quantile_gradient_wrt_d(
    y_tild: np.ndarray,
    y_hat: np.ndarray,
    q: np.ndarray,
    beta: float,
    S: np.ndarray,
) -> np.ndarray:
    m = S.shape[1]
    return (
        par_q_tilde(y_tild, q, beta).reshape(
            (
                y_tild.shape[0] * y_tild.shape[1],
                y_tild.shape[0] * y_tild.shape[1] * y_tild.shape[2],
            )
        )
        @ grad_f_U_d_par(y_hat, S)
    ).reshape((y_tild.shape[0], y_tild.shape[1], m))


def loss_and_gradients(
    data1: np.ndarray,
    data2: np.ndarray,
    d: np.ndarray,
    G: np.ndarray,
    alpha: float,
    beta: float,
    S: np.ndarray,
) -> LossGradientResult:
    y_tilde = y_tilde_var(np.copy(data2), d, G, S)
    q_tilde = quantile(y_tilde, alpha)
    cm_tst = grad_f_U_com(data1, y_tilde, q_tilde, alpha, beta)
    gradients_d = (cm_tst @ grad_f_U_d_par(data2, S)).reshape(S.shape[1])
    gradients_G = (cm_tst @ grad_f_U_G_par(data2, S)).reshape(
        (S.shape[1], data2.shape[0])
    )
    return LossGradientResult(
        y_tilde=y_tilde,
        q_tilde=q_tilde,
        loss=f_U(data1, q_tilde, alpha, beta),
        gradients_d=gradients_d,
        gradients_G=gradients_G,
    )
