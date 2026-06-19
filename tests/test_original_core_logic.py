import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opt_rec_quantile import original_core
from opt_rec_quantile.loss import ApproxPinballLoss, approx_pinball_loss
from opt_rec_quantile.model import QOptRec


# true
def ref_quantile(x, alpha):
    idx = np.ceil(len(x[0, 0, :]) * alpha) - 1
    idx = idx.astype(int)
    x_tmp = np.sort(x)
    return x_tmp[:, :, idx]


# true
def ref_f(x, z, beta):
    f_tmp = np.exp(beta * (x - z))
    return f_tmp


# true
def ref_y_tilde_var(x, d_tmp, G_tmp, S):
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


# true
def ref_f_U(data_tmp, q, alpha, beta):
    return sum(
        sum((alpha - 1) * (data_tmp - q) + np.log(1 + ref_f(data_tmp, q, beta)) / beta)
    )


# true
def ref_par_y_tilde_d(x, S):
    m = S.shape[1]
    par_d = np.empty((x.shape[0], x.shape[1], x.shape[2], m))
    for i in range(x.shape[0]):
        for k in range(m):
            par_d[i, :, :, k] = S[i, k]
    return par_d


# pass
def ref_par_y_tilde_G(x, S):
    m = S.shape[1]
    par_G = np.empty((x.shape[0], x.shape[1], x.shape[2], m, x.shape[0]))
    for i in range(x.shape[0]):
        for k in range(m):
            for ell in range(x.shape[0]):
                par_G[i, :, :, k, ell] = x[ell, :, :] * S[i, k]
    return par_G


# true
def ref_par2_f_L_y_z(x, z, beta):
    return -beta * ref_f(x, z, beta) / ((1 + ref_f(x, z, beta)) ** 2)


# true
def ref_par2_f_L_z2(x, z, beta):
    return sum(beta * ref_f(x, z, beta) / ((1 + ref_f(x, z, beta)) ** 2))


# true
def ref_par_q_tilde(y_tild, q, beta):
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
            par_sum = ref_par2_f_L_z2(y_tild[i, t, :], q[i, t], beta)
            for j in range(y_tild.shape[2]):
                par_q_tilde_tmp[i, t, i, t, j] = -(
                    ref_par2_f_L_y_z(y_tild[i, t, j], q[i, t], beta) / par_sum
                )
    return par_q_tilde_tmp


# pass
def ref_par_F_U(data_tmp, q, alpha, beta):
    return (1 / (1 + ref_f(data_tmp, q, beta)) - alpha).reshape(
        1, q.shape[0], q.shape[1]
    )


# pass
def ref_grad_f_U_com(data_tmp, x, q, alpha, beta):
    return ref_par_F_U(data_tmp, q, alpha, beta).reshape(
        (1, x.shape[0] * x.shape[1])
    ) @ ref_par_q_tilde(x, q, beta).reshape(
        (x.shape[0] * x.shape[1], x.shape[0] * x.shape[1] * x.shape[2])
    )


# pass
def ref_grad_f_U_d_par(x, S):
    m = S.shape[1]
    return ref_par_y_tilde_d(x, S).reshape((x.shape[0] * x.shape[1] * x.shape[2], m))


# pass
def ref_grad_f_U_G_par(x, S):
    m = S.shape[1]
    return ref_par_y_tilde_G(x, S).reshape(
        (x.shape[0] * x.shape[1] * x.shape[2], m * x.shape[0])
    )


class TestOriginalCoreLogic(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20240618)
        self.S = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        self.alpha = 0.6
        self.beta = 2.5
        self.y = rng.normal(size=(3, 4))
        self.y_hat = rng.normal(size=(3, 4, 6))
        self.d = rng.normal(size=2)
        self.G = rng.normal(size=(2, 3))

    def test_quantile_gradient_wrt_G_matches_original_notebook_logic(self):
        y_tilde = ref_y_tilde_var(self.y_hat, self.d, self.G, self.S)
        q_tilde = ref_quantile(y_tilde, self.alpha)

        expected = (
            ref_par_q_tilde(y_tilde, q_tilde, self.beta).reshape(
                (
                    y_tilde.shape[0] * y_tilde.shape[1],
                    y_tilde.shape[0] * y_tilde.shape[1] * y_tilde.shape[2],
                )
            )
            @ ref_grad_f_U_G_par(self.y_hat, self.S)
        ).reshape((3, 4, 2, 3))

        actual = original_core.quantile_gradient_wrt_G(
            y_tilde, self.y_hat, q_tilde, self.beta, self.S
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_loss_and_gradients_match_original_notebook_logic(self):
        y_tilde = ref_y_tilde_var(self.y_hat, self.d, self.G, self.S)
        q_tilde = ref_quantile(y_tilde, self.alpha)
        cm_tst = ref_grad_f_U_com(self.y, y_tilde, q_tilde, self.alpha, self.beta)

        expected_loss = ref_f_U(self.y, q_tilde, self.alpha, self.beta)
        expected_grad_d = (cm_tst @ ref_grad_f_U_d_par(self.y_hat, self.S)).reshape(2)
        expected_grad_G = (cm_tst @ ref_grad_f_U_G_par(self.y_hat, self.S)).reshape(
            (2, 3)
        )

        actual = original_core.loss_and_gradients(
            self.y,
            self.y_hat,
            self.d,
            self.G,
            self.alpha,
            self.beta,
            self.S,
        )

        np.testing.assert_allclose(actual.y_tilde, y_tilde, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.q_tilde, q_tilde, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(actual.loss, expected_loss)
        np.testing.assert_allclose(
            actual.gradients_d, expected_grad_d, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            actual.gradients_G, expected_grad_G, rtol=1e-12, atol=1e-12
        )


class TestCurrentImplementationAgainstOriginalCore(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20240618)
        self.S = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        self.A = torch.as_tensor([[1.0, 1.0]], dtype=torch.float64)
        self.alpha = 0.6
        self.beta = 2.5
        self.y = rng.normal(size=(3, 4))
        self.y_hat = rng.normal(size=(3, 4, 6))
        self.d = rng.normal(size=2)
        self.G = rng.normal(size=(2, 3))

    def test_current_empirical_quantile_gradient_wrt_G_matches_original(self):
        S = torch.as_tensor(self.S, dtype=torch.float64)
        G = torch.as_tensor(self.G, dtype=torch.float64).requires_grad_(True)
        y_hat = torch.as_tensor(
            np.transpose(self.y_hat, (1, 0, 2)), dtype=torch.float64
        )

        q_current = ApproxPinballLoss.apply(S, G, y_hat, self.alpha, self.beta)
        q_current.mean().backward()

        y_tilde = original_core.y_tilde_var(
            self.y_hat, np.zeros(self.S.shape[1]), self.G, self.S
        )
        q_original = original_core.quantile(y_tilde, self.alpha)
        expected_local = original_core.quantile_gradient_wrt_G(
            y_tilde, self.y_hat, q_original, self.beta, self.S
        )
        expected = np.transpose(expected_local, (1, 0, 2, 3)).mean(axis=(0, 1))

        np.testing.assert_allclose(
            q_current.detach().numpy(), q_original.T, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            G.grad.detach().numpy(), expected, rtol=1e-12, atol=1e-12
        )

    def test_current_loss_helper_matches_original_core_loss_formula(self):
        q_tilde = original_core.quantile(
            original_core.y_tilde_var(self.y_hat, self.d, self.G, self.S), self.alpha
        )

        expected = original_core.f_U(self.y, q_tilde, self.alpha, self.beta) / (
            self.y.shape[0] * self.y.shape[1]
        )
        actual = approx_pinball_loss(
            torch.as_tensor(self.y.T - q_tilde.T, dtype=torch.float64),
            self.beta,
            self.alpha,
        ).item()

        self.assertAlmostEqual(actual, expected)

    def test_current_model_loss_and_gradients_wrt_G_and_d_match_original_core(self):
        model = QOptRec(A=self.A, alpha=self.alpha, beta=self.beta)
        y = torch.as_tensor(self.y.T, dtype=torch.float64)
        y_pred = torch.as_tensor(
            np.transpose(self.y_hat, (1, 0, 2)), dtype=torch.float64
        )
        G = torch.as_tensor(self.G, dtype=torch.float64).requires_grad_(True)
        d = torch.as_tensor(self.d, dtype=torch.float64).requires_grad_(True)

        actual_loss = model._loss(y, y_pred, G, d)
        actual_loss.backward()

        expected = original_core.loss_and_gradients(
            self.y,
            self.y_hat,
            self.d,
            self.G,
            self.alpha,
            self.beta,
            self.S,
        )
        normalizer = self.y.shape[0] * self.y.shape[1]

        self.assertAlmostEqual(actual_loss.detach().item(), expected.loss / normalizer)
        np.testing.assert_allclose(
            G.grad.detach().numpy(),
            expected.gradients_G / normalizer,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            d.grad.detach().numpy(),
            expected.gradients_d / normalizer,
            rtol=1e-12,
            atol=1e-12,
        )


class TestQOptRecTrainingLoop(unittest.TestCase):
    def test_train_returns_parameters_with_best_tracked_pinball_loss(self):
        class FixedStepOptimizer:
            def __init__(self, params, values):
                self.params = list(params)
                self.values = values
                self.step_idx = 0

            def zero_grad(self):
                for param in self.params:
                    if param.grad is not None:
                        param.grad.zero_()

            def step(self):
                G, d = self.params
                value = self.values[self.step_idx]
                self.step_idx += 1
                with torch.no_grad():
                    G.fill_(value)
                    d.fill_(value + 10.0)

        class LoopProbeQOptRec(QOptRec):
            def _loss(self, y, y_pred, G, d):
                return G.sum() * 0.0 + d.sum() * 0.0

            def _pinball_loss(self, y, y_pred, G, d):
                return torch.abs(G[0, 0] - 2.0)

        y = torch.zeros((1, 2), dtype=torch.float64)

        def sample():
            return torch.zeros((1, 2, 3), dtype=torch.float64)

        model = LoopProbeQOptRec(
            A=torch.as_tensor([[1.0]], dtype=torch.float64),
            alpha=0.6,
            beta=2.5,
            optimizer_cls=FixedStepOptimizer,
            optimizer_kwargs={"values": [3.0, 1.0, 2.0, 4.0]},
        )

        G, d = model.train(y, sample, max_iter=4)

        self.assertEqual(model.pinball_loss_history_, [1.0, 1.0, 0.0, 2.0])
        self.assertEqual(model.best_pinball_loss_, 0.0)
        torch.testing.assert_close(G, torch.full((1, 2), 2.0, dtype=torch.float64))
        torch.testing.assert_close(d, torch.full((1,), 12.0, dtype=torch.float64))
        torch.testing.assert_close(
            model.final_G_, torch.full((1, 2), 4.0, dtype=torch.float64)
        )
        torch.testing.assert_close(
            model.final_d_, torch.full((1,), 14.0, dtype=torch.float64)
        )


if __name__ == "__main__":
    unittest.main()
