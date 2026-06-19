import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sstudentt import SST

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import SkewStudentT


class TestSkewStudentT(unittest.TestCase):
    def setUp(self):
        self.mu = 1.0
        self.scale = 0.3
        self.xi = 0.2
        self.df = 5.0
        self.expected = SST(self.mu, self.scale, self.xi, self.df)
        self.actual = SkewStudentT(self.xi, self.df, self.mu, self.scale)

    def assert_torch_matches_sst(self, actual, expected):
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            np.asarray(expected),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_quantile_matches_sstudentt_sst(self):
        probabilities = torch.tensor(
            [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99],
            dtype=torch.float64,
        )

        actual_quantiles = self.actual.q(probabilities)
        expected_quantiles = self.expected.q(probabilities.numpy())

        self.assert_torch_matches_sst(actual_quantiles, expected_quantiles)

    def test_density_matches_sstudentt_sst(self):
        values = torch.tensor([0.5, 0.8, 1.0, 1.2, 1.5], dtype=torch.float64)

        actual_density = self.actual.prob(values)
        expected_density = self.expected.d(values.numpy())

        self.assert_torch_matches_sst(actual_density, expected_density)
        self.assert_torch_matches_sst(
            self.actual.log_prob(values),
            np.log(expected_density),
        )

    def test_sample_uses_same_inverse_cdf_as_sstudentt_sst(self):
        uniforms = torch.tensor(
            [[0.05, 0.25, 0.50, 0.95], [0.10, 0.40, 0.70, 0.90]],
            dtype=torch.float64,
        )

        with patch("torch.rand", return_value=uniforms):
            samples = self.actual.sample((2, 4))

        expected_samples = self.expected.q(uniforms.numpy())
        self.assert_torch_matches_sst(samples, expected_samples)
        self.assertEqual(samples.shape, uniforms.shape)
        self.assertEqual(samples.dtype, uniforms.dtype)

    def test_sample_clamps_boundary_uniforms_to_finite_quantiles(self):
        uniforms = torch.tensor(
            [[0.0, 1.0], [torch.finfo(torch.float64).tiny, 1.0]],
            dtype=torch.float64,
        )

        with patch("torch.rand", return_value=uniforms):
            samples = self.actual.sample((2, 2))

        self.assertTrue(torch.isfinite(samples).all())

    def test_rejects_invalid_scale_and_degrees_of_freedom(self):
        with self.assertRaises(ValueError):
            SkewStudentT(self.xi, self.df, self.mu, -self.scale)

        with self.assertRaises(ValueError):
            SkewStudentT(self.xi, 2.0, self.mu, self.scale)


if __name__ == "__main__":
    unittest.main()
