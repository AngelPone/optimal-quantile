import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import expanding_window


class TestExpandingWindowIterator(unittest.TestCase):
    def test_collect_test_returns_numpy_test_windows(self):
        dataset = np.arange(10)

        actual = expanding_window(T=10, S=4, h=1, dataset=dataset).collect_test()

        expected = np.array([[4], [5], [6], [7], [8], [9]])
        np.testing.assert_array_equal(actual, expected)

    def test_collect_test_returns_multistep_torch_test_windows(self):
        dataset = torch.arange(20).reshape(10, 2)

        actual = expanding_window(T=10, S=4, h=3, dataset=dataset).collect_test()

        expected = torch.stack(
            [
                dataset[4:7],
                dataset[5:8],
                dataset[6:9],
                dataset[7:10],
            ]
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (4, 3, 2))

    def test_collect_test_returns_test_slices_without_dataset(self):
        actual = expanding_window(T=10, S=4, h=3).collect_test()

        self.assertEqual(actual, [slice(4, 7), slice(5, 8), slice(6, 9), slice(7, 10)])


if __name__ == "__main__":
    unittest.main()
