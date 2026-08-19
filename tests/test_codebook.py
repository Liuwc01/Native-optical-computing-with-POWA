import unittest
from unittest.mock import patch

import numpy as np
import torch

from powa import WPUCodebook, load_codebook


class CodebookTests(unittest.TestCase):
    def test_explicit_synthetic_archive_has_paper_grid_and_signed_weights(self):
        wavelength_nm = np.linspace(1540.0, 1560.0, 101, dtype=np.float64)
        base = np.linspace(0.0, 1.0, 101, dtype=np.float32)
        normalized = np.stack([np.roll(base, row) for row in range(8)])

        class SyntheticArchive(dict):
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        archive = SyntheticArchive(
            codebook=normalized,
            wavelength_nm=wavelength_nm,
        )
        with patch("powa.codebook.np.load", return_value=archive) as mocked_load:
            bank = load_codebook("synthetic_codebook.npz")
        mocked_load.assert_called_once()

        self.assertEqual(tuple(bank.weights.shape), (8, 101))
        self.assertTrue(
            np.allclose(bank.wavelength_nm.numpy(), wavelength_nm, atol=1e-7)
        )
        self.assertTrue(
            torch.allclose(bank.weights.mean(dim=1), torch.zeros(8), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                bank.weights.std(dim=1, correction=1),
                torch.full((8,), 0.20),
                atol=2e-6,
            )
        )

    def test_loader_accepts_dynamic_k_on_uniform_point_two_nm_grid(self):
        wavelength_nm = np.linspace(1548.0, 1552.0, 21, dtype=np.float64)
        base = np.linspace(0.0, 1.0, 21, dtype=np.float32)
        normalized = np.stack([np.roll(base, row) for row in range(8)])

        class SyntheticArchive(dict):
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        archive = SyntheticArchive(
            codebook=normalized,
            wavelength_nm=wavelength_nm,
        )
        with patch("powa.codebook.np.load", return_value=archive):
            bank = load_codebook("synthetic_21_point_codebook.npz")
        self.assertEqual(tuple(bank.weights.shape), (8, 21))
        self.assertEqual(bank.points, 21)

    def test_loader_rejects_nonuniform_wavelength_grid(self):
        wavelength_nm = np.linspace(1548.0, 1552.0, 21, dtype=np.float64)
        wavelength_nm[10] += 0.01
        base = np.linspace(0.0, 1.0, 21, dtype=np.float32)
        normalized = np.stack([np.roll(base, row) for row in range(8)])

        class SyntheticArchive(dict):
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        archive = SyntheticArchive(
            codebook=normalized,
            wavelength_nm=wavelength_nm,
        )
        with patch("powa.codebook.np.load", return_value=archive):
            with self.assertRaisesRegex(ValueError, "uniform 0.2-nm"):
                load_codebook("nonuniform_codebook.npz")

    def test_explicit_archived_interval_is_supported(self):
        wavelength_nm = np.linspace(1545.0, 1555.0, 101, dtype=np.float64)
        base = np.linspace(0.0, 1.0, 101, dtype=np.float32)
        normalized = np.stack([np.roll(base, row) for row in range(8)])

        class SyntheticArchive(dict):
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        archive = SyntheticArchive(
            codebook=normalized,
            wavelength_nm=wavelength_nm,
        )
        with patch("powa.codebook.np.load", return_value=archive):
            bank = load_codebook(
                "archived_101_point_codebook.npz",
                expected_interval_nm=0.1,
            )
        self.assertEqual(bank.points, 101)
        self.assertAlmostEqual(float(bank.wavelength_nm[0]), 1545.0)
        self.assertAlmostEqual(float(bank.wavelength_nm[-1]), 1555.0)

    def test_loader_has_no_implicit_archive(self):
        with self.assertRaises(TypeError):
            load_codebook()

    def test_axis_length_must_match_weights(self):
        with self.assertRaisesRegex(ValueError, r"shape \[K\]"):
            WPUCodebook(torch.randn(8, 101), torch.linspace(1540, 1560, 100))


if __name__ == "__main__":
    unittest.main()
