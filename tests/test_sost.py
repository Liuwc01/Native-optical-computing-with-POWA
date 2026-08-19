import unittest

import torch

from powa import SOSTConv2d, WPUCodebook, project_sost_wavelengths


class SOSTLayerTests(unittest.TestCase):
    def setUp(self):
        columns = torch.arange(5, dtype=torch.float32)
        weights = torch.stack([columns + 10.0 * row for row in range(8)])
        wavelength = torch.linspace(1540.0, 1540.8, 5, dtype=torch.float64)
        self.bank = WPUCodebook(weights, wavelength)

    def test_eq_s5_continuous_interpolation(self):
        layer = SOSTConv2d(self.bank, in_channels=1, out_channels=8)
        with torch.no_grad():
            layer.wavelength_u.fill_(0.375)  # k = 1.5
        weight = layer.current_weight()
        for output in range(8):
            expected = torch.full((2, 2), 1.5 + 10.0 * output)
            self.assertTrue(torch.allclose(weight[output, 0], expected))

    def test_deployment_uses_exact_column_and_half_ties_round_up(self):
        layer = SOSTConv2d(self.bank, in_channels=1, out_channels=8)
        with torch.no_grad():
            layer.wavelength_u.fill_(0.375)  # k = 1.5
        layer.set_deployment(True)
        self.assertTrue(torch.equal(layer.rounded_indices(), torch.full((1, 1, 4), 2)))
        weight = layer.current_weight()
        for output in range(8):
            expected = torch.full((2, 2), 2.0 + 10.0 * output)
            self.assertTrue(torch.equal(weight[output, 0], expected))

    def test_continuous_path_backpropagates(self):
        layer = SOSTConv2d(self.bank, in_channels=1, out_channels=8)
        layer(torch.randn(2, 1, 3, 3)).square().mean().backward()
        self.assertIsNotNone(layer.wavelength_u.grad)
        self.assertGreater(float(layer.wavelength_u.grad.abs().sum()), 0.0)

    def test_hard_ste_uses_rounded_forward_and_interpolated_gradient(self):
        layer = SOSTConv2d(
            self.bank,
            in_channels=1,
            out_channels=8,
            training_mode="hard_ste",
        )
        with torch.no_grad():
            layer.wavelength_u.fill_(0.375)  # k = 1.5, rounded half-up to 2
        weight = layer.current_weight()
        for output in range(8):
            expected = torch.full((2, 2), 2.0 + 10.0 * output)
            self.assertTrue(torch.equal(weight[output, 0], expected))
        layer(torch.randn(2, 1, 3, 3)).square().mean().backward()
        self.assertIsNotNone(layer.wavelength_u.grad)
        self.assertGreater(float(layer.wavelength_u.grad.abs().sum()), 0.0)

    def test_projection_enforces_u_interval(self):
        layer = SOSTConv2d(self.bank, in_channels=1, out_channels=8)
        with torch.no_grad():
            layer.wavelength_u[0, 0] = torch.tensor([-1.0, 0.2, 0.8, 2.0])
        project_sost_wavelengths(layer)
        self.assertTrue(
            torch.equal(
                layer.wavelength_u[0, 0],
                torch.tensor([0.0, 0.2, 0.8, 1.0]),
            )
        )

    def test_optical_parameter_ratio(self):
        layer = SOSTConv2d(self.bank, in_channels=16, out_channels=32)
        optical = layer.wavelength_u.numel()
        dense = 32 * 16 * 2 * 2
        self.assertEqual(optical, dense // 8)


if __name__ == "__main__":
    unittest.main()
