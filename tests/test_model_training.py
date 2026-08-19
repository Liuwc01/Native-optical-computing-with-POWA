import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms

from powa import (
    OVGGBackbone,
    OVGGClassifier,
    RightBottomPad2d,
    SOSTConv2d,
    TrainingConfig,
    WPUCodebook,
    make_optimizer,
    validate_sost_codebook_buffers,
    validate_sost_state_dict_codebook,
)
from powa.data import build_train_transform
from powa.evaluate import parse_args as parse_evaluate_args
from powa.train import parse_args as parse_train_args


class ModelTrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wavelength_nm = torch.linspace(1540.0, 1560.0, 101, dtype=torch.float64)
        phase = torch.linspace(0.0, 4.0, 101)
        weights = torch.stack([torch.sin(phase + row) for row in range(8)])
        cls.codebook = WPUCodebook(weights, wavelength_nm)

    def test_backbone_has_documented_topology(self):
        backbone = OVGGBackbone(self.codebook)
        self.assertEqual(
            sum(isinstance(layer, SOSTConv2d) for layer in backbone.modules()),
            13,
        )
        self.assertEqual(
            sum(isinstance(layer, nn.MaxPool2d) for layer in backbone.modules()),
            5,
        )
        self.assertEqual(
            sum(isinstance(layer, RightBottomPad2d) for layer in backbone.modules()),
            13,
        )
        self.assertEqual(
            sum(isinstance(layer, nn.BatchNorm2d) for layer in backbone.modules()),
            13,
        )
        batch_norm_layers = [
            layer for layer in backbone.modules() if isinstance(layer, nn.BatchNorm2d)
        ]
        self.assertTrue(all(layer.momentum == 0.9 for layer in batch_norm_layers))
        with torch.no_grad():
            output = backbone(torch.randn(1, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (1, 512, 1, 1))

    def test_classifier_uses_two_fc_head_and_backpropagates(self):
        model = OVGGClassifier(self.codebook, num_classes=10)
        self.assertEqual(model.head_variant, "gap_2fc")
        self.assertEqual(
            [type(layer) for layer in model.classifier],
            [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear],
        )
        linear = [layer for layer in model.classifier if isinstance(layer, nn.Linear)]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in linear],
            [(512, 512), (512, 10)],
        )
        self.assertEqual(model.classifier[2].p, 0.5)
        output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 10))
        output.square().mean().backward()
        self.assertIsNotNone(linear[0].weight.grad)
        self.assertIsNotNone(linear[1].weight.grad)
        first_sost = next(
            layer for layer in model.modules() if isinstance(layer, SOSTConv2d)
        )
        self.assertEqual(first_sost.training_mode, "hard_ste")
        self.assertIsNotNone(first_sost.wavelength_u.grad)

    def test_figure3_whole_network_parameter_counts(self):
        observed = []
        for classes in (10, 100):
            model = OVGGClassifier(self.codebook, num_classes=classes)
            optical = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            wavelength = sum(
                layer.wavelength_u.numel()
                for layer in model.modules()
                if isinstance(layer, SOSTConv2d)
            )
            dense_convolution = sum(
                layer.in_channels * layer.out_channels * 2 * 2
                for layer in model.modules()
                if isinstance(layer, SOSTConv2d)
            )
            electronic = optical - wavelength + dense_convolution
            observed.append((optical, electronic))
        self.assertEqual(
            observed,
            [(1_093_482, 6_814_218), (1_139_652, 6_860_388)],
        )

    def test_padding_reflects_only_on_right_and_bottom(self):
        observed = RightBottomPad2d()(
            torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        )
        expected = torch.tensor(
            [[[[1.0, 2.0, 1.0], [3.0, 4.0, 3.0], [1.0, 2.0, 1.0]]]]
        )
        self.assertTrue(torch.equal(observed, expected))

    def test_training_contract_and_adam_decay_exclusions(self):
        config = TrainingConfig()
        self.assertEqual(
            (
                config.epochs,
                config.batch_size,
                config.learning_rate,
                config.weight_decay,
                config.label_smoothing,
                config.gradient_clip_norm,
                config.seed,
            ),
            (400, 256, 2e-3, 1e-3, 0.1, 5.0, 42),
        )
        model = OVGGClassifier(self.codebook, num_classes=10)
        optimizer = make_optimizer(model)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        decay = next(
            group for group in optimizer.param_groups if group["group_name"] == "decay"
        )
        no_decay = next(
            group
            for group in optimizer.param_groups
            if group["group_name"] == "no_decay"
        )
        self.assertEqual(decay["weight_decay"], 1e-3)
        self.assertEqual(no_decay["weight_decay"], 0.0)
        no_decay_ids = {id(parameter) for parameter in no_decay["params"]}
        for name, parameter in model.named_parameters():
            if name.endswith("wavelength_u") or name.endswith("bias"):
                self.assertIn(id(parameter), no_decay_ids)
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                for parameter in layer.parameters(recurse=False):
                    self.assertIn(id(parameter), no_decay_ids)

    def test_cifar_augmentation_contract(self):
        pipeline = build_train_transform("cifar10").transforms
        self.assertEqual(
            [type(transform) for transform in pipeline],
            [
                transforms.RandomCrop,
                transforms.RandomHorizontalFlip,
                transforms.ColorJitter,
                transforms.ToTensor,
                transforms.Normalize,
                transforms.RandomErasing,
            ],
        )
        self.assertEqual(pipeline[0].padding, 4)
        self.assertEqual(pipeline[0].padding_mode, "constant")
        self.assertEqual(pipeline[-1].p, 0.4)

    def test_training_and_evaluation_require_an_explicit_codebook(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_train_args(
                ["--dataset", "cifar10", "--output-dir", "training-output"]
            )
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_evaluate_args(
                [
                    "--checkpoint",
                    "model.pth",
                    "--output-dir",
                    "evaluation-output",
                ]
            )

        train_args = parse_train_args(
            [
                "--dataset",
                "cifar10",
                "--codebook",
                "custom-bank.npz",
                "--output-dir",
                "training-output",
            ]
        )
        evaluate_args = parse_evaluate_args(
            [
                "--checkpoint",
                "model.pth",
                "--codebook",
                "custom-bank.npz",
                "--output-dir",
                "evaluation-output",
            ]
        )
        self.assertEqual(train_args.codebook, Path("custom-bank.npz"))
        self.assertEqual(evaluate_args.codebook, Path("custom-bank.npz"))
        self.assertEqual(evaluate_args.device, "auto")

    def test_loaded_state_cannot_override_selected_codebook(self):
        original_model = OVGGClassifier(self.codebook, num_classes=10)
        state = original_model.state_dict()
        codebook_key = next(key for key in state if key.endswith(".codebook"))
        state[codebook_key] = state[codebook_key] + 1.0
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_sost_state_dict_codebook(state, self.codebook)
        loaded_model = OVGGClassifier(self.codebook, num_classes=10)
        loaded_model.load_state_dict(state, strict=True)
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_sost_codebook_buffers(loaded_model, self.codebook)


if __name__ == "__main__":
    unittest.main()
