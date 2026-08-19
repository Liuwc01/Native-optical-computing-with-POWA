import io
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from powa import SOSTConv2d, WPUCodebook
from powa.style_transfer import (
    OpticalStyleBank,
    OpticalStyleBankCollection,
    POWAStyleTransferNetwork,
    ResizeLongSide,
    StyleTrainingConfig,
    StyleTransferTrainer,
    VGG16FeatureExtractor,
    content_training_transform,
    create_style_optimizer,
    create_style_scheduler,
    deployment_wavelengths_nm,
    finalize_style_network,
    fuse_style_banks,
    gram_matrix,
    inference_transform,
    load_style_model_state,
    style_parameter_statistics,
    style_transfer_loss,
    total_variation_loss,
)
from powa.style_transfer.cli import build_parser


def synthetic_bank() -> WPUCodebook:
    wavelength = torch.linspace(1540.0, 1560.0, 101, dtype=torch.float64)
    weights = torch.stack(
        [torch.linspace(-1.0 + row, 1.0 + row, 101) for row in range(8)]
    )
    return WPUCodebook(weights, wavelength)


def different_synthetic_bank() -> WPUCodebook:
    bank = synthetic_bank()
    weights = bank.weights.clone()
    weights[0, 0] += 0.25
    return WPUCodebook(weights, bank.wavelength_nm)


def small_style_network(codebook: WPUCodebook) -> POWAStyleTransferNetwork:
    return POWAStyleTransferNetwork(
        codebook,
        styles=2,
        latent_channels=8,
        encoder_channels=(4, 8, 8),
    )


def small_vgg(convolutions: int = 4) -> nn.Sequential:
    layers = []
    for _ in range(convolutions):
        layers.extend((nn.Conv2d(3, 3, 3, padding=1), nn.ReLU(inplace=False)))
    return nn.Sequential(*layers)


class StyleTransferTests(unittest.TestCase):
    def test_reported_training_configuration_is_machine_readable(self) -> None:
        config = StyleTrainingConfig()
        self.assertEqual(
            (
                config.styles,
                config.content_crop_pixels,
                config.style_long_side_pixels,
                config.test_images,
                config.epochs,
                config.batch_size,
            ),
            (15, 512, 600, 25, 150, 4),
        )
        self.assertEqual(config.optimizer, "AdamW")
        self.assertTrue(config.freeze_encoder)
        self.assertEqual(config.learning_rate, 1.0e-3)
        self.assertEqual((config.decay_factor, config.decay_every_epochs), (0.8, 10))
        self.assertEqual(
            (config.content_weight, config.style_weight, config.regularization_weight),
            (1.0, 1.0e6, 1.0e-5),
        )

    def test_paper_image_transforms(self) -> None:
        image = Image.new("RGB", (700, 640), color="white")
        content = content_training_transform()(image)
        self.assertEqual(tuple(content.shape), (3, 512, 512))
        resized = ResizeLongSide(600)(Image.new("RGB", (1200, 400)))
        self.assertEqual(resized.size, (600, 200))
        full_resolution = inference_transform()(Image.new("RGB", (31, 29)))
        self.assertEqual(tuple(full_resolution.shape), (3, 29, 31))

    def test_vgg_features_remain_pre_relu_with_inplace_source(self) -> None:
        features = nn.Sequential(
            nn.Conv2d(3, 3, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            features[0].weight.fill_(-1.0)
            features[2].weight.fill_(1.0)
        extractor = VGG16FeatureExtractor(features, layers=(1, 2))
        selected = extractor(torch.ones(1, 3, 4, 4))
        self.assertLess(float(selected[0].min()), 0.0)
        self.assertTrue(
            all(
                not module.inplace
                for module in extractor.features.modules()
                if isinstance(module, nn.ReLU)
            )
        )

    def test_optical_bank_and_collection_have_reported_structure(self) -> None:
        bank = OpticalStyleBank(synthetic_bank(), channels=8)
        self.assertEqual(len(bank.layers), 4)
        self.assertTrue(all(isinstance(layer, SOSTConv2d) for layer in bank.layers))
        inputs = torch.randn(1, 8, 7, 9)
        self.assertEqual(tuple(bank(inputs).shape), (1, 8, 7, 9))
        collection = OpticalStyleBankCollection(synthetic_bank(), styles=15, channels=8)
        self.assertEqual(len(collection), 15)

    def test_complete_network_preserves_arbitrary_image_shape(self) -> None:
        model = small_style_network(synthetic_bank()).eval()
        inputs = torch.randn(2, 3, 31, 29)
        with torch.no_grad():
            output = model(inputs, [0, 1])
            fused = model.forward_fused(inputs[:1], [0, 1], [0.25, 0.75])
        self.assertEqual(tuple(output.shape), tuple(inputs.shape))
        self.assertEqual(tuple(fused.shape), tuple(inputs[:1].shape))

    def test_exact_paper_width_parameter_statistics(self) -> None:
        model = POWAStyleTransferNetwork(synthetic_bank())
        statistics = style_parameter_statistics(model)
        self.assertEqual(statistics.style_bank_parameters, 1_966_080)
        self.assertEqual(statistics.encoder_parameters, 394_848)
        self.assertEqual(statistics.decoder_parameters, 394_848)
        self.assertEqual(statistics.fixed_encoder_trainable_parameters, 2_360_928)
        self.assertEqual(statistics.total_parameters, 2_755_776)
        self.assertAlmostEqual(statistics.style_bank_share_of_total, 0.7134396990)
        self.assertAlmostEqual(
            statistics.style_bank_share_with_fixed_encoder, 0.8327572887
        )

        model.set_encoder_trainable(False)
        frozen = style_parameter_statistics(model)
        self.assertEqual(frozen.trainable_parameters, 2_360_928)
        self.assertAlmostEqual(frozen.style_bank_share_of_trainable, 0.8327572887)

    def test_cli_defaults_to_the_fixed_encoder_training_configuration(self) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--codebook",
                "codebook.npz",
                "--content-dir",
                "content",
                "--style-dir",
                "styles",
                "--output-dir",
                "output",
            ]
        )
        self.assertFalse(arguments.train_encoder)

    def test_fusion_uses_interior_kernel_mixture(self) -> None:
        banks = [
            OpticalStyleBank(synthetic_bank(), channels=8).eval(),
            OpticalStyleBank(synthetic_bank(), channels=8).eval(),
        ]
        with torch.no_grad():
            for layer in banks[0].layers:
                layer.wavelength_u.fill_(0.1)
            for layer in banks[1].layers:
                layer.wavelength_u.fill_(0.9)
        inputs = torch.randn(1, 8, 10, 10)
        with torch.no_grad():
            fused = fuse_style_banks(inputs, banks, [0.25, 0.75])
            output = inputs
            for index, normalization in enumerate(banks[0].normalizations):
                weight = 0.25 * banks[0].layers[index].current_weight()
                weight = weight + 0.75 * banks[1].layers[index].current_weight()
                output = banks[0].padding(output)
                output = torch.relu(normalization(torch.conv2d(output, weight)))
        torch.testing.assert_close(fused, output)

    def test_deployment_state_survives_state_dict_round_trip(self) -> None:
        bank = OpticalStyleBank(synthetic_bank(), channels=8)
        with torch.no_grad():
            bank.layers[0].wavelength_u.fill_(0.375)
        finalize_style_network(bank)
        stream = io.BytesIO()
        torch.save(bank.state_dict(), stream)
        stream.seek(0)
        restored = OpticalStyleBank(synthetic_bank(), channels=8)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        self.assertTrue(all(layer.deployment for layer in restored.layers))
        values = deployment_wavelengths_nm(restored)
        self.assertTrue(
            all(
                torch.allclose((wavelength - 1540.0) / 0.2, ((wavelength - 1540.0) / 0.2).round())
                for wavelength in values.values()
            )
        )

    def test_style_checkpoint_rejects_a_different_selected_codebook(self) -> None:
        trained_bank = synthetic_bank()
        selected_bank = different_synthetic_bank()
        trained_model = small_style_network(trained_bank)
        selected_model = small_style_network(selected_bank)
        checkpoint = {
            "model": trained_model.state_dict(),
            "codebook_fingerprint": trained_bank.fingerprint,
        }

        with self.assertRaisesRegex(ValueError, "checkpoint fingerprint"):
            load_style_model_state(selected_model, checkpoint, selected_bank)

        # A forged or stale fingerprint cannot bypass the value-level buffer
        # check that runs before load_state_dict.
        checkpoint["codebook_fingerprint"] = selected_bank.fingerprint
        before = next(
            layer.codebook.clone()
            for layer in selected_model.modules()
            if isinstance(layer, SOSTConv2d)
        )
        with self.assertRaisesRegex(ValueError, "checkpoint layer"):
            load_style_model_state(selected_model, checkpoint, selected_bank)
        after = next(
            layer.codebook
            for layer in selected_model.modules()
            if isinstance(layer, SOSTConv2d)
        )
        torch.testing.assert_close(after, before)

    def test_legacy_style_state_dict_is_checked_and_remains_loadable(self) -> None:
        bank = synthetic_bank()
        source = small_style_network(bank)
        restored = small_style_network(bank)
        load_style_model_state(restored, source.state_dict(), bank)
        for expected, observed in zip(source.parameters(), restored.parameters()):
            torch.testing.assert_close(observed, expected)

    def test_formula_reductions_match_squared_norms(self) -> None:
        features = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
        expected_gram = torch.tensor([[[5.0, 11.0], [11.0, 25.0]]])
        torch.testing.assert_close(gram_matrix(features), expected_gram)

        image = torch.tensor([[[[0.0, 4.0], [3.0, 0.0]]]])
        self.assertEqual(float(total_variation_loss(image)), 5.0)
        zero = torch.zeros_like(features)
        losses = style_transfer_loss(
            [features],
            [zero],
            [zero],
            image,
            content_weight=1.0,
            style_weight=1.0,
            regularization_weight=1.0,
        )
        expected_content = features.square().sum()
        expected_style = expected_gram.square().sum()
        torch.testing.assert_close(losses.content, expected_content)
        torch.testing.assert_close(losses.style, expected_style)
        torch.testing.assert_close(
            losses.total, expected_content + expected_style + 5.0
        )

    def test_optimizer_scheduler_and_training_smoke(self) -> None:
        config = StyleTrainingConfig(
            styles=2,
            content_crop_pixels=16,
            style_long_side_pixels=16,
            test_images=2,
            epochs=1,
            batch_size=2,
            latent_channels=8,
            content_vgg_layer=4,
            style_vgg_layers=(2, 4),
            style_layer_weights=(1.0, 1.0),
        )
        bank = synthetic_bank()
        model = small_style_network(bank)
        optimizer = create_style_optimizer(model, config)
        scheduler = create_style_scheduler(optimizer, config)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 8.0e-4)

        extractor = VGG16FeatureExtractor(small_vgg(), layers=(2, 4))
        trainer = StyleTransferTrainer(
            model,
            extractor,
            [torch.rand(3, 16, 16), torch.rand(3, 16, 16)],
            config=config,
        )
        encoder_parameter_ids = {id(parameter) for parameter in model.encoder.parameters()}
        optimized_parameter_ids = {
            id(parameter)
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.encoder.parameters())
        )
        self.assertTrue(encoder_parameter_ids.isdisjoint(optimized_parameter_ids))
        self.assertTrue(
            all(
                id(parameter) in optimized_parameter_ids
                for parameter in model.style_banks.parameters()
            )
        )
        self.assertTrue(
            all(
                id(parameter) in optimized_parameter_ids
                for parameter in model.decoder.parameters()
            )
        )
        metrics = trainer.train_epoch([torch.rand(2, 3, 16, 16)])
        self.assertEqual(metrics.batches, 1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics.total)))
        self.assertTrue(
            all(
                0.0 <= float(layer.wavelength_u.detach().min())
                and float(layer.wavelength_u.detach().max()) <= 1.0
                for bank in model.style_banks.banks
                for layer in bank.layers
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "style.pt"
            trainer.save_checkpoint(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.assertEqual(checkpoint["codebook_fingerprint"], bank.fingerprint)
        self.assertEqual(
            set(checkpoint["software_versions"]),
            {"torch", "torchvision", "cuda_runtime"},
        )
        self.assertEqual(
            checkpoint["software_versions"]["torch"], str(torch.__version__)
        )


if __name__ == "__main__":
    unittest.main()
