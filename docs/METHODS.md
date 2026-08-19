# Computational methods

## Codebook loading

`load_codebook` reads an NPZ archive with an `8 x K` `codebook` array and a
matching `wavelength_nm` axis. It requires finite values, a strictly increasing
uniform wavelength grid, and a `[0, 1]` span in every output row. The loader
mean-centres each row, divides it by its sample standard deviation plus `1e-6`,
and scales it by `0.20`.

The default interval is 0.2 nm. Callers may supply another expected uniform
interval when loading a compatible checkpoint. The resulting `WPUCodebook`
stores the processed weights, wavelength axis, number of points, and a SHA-256
fingerprint of the numerical contents.

## SOST convolution

`SOSTConv2d` maps each learned scalar `u` in `[0, 1]` to a continuous codebook
index `k = (K - 1)u`. Continuous mode linearly interpolates the two adjacent
response columns. Hard-STE mode selects the nearest column in the forward pass
and uses the interpolation gradient in the backward pass. Deployment mode
selects exact grid columns and rounds half-grid ties upward.

Call `project_sost_wavelengths(model)` after optimizer updates to keep all
wavelength variables in range, and `set_sost_deployment(model, True)` before
rounded inference. Each wavelength selection produces eight output weights in
parallel. A 2 x 2 layer therefore uses `Cin x Cout / 2` learned wavelength
scalars when `Cout` is divisible by eight.

## O-VGG

`OVGGBackbone` contains 13 wavelength-parameterized 2 x 2 convolutions in the
VGG channel sequence. Each convolution uses right/bottom reflection padding,
BatchNorm (`eps=1e-5`, `momentum=0.9`), and ReLU. Five max-pooling stages reduce
the spatial dimensions. `OVGGClassifier` adds:

```text
AdaptiveAvgPool2d(1) -> FC(512, 512) -> ReLU -> Dropout(0.5)
                     -> FC(512, classes)
```

O-VGG training uses hard-STE sampling. Its forward pass selects the nearest
response-grid column, while its backward pass uses the gradient of the
adjacent-column linear interpolation. Deployment also selects exact nearest
grid columns.

The training command uses 400 epochs, batch size 256, Adam, cosine learning-rate
annealing from `2e-3` to zero, weight decay `1e-3`, label smoothing `0.1`, and
gradient clipping at norm 5. Its default seed is 42. Wavelength variables,
biases, and BatchNorm parameters are excluded from weight decay. It writes the
epoch history, final checkpoint, and rounded evaluation metrics to the selected
output directory.

## Optical style transfer

`POWAStyleTransferNetwork` combines a fully convolutional encoder, 15 optical
style banks, and a fully convolutional decoder. Each bank contains four
256-channel SOST convolutions followed by instance normalization and ReLU. The
network supports single-bank inference and layer-wise convex fusion through
`forward_fused`.

The encoder is fixed by default, so the optimized parameters are the decoder
and style banks. The bank count is

`15 x 4 x 2 x 2 x 256 x 256 / 8 = 1,966,080`.

Together with 394,848 decoder parameters, the optical-bank share is

`1,966,080 / (1,966,080 + 394,848) = 83.2757%`,

or approximately 84% of the optimized parameters when the encoder is fixed.
Fixed encoder parameters are excluded from this denominator.
`style_parameter_statistics` calculates these values from a model instance.
See [STYLE_TRANSFER.md](STYLE_TRANSFER.md) for the CLI, image inputs, losses,
and generated files.

## Checkpoint and codebook consistency

Checkpoints store the codebook fingerprint, grid metadata, architecture
metadata, and software versions. Evaluation and style-transfer inference
require an explicit codebook archive. Loading verifies the fingerprint and the
persisted SOST codebook and wavelength buffers before inference. Codebook
values, wavelength positions, or output-port order must not be changed between
training and inference.
