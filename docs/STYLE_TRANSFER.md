# Optical style transfer

`powa.style_transfer` implements an encoder--optical-style-bank--decoder image
pipeline. The encoder maps RGB images to 256 latent channels, one of 15 optical
banks transforms the latent features, and the decoder reconstructs an RGB image
at the original dimensions. Each bank contains four 2 x 2 SOST convolutions and
supports discrete deployment or layer-wise fusion.

## Inputs

The training command requires:

- an NPZ optical codebook accepted by `load_codebook`;
- a flat or nested content-image directory;
- exactly 15 style-reference images;
- an output directory.

Supported image formats are BMP, JPEG, PNG, TIFF, and WebP. Content images must
be at least 512 x 512 because training uses random 512 x 512 crops. Style images
are converted to RGB and resized so their longer side is 600 pixels.

Style files are discovered recursively and sorted by path. Their positions in
that order define style IDs `0` through `14`. Prefixing filenames makes this
mapping explicit:

```text
styles/
|-- 00_style.jpg
|-- 01_style.jpg
|-- ...
`-- 14_style.jpg
```

Keep the filename-to-ID mapping with the checkpoint.

## Training

```bash
python -m powa.style_transfer train \
  --codebook /path/to/codebook.npz \
  --content-dir /path/to/content_images \
  --style-dir /path/to/15_style_images \
  --output-dir outputs/style_transfer
```

The default configuration uses 150 epochs, batch size 4, AdamW with initial
learning rate `1e-3`, and a learning-rate factor of `0.8` every 10 epochs. The
encoder is fixed; add `--train-encoder` to optimize it. The default VGG-16
perceptual weights are downloaded through torchvision if not cached. Use
`--vgg-weights /path/to/vgg16_state_dict.pt` for local weights or
`--vgg-weights none` for structural tests.

The loss combines content features, Gram-matrix style features, and isotropic
total variation with weights `1`, `1e6`, and `1e-5`. The VGG-16 feature
extractor remains fixed.

Training writes `epoch_NNN.pt` after each epoch and a discretized
`deployment.pt` at completion. Keep the exact codebook used for training with
these files.

## Inference

Apply a single style bank:

```bash
python -m powa.style_transfer infer \
  --codebook /path/to/codebook.npz \
  --checkpoint outputs/style_transfer/deployment.pt \
  --input-dir /path/to/input_images \
  --output-dir outputs/style_03 \
  --style-id 3 \
  --expected-images 25
```

Input images are discovered recursively and sorted by path. The command expects
25 images by default; set `--expected-images N` to the actual count. Results are
full-resolution PNG files named `NNN_<input-stem>.png` in the output directory.

Fuse two or more banks by replacing `--style-id` with matching style IDs and
nonnegative coefficients that sum to one:

```bash
python -m powa.style_transfer infer \
  --codebook /path/to/codebook.npz \
  --checkpoint outputs/style_transfer/deployment.pt \
  --input-dir /path/to/input_images \
  --output-dir outputs/fused \
  --fusion-style-ids 2 14 \
  --fusion-coefficients 0.25 0.75 \
  --expected-images 25
```

The equivalent Python interface is
`POWAStyleTransferNetwork.forward_fused`.

## Codebook matching

Inference requires the same codebook values, wavelength axis, and output-port
order used for training. It validates the checkpoint fingerprint and every
persisted SOST codebook and wavelength buffer before loading, then checks the
loaded model again. Any mismatch stops inference instead of silently changing
the optical weights.

## Parameter statistics

With the encoder fixed, 15 four-layer 256-channel optical banks contain

`15 x 4 x 2 x 2 x 256 x 256 / 8 = 1,966,080`

optimized wavelength parameters. The decoder contains 394,848 parameters, so
the optical-bank share is

`1,966,080 / (1,966,080 + 394,848) = 83.2757%`,

or approximately 84% of the optimized parameters when the encoder is fixed.
Fixed encoder parameters are excluded from this denominator.

Print counts from a concrete model with:

```bash
python -m powa.style_transfer stats \
  --codebook /path/to/codebook.npz
```

Add `--train-encoder` to include the encoder in the trainable-parameter count.
