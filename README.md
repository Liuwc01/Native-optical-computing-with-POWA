# Native optical computing with POWA

This repository accompanies the article *Native optical computing with
wavelength-parameterized passive photonics*.

POWA is a Python library for wavelength-parameterized optical computing. It
provides:

- SOST (`SOSTConv2d`), a differentiable 2 x 2 convolution backed by a measured
  eight-output optical response codebook;
- an O-VGG classifier for CIFAR-10 and CIFAR-100;
- an encoder--optical-style-bank--decoder network for single-style and fused
  style transfer;
- measured spectra, numerical source data, examples, and integrity tests.

The package is pure Python and does not require compiled project-specific
extensions.

## Installation

Requirements:

- Python 3.10 or newer
- h5py 3.9 or newer
- NumPy 1.24 or newer
- SciPy 1.10 or newer
- PyTorch 2.1 or newer
- torchvision 0.16 or newer

Create the supplied Conda environment and install the package:

```bash
conda env create -f environment.yml
conda activate powa
python -m pip install -e .
```

Select a CPU or CUDA build of PyTorch appropriate for the local system. A GPU
is recommended for model training but is not required for the demo, tests, or
offline checkpoint evaluation. Optical hardware is needed to acquire a new
response codebook or to execute a physically deployed model.

The package was tested on Windows 11 with Python 3.12.13, NumPy 2.5.2, SciPy
1.18.0, h5py 3.16.0, PyTorch 2.13.0+cpu and torchvision 0.28.0+cpu. CUDA tests
used PyTorch 2.10.0, torchvision 0.25.0, CUDA 13.0 and an RTX 4060 Ti. A cached
installation took about 86 seconds, and the CPU demo took about 2.3 seconds.

## Quick demo

```bash
python -m examples.sost_demo
```

The demo constructs a small synthetic response bank, runs continuous and
rounded SOST convolutions, checks gradients, and prints a JSON record ending in
`"status": "ok"`. It does not write output files.

## Optical codebook

Model commands accept an NPZ archive containing:

- `codebook`: a finite floating-point array with shape `[8, K]`;
- `wavelength_nm`: a finite one-dimensional array with shape `[K]`.

The wavelength axis must be strictly increasing and uniformly sampled. Each
codebook row must span `[0, 1]`. `load_codebook` mean-centres each row, divides
it by its sample standard deviation plus `1e-6`, and scales it by `0.20`.

The default loader interval is 0.2 nm. O-VGG evaluation reads the required
interval from the checkpoint metadata. A checkpoint must be used with the same
codebook values, wavelength axis, and output-port order used to create it.
Loading verifies the stored fingerprint and the SOST buffers before inference;
a mismatch raises an error.

## Python API

```python
import torch
from powa import SOSTConv2d, load_codebook, project_sost_wavelengths

bank = load_codebook("/path/to/codebook.npz")
layer = SOSTConv2d(bank, in_channels=3, out_channels=8)
optimizer = torch.optim.Adam(layer.parameters(), lr=2e-3)

x = torch.randn(2, 3, 16, 16)
loss = layer(x).square().mean()
loss.backward()
optimizer.step()
project_sost_wavelengths(layer)

layer.set_deployment(True)
y = layer(x)
```

`project_sost_wavelengths` constrains wavelength variables to `[0, 1]`.
Deployment selects exact response-grid columns, with half-grid ties rounded
upward.

The main public objects are:

- `WPUCodebook` and `load_codebook`;
- `SOSTConv2d`, `project_sost_wavelengths`, and `set_sost_deployment`;
- `OVGGBackbone` and `OVGGClassifier`;
- `POWAStyleTransferNetwork` and `style_parameter_statistics` in
  `powa.style_transfer`.

## O-VGG

Train on CIFAR-10 or CIFAR-100:

```bash
python -m powa.train \
  --dataset cifar10 \
  --codebook /path/to/codebook.npz \
  --data-dir data/cifar \
  --output-dir outputs/cifar10 \
  --seed 42
```

The training directory contains `history.csv`, `checkpoint_last.pth`, and
`test_metrics.json`. CIFAR is downloaded through torchvision if it is absent
from `--data-dir`. O-VGG training uses a hard straight-through estimator: the
forward pass selects the nearest response-grid column, while gradients follow
the adjacent-column linear interpolation.

Evaluate a checkpoint:

```bash
python -m powa.evaluate \
  --checkpoint outputs/cifar10/checkpoint_last.pth \
  --codebook /path/to/codebook.npz \
  --data-dir data/cifar \
  --output-dir outputs/cifar10-evaluation \
  --device auto
```

Evaluation writes `test_metrics.json` and `test_predictions.npz`. A
self-contained offline software-evaluation command and matched assets are
available in
[`examples/ovgg_cifar10_sample`](examples/ovgg_cifar10_sample). This example
does not implement instrument control or physical optical deployment.

## Optical style transfer

Train 15 style banks:

```bash
python -m powa.style_transfer train \
  --codebook /path/to/codebook.npz \
  --content-dir /path/to/content_images \
  --style-dir /path/to/15_style_images \
  --output-dir outputs/style_transfer
```

Apply one bank:

```bash
python -m powa.style_transfer infer \
  --codebook /path/to/codebook.npz \
  --checkpoint outputs/style_transfer/deployment.pt \
  --input-dir /path/to/input_images \
  --output-dir outputs/stylized \
  --style-id 3 \
  --expected-images 25
```

With the encoder fixed, the 15 optical banks contain 1,966,080 optimized
parameters and the decoder contains 394,848. The banks therefore account for
`1,966,080 / (1,966,080 + 394,848) = 83.2757%`, or approximately 84% of the
optimized parameters; fixed encoder parameters are excluded from this
denominator.

See [docs/STYLE_TRANSFER.md](docs/STYLE_TRANSFER.md) for image requirements,
fusion, outputs, and parameter reporting.

## Verification

```bash
python scripts/verify_repository.py
python -m unittest discover -s tests -v
```

The verifier checks committed arrays, manifests, and SHA-256 records. The test
suite covers codebook validation, SOST gradients and rounding, architectures,
training utilities, checkpoint matching, and inference.

## Data and documentation

- [Data dictionary](docs/DATA_DICTIONARY.md)
- [Numerical data overview](docs/FIGURE_DATA.md)
- [Computational methods](docs/METHODS.md)
- [Availability](docs/AVAILABILITY.md)
- [Style transfer](docs/STYLE_TRANSFER.md)
- [O-VGG sample](examples/ovgg_cifar10_sample)

## License

Source code and documentation are released under the [MIT License](LICENSE).
Third-party datasets, images, and pretrained weights retain their original
licenses.
