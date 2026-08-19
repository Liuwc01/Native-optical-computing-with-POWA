# O-VGG CIFAR-10 offline evaluation sample

This directory contains a matched optical response bank, O-VGG checkpoint and
evaluation metrics for a standalone offline software-evaluation example. It
does not implement hardware control or physical optical deployment. Physical
deployment requires programming the selected wavelengths into the optical
system and collecting its outputs.

| File | Contents |
|---|---|
| `ovgg_cifar10_k101_codebook.mat` | MATLAB response-bank archive |
| `ovgg_cifar10_k101_codebook.npz` | Codebook used by the Python evaluator |
| `ovgg_cifar10_k101_reflect.pth` | Matching two-FC O-VGG checkpoint |
| `test_metrics.json` | Stored evaluation result |

From the repository root, run:

```bash
python -m pip install -e .
python -m powa.evaluate \
  --checkpoint examples/ovgg_cifar10_sample/ovgg_cifar10_k101_reflect.pth \
  --codebook examples/ovgg_cifar10_sample/ovgg_cifar10_k101_codebook.npz \
  --data-dir data/cifar \
  --output-dir outputs/ovgg_cifar10_sample \
  --device cpu \
  --num-workers 0
```

Torchvision downloads CIFAR-10 if needed. The command writes
`test_metrics.json` and `test_predictions.npz`. A verified CPU run took about
11 seconds after the dataset and dependencies were available. This command
evaluates the fixed bundled checkpoint; it does not train a model or assert a
training seed for that checkpoint.

See `manifest.json` for metadata and `SHA256SUMS.txt` for file integrity.
