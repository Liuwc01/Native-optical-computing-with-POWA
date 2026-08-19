# O-VGG summary data

This package contains author-generated O-VGG summary arrays and MATLAB plotting
scripts.

- `summary_tables/fig3f_accuracy_and_parameter_counts.csv`: dataset, model or
  training mode, test accuracy (%) and parameter count.
- `summary_tables/fig3g_parameter_count_vs_depth.csv`: convolutional depth and
  electronic or wavelength-parameterized counts (thousands).
- `summary_tables/fig3h_accuracy_vs_wavelength_range.csv`: wavelength-index
  intervals, sampled points, switching range (nm) and CIFAR accuracy (%).
- `plot_sources/`: MATLAB plotting scripts containing the same summary arrays.

The wavelength range is calculated as `interval count x 0.2 nm`, and the number
of sampled points is `interval count + 1`. `manifest.json` records file-level
metadata, while `SHA256SUMS.txt` provides integrity checks.

CIFAR image files are third-party data and are not redistributed. A runnable
sample is available under `examples/ovgg_cifar10_sample/`.
