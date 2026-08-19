# Data dictionary

| Path | Contents | Key variables / units |
|---|---|---|
| `data/source/wpu_spectrum_1500_1620_db.mat` | Full-band eight-port WPU spectrum | `xiangying2`: 8 x 2,400,001 in MATLAB; dB |
| `data/metadata/wpu_spectrum.json` | Spectrum orientation, range and digest | JSON |
| `data/figure_source_data/fig2c_figS1_wpu_broadband_response/` | Compact eight-port spectrum | `wavelength_nm` (nm), `response_db` (dB), port and source indices |
| `data/figure_source_data/device_and_system/` | WPU and microring spectra plus input and measured waveforms | FIG, MAT, headerless CSV/TXT |
| `data/figure_source_data/fig3_ovgg/` | O-VGG accuracy and parameter summaries | Accuracy (%), wavelength range (nm), parameter counts |
| `data/figure_source_data/fig4_style_transfer/` | Training histories, loss curves, fusion weights and parameter counts | TXT, CSV |
| `examples/ovgg_cifar10_sample/` | Runnable O-VGG sample with response bank, checkpoint and metrics | MAT, NPZ, PTH, JSON |

The compact WPU spectrum selects every 200th stored sample from the full array;
no interpolation, smoothing, normalization or filtering is applied. Other
deterministic selections and conversions are documented in the package README
and `manifest.json` files. SHA-256 checksum lists accompany each data package.

CIFAR and style/content image datasets are third-party resources and are not
redistributed.
