# Numerical data overview

Author-generated numerical records are organized under
`data/figure_source_data/`.

| Package | Description |
|---|---|
| `fig2c_figS1_wpu_broadband_response/` | Compact eight-port WPU broadband response |
| `device_and_system/fig2c_figS1_wpu_single_port_response/` | Explicit-axis single-port WPU response |
| `device_and_system/fig2c_microring_response/` | Full and sampled microring responses |
| `device_and_system/fig2f_figS8_figS11_high_speed_waveforms/` | Input sequence and oscilloscope waveforms |
| `fig3_ovgg/` | Classification accuracy and parameter-count summaries |
| `fig4_style_transfer/` | Style-transfer logs, losses, fusion weights and parameter counts |

Each package provides a README, `manifest.json` and `SHA256SUMS.txt`. The
manifests describe variables, units and processing relationships; checksum
files verify the distributed bytes. A matched runnable O-VGG sample is located
under `examples/ovgg_cifar10_sample/`.

Third-party image datasets are not redistributed.
