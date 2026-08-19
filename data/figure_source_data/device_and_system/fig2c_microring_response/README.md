# Microring spectral response

| File | Variable or content | Size / relationship |
|---|---|---|
| `microring_response_full_source_1500_1620.fig` | MATLAB source figure with `XData` and `YData` | 2,400,001 samples over 1500-1620 nm |
| `microring_response_full_y_values.mat` | `all_y_coordinates` | Exact copy of the full `YData` array |
| `microring_response_10pm_y_values.mat` | `y` | Every 200th full-response value; 12,001 samples |

The extracted MAT files contain response ordinates only; their unit and display
convention remain defined by the source figure. No smoothing, interpolation or
normalization was applied. See `manifest.json` and `SHA256SUMS.txt` for metadata
and integrity checks.
