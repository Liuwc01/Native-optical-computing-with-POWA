# WPU broadband response

`wpu_broadband_response_1500_1620_10pm.mat` is a compact MATLAB file derived
from `data/source/wpu_spectrum_1500_1620_db.mat`.

| Variable | Shape | Unit | Description |
|---|---:|---|---|
| `wavelength_nm` | 1 x 12,001 | nm | Uniform 1500-1620 nm grid with 0.01-nm spacing |
| `response_db` | 8 x 12,001 | dB | Rows are WPU output ports; columns are wavelengths |
| `output_port_id` | 1 x 8 | unitless | One-based port identifiers |
| `source_sample_index_1based` | 1 x 12,001 | unitless | Selected source-array indices |

The compact response selects every 200th stored sample and transposes the result
to port-by-wavelength orientation. No interpolation, smoothing, normalization
or filtering was applied. The wavelength grid is supplied separately because
the source MAT file contains no wavelength vector.

See `manifest.json` for provenance and `SHA256SUMS.txt` for integrity checks.
