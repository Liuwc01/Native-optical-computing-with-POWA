# Style-transfer numerical data

This package contains raw training logs and derived numeric tables.

| Files | Contents |
|---|---|
| `*_training_log_raw.txt` | Raw electronic and optical logs |
| `*_training_history.csv` | Parsed loss and learning-rate records |
| `fig4e_content_style_loss_curves.csv` | Paired content and style losses |
| `fig4d_figS14_style_fusion_weights.csv` | Complementary fusion weights |
| `fig4f_style_bank_parameter_comparison.csv` | Layer and parameter counts |

The histories are parsed from the raw logs. Paired losses are joined by logged
index without smoothing or normalization, and fusion weights sum to one.
`manifest.json` records the processing details; `SHA256SUMS.txt` verifies file
integrity. Image datasets and generated images are not included.
