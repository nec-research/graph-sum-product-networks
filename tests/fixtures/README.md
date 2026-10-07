<!-- GSPN-GPT-FIXED: Preserve independently captured numerical evidence after retiring old code. -->
# GSPN regression fixture

`gspn_regression.pt.gz` contains 569 CPU cases captured from the original implementation before its removal. The source revision is `a9e33246bd0745ea874c7c6074a5b6a28a46090b`. The existing original-versus-refactored tests passed before capture. Loop-free reference graphs had optional self-loop insertion disabled; the current model preserves these results with one automatic loop per node. Duplicate-loop behavior is tested separately against analytic expectations.

| Group | Cases |
| --- | ---: |
| outputs | 288 |
| topology | 16 |
| readouts | 64 |
| gaussian_readouts | 3 |
| updates | 192 |
| initialization | 6 |

Records contain input tensors, architecture settings, legacy parameter states, and expected predictions, embeddings, and all ten auxiliary outputs. Applicable groups also contain gradients (including intentionally absent gradients), one Adam update, or initialized states and RNG state. They contain no executable model objects. Tests reconstruct current models and transfer legacy parameter tensors through `load_reference_state_dict()`.

Load with `gzip.open()` and `torch.load(..., map_location="cpu", weights_only=True)`. The fixture format version is 1. Tests use float64 tolerances `rtol=1e-8, atol=1e-10` and float32 tolerances `rtol=1e-5, atol=1e-6`.

Fixture SHA-256: `907b4c25f7f9e69ccdfc0c466fa39c0e378d5a09d45978701d1e217d42f1a037`.

Source SHA-256 checksums at capture:

- `model.py`: `7d5d702e16b12d21613d902e8b9a2d4ba40e1ff27e1cadefc6fc1ee71a5f894e`
- `sup_model.py`: `27147483e2e4455a70e7499c238f34005f837e03561a1d192557340bccd12bec`
- `readout.py`: `97dee44167ea9217494416542c24446943af5cce7720fe480d748a6cde45a16e`

Dependency versions at capture:

- torch: `2.14.1`
- torch-geometric: `2.8.0.post1`
- mlwiz: `1.7.6`
- scikit-learn: `1.9.1`

Keep this independent oracle when changing the current implementation. Do not regenerate expected values from the implementation being tested. Any replacement needs independently verified numerical evidence and an explanation of the intended behavior change. Historical source remains available in Git at the recorded revision.

<!-- GSPN-GPT-FIXED: Capture predictor behavior independently before the readability cleanup. -->
## Embedding predictor regression fixture

`predictor_regression.pt.gz` contains 18 cases captured before simplifying the predictors: all three predictor classes, sum/mean/max pooling, and float32/float64. Records include inputs, initial parameter states, both returned tensors, parameter gradients, one Adam update, and post-update outputs. Each case used seed 451, generated three Gaussian feature tensors for graphs with 3, 1, and 2 nodes, then constructed the predictor. Replaying those draws verifies unchanged native initialization. Load with the same tensor-only gzip procedure and numerical tolerances as the GSPN fixture.

Source revision: `014ce1c4e00c4c9634bbbeabf695f50d576f8256`. Source `readout.py` SHA-256: `3730ee97b612eb7f3c38dffda23b2e8cd3d9a9aed5f83214c42d28e54c448583`. PyTorch version: `2.14.1`. Fixture SHA-256: `78555dc1f408a526da0c34e6de19bbeabcd5b0b6ce85bb1a28d3d49fb838db8d`. The existing GSPN fixture is unchanged.

<!-- GSPN-GPT-FIXED: Record numerical and end-to-end validation of the readability cleanup. -->
## Readability cleanup validation

The cleanup passed 682 repository tests, lint, and formatting checks. The original 569-case GSPN fixture remained byte-identical. Across frozen regression and direct indexing checks, the largest absolute differences were `2.86102294921875e-6` in float32 and `4.440892098500626e-16` in float64, within the declared relative/absolute tolerances. Native predictor initialization, outputs, gradients, and Adam updates are checked by `tests/test_simplification.py`.

Fresh one-epoch CPU CLI runs for supervised GSPN and the combined pipeline were executed before and after editing. Assessment metrics matched; 279 tensor comparisons across model/optimizer checkpoints and cached graph embeddings had maximum absolute difference `0.0`. Runtime fields were excluded. The MLWiz audit reported zero errors and the existing 52 custom experiment/provider warnings. These runs are smoke tests, not scientific results.
