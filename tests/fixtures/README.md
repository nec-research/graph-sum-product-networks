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
