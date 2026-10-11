# GSPN models

[Back to the experiment instructions](../README.md).

## Likelihoods and outputs

The implementation now marginalizes missing categorical and Gaussian evidence, combines multi-categorical features within a shared mixture component, computes posteriors in log space, and imputes with posterior weights. Categorical imputation returns probabilities; multi-categorical output concatenates one normalized block per feature. One-hot categorical observations must be masked as a whole; integer categorical features can be masked independently.

Shortcuts use preceding layers. Gaussian shortcuts combine variances according to the distribution of an average, rather than averaging standard deviations. Gaussian K-means uses observed training evidence only and honors `init_max_variance`; small batches repeat fitted centers without changing the configured mixture count.

MLWiz models accept `(dim_input_features, dim_target, config)`, with graph dimensions `(node_width, edge_width)`. Dataset samples are `(graph, target)`. The standard engine returns six metric dictionaries; the pipeline extracts node embeddings explicitly. Model outputs remain `(predictions, node_embeddings, extras)`. The first nine extra positions keep their meanings:

| Index | Value |
| --- | --- |
| 0–1 | Node and optional graph log likelihood |
| 2–3 | Original input features |
| 4 | Posterior imputation means/probabilities |
| 5–6 | Missing and observed masks, or `None` without a mask |
| 7–8 | Final prior mixture weights and emission parameters |
| 9 | Conditional missing-feature log likelihood under the same final mixture |

Conditional likelihood is complete minus observed log likelihood. Missing ground truth is excluded from evaluation; an unavailable conditional score is `NaN`, and missing-feature metrics skip it. Completely missing evidence has zero log likelihood and preserves the prior as posterior.

Existing PyDGN checkpoints and embedding caches need regeneration. Outputs use new MLWiz roots and dataset/configuration-specific experiment names; original data and splits are preserved. Analysis notebooks load current final-run artifacts through `notebook_utils.load_run`, and their historical outputs have been cleared. Two configurations referencing absent original modules/classes are retained as `.yml.disabled` files under [`ARCHIVED_CONFIGS`](../ARCHIVED_CONFIGS/README.md); arbitrary SPN templates and Bernoulli extensions are outside this migration.


## Using and extending the models

`model.GSPN` and `model.SupGSPN` are the sole GSPN implementations. Both models, emissions, transitions, and probabilistic graph heads live in `model.py`. Experiment configurations use these canonical paths. `readout.py` retains the embedding predictor classes used by the combined pipeline.

```python
# Use the canonical model with named inference and embedding extraction.
from model import GSPN, SupGSPN

config = {
    "num_layers": 2,
    "num_mixtures": 3,
    "emission_class": "model.GSPNGaussianEmission",
    "avg_parameters_across_layers": False,
}
gspn = GSPN((2, 0), 2, config)

# batch is a PyG graph batch with two input features per node.
predictions, embeddings, extras = gspn(batch)  # Existing ten auxiliary outputs.
result = gspn.infer(batch)                    # Named InferenceResult.
node_posteriors = result.node_posteriors       # [nodes, layers, components].
final_posterior = result.layers[-1].posterior
embeddings_only = gspn.encode(batch)          # Skips readout/imputation/diagnostics.
```

`infer(batch, include_imputation=False, include_diagnostics=False)` skips the corresponding optional computations. Graph heads predict without `batch.y`; their graph likelihood is `None` when targets are absent. All inference paths retain gradients. Gaussian initialization is available through `initialize_from_evidence(x, observed_mask)`; it also runs automatically on the first training batch when enabled, and never during evaluation. `observed_mask=True` means observed.

Graph preparation ensures **exactly one self-loop per node**, removing existing self-loop duplicates while retaining non-self edges and their multiplicity. The `add_self_loops` key is rejected and has been removed from active configurations. Isolated nodes receive their own message. This approved loop normalization is the only intentional mathematical difference from the previous implementation.

Custom emissions subclass `GSPNEmission` and implement its component-density, predictive-mean, shortcut, parameter, and value-validity methods. Custom transitions subclass `GSPNBaseConv`, and custom heads subclass `GraphHead`. Their constructor contracts are `(dim_observable, num_components)` via `from_dimensions(...)`, `(num_components, use_prior)`, and `(dim_target, config)`, respectively.

The retired `model_refactored.py`, `sup_model.py`, and duplicate probabilistic heads are removed. Update external imports to `model`; probabilistic readout paths now use `model.ProbabilisticGraphReadout...`. Native saves use `state_dict()` and `load_state_dict()`. For an existing legacy **tensor state dictionary**, `load_reference_state_dict()` validates and translates every parameter and initialization buffer before updating the model. Matching architecture and tensor shapes are required; this does not convert old optimizer states or provide automatic experiment resumption.

GSPN experiment names now end in `_refactored` to create fresh result directories. Existing result artifacts, raw data, and saved splits remain intact. Regenerate framework checkpoints and embeddings for these runs; the embedding cache also includes the changed code fingerprint.

```sh
uv run pytest tests/test_model.py
```

Regression tests replay 569 frozen cases captured from the previously verified original implementation, without importing its code. They check predictions, embeddings, all ten auxiliary outputs, parameter gradients, optimizer updates, and initialization. Additional tests cover loop normalization, named interfaces, extension contracts, input preservation, and CPU training-engine runs. [Fixture provenance](../tests/fixtures/README.md) records the source revision and checksums. Float64 comparisons use `rtol=1e-8, atol=1e-10`; float32 comparisons use `rtol=1e-5, atol=1e-6`. Failures report the maximum finite absolute difference.
