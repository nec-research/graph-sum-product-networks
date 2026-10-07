# GSPN-GPT-FIXED: Compare independent implementations, all outputs, gradients, and updates.
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from mlwiz.evaluation.grid import Grid
from mlwiz.experiment import Experiment
from mlwiz.log.logger import Logger
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops

import model as reference
import model_refactored as refactored
from sup_model import SupGSPN as ReferenceSupGSPN

# GSPN-GPT-FIXED: Include both categorical encodings, shared-component features, and NaNs.
EMISSIONS = {
    "gaussian": (2, "GSPNGaussianEmission"),
    "integer": (3, "GSPNCategoricalEmission"),
    "onehot": (3, "GSPNCategoricalEmission"),
    "multi": (2, "GSPNMultiCategoricalEmission"),
}
READOUTS = [
    "ProbabilisticGraphReadout",
    "ProbabilisticGraphReadoutNoLayerAttention",
    "ProbabilisticGraphReadoutNoLayerAttentionMLP",
    "ProbabilisticGraphReadoutNoLayerAttentionMLPVersion2",
]
ROOT = Path(__file__).resolve().parents[1]


def config(kind="gaussian", layers=2, shortcut=False, **kwargs):
    return {
        "num_layers": layers,
        "num_mixtures": 2,
        "num_hidden_neurons": 0,
        "emission_class": f"model.{EMISSIONS[kind][1]}",
        "convolution_class": "model.GSPNBaseConv",
        "dim_categorical_features": [2, 3],
        "avg_parameters_across_layers": shortcut,
        "num_graph_mixtures": 3,
        "global_readout": "mean",
        "global_pooling": "mean",
        "graph_emission_class": "model.GSPNCategoricalEmission",
        **kwargs,
    }


def graphs(kind="gaussian", mask="feature", dtype=torch.float64, unknown=False, edgeless=False):
    if kind == "gaussian":
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [2.0, 1.0], [4.0, 3.0]], dtype=dtype)
    elif kind in ("integer", "onehot"):
        labels = torch.tensor([0, 1, 2, 1, 0])
        x = (
            labels[:, None].to(dtype)
            if kind == "integer"
            else torch.nn.functional.one_hot(labels, num_classes=3).to(dtype)
        )
    else:
        x = torch.tensor([[0.0, 1.0], [1.0, 2.0], [0.0, 0.0], [1.0, 1.0], [0.0, 2.0]], dtype=dtype)
    observed = torch.tensor([True, False, True, True, False])
    if mask == "feature":
        observed = observed[:, None].expand_as(x).clone()
        if kind in ("gaussian", "multi"):
            observed[0, 1] = False
            observed[3, 0] = False
    if unknown:
        expanded = observed[:, None].expand_as(x) if observed.ndim == 1 else observed
        x[~expanded] = -1 if kind == "multi" else float("nan")
    edges = [torch.tensor([[0, 1], [1, 0]]), torch.tensor([[0], [1]])]
    if edgeless:
        edges = [torch.empty(2, 0, dtype=torch.long)] * 2
    result = []
    for i, (start, end) in enumerate([(0, 3), (3, 5)]):
        graph = Data(x=x[start:end], edge_index=edges[i], y=torch.tensor([i]))
        if mask != "none":
            graph.mask = observed[start:end]
        result.append(graph)
    return Batch.from_data_list(result)


# GSPN-GPT-FIXED: Shape/dtype/NaN parity is checked alongside maximum finite numeric error.
def assert_close(actual, expected, path="outputs"):
    if isinstance(expected, torch.Tensor):
        rtol, atol = (1e-8, 1e-10) if expected.dtype == torch.float64 else (1e-5, 1e-6)
        error = 0.0
        if expected.is_floating_point():
            valid = torch.isfinite(actual) & torch.isfinite(expected)
            if valid.any():
                error = (actual[valid] - expected[valid]).abs().max().item()
        torch.testing.assert_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            msg=lambda message: f"{path}: max finite absolute difference={error}\n{message}",
        )
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_close(a, e, f"{path}[{i}]")
    else:
        assert actual is expected


def models(kind, cfg, supervised=False, dtype=torch.float64, target_dim=2):
    old_type = ReferenceSupGSPN if supervised else reference.GSPN
    new_type = refactored.SupGSPN if supervised else refactored.GSPN
    torch.manual_seed(123)
    old = old_type((EMISSIONS[kind][0], 0), target_dim, cfg).to(dtype)
    new = new_type((EMISSIONS[kind][0], 0), target_dim, cfg).to(dtype)
    new.load_reference_state_dict(old.state_dict())
    return old, new


def assert_unchanged(data, before):
    assert set(data.keys()) == set(before.keys())
    for key in sorted(before.keys()):
        assert_close(data[key], before[key], f"input.{key}")


@pytest.mark.parametrize("kind", EMISSIONS)
@pytest.mark.parametrize("layers", [1, 2, 4])
@pytest.mark.parametrize("shortcut", [False, True])
@pytest.mark.parametrize("mask", ["none", "whole", "feature"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("supervised", [False, True])
def test_all_outputs_equivalent(kind, layers, shortcut, mask, dtype, supervised):
    cfg = config(kind, layers, shortcut)
    old, new = models(kind, cfg, supervised, dtype)
    data = graphs(kind, mask, dtype)
    before = data.clone()
    expected, actual = old(data), new(data)
    assert len(actual[2]) == 10
    assert_close(actual, expected)
    assert_unchanged(data, before)
    result = new.infer(data)
    assert_close(result.to_reference_outputs(), actual)
    assert_close(new.encode(data), actual[1])
    assert_close(result.node_posteriors.sum(-1), torch.ones_like(result.node_posteriors[..., 0]))
    if kind == "multi":
        for block in result.imputation.split([2, 3], dim=-1):
            assert_close(block.sum(-1), torch.ones_like(block[:, 0]))


@pytest.mark.parametrize("kind", EMISSIONS)
@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("edgeless", [False, True])
def test_missing_truth_and_topology_parity(kind, unknown, edgeless):
    old, new = models(kind, config(kind, layers=3, shortcut=True))
    data = graphs(kind, unknown=unknown, edgeless=edgeless)
    before = data.clone()
    assert_close(new(data), old(data))
    assert_unchanged(data, before)
    if unknown:
        assert torch.isnan(new(data)[2][9][1])


# GSPN-GPT-FIXED: Readout parity covers all variants and preserves absent/unused gradients.
@pytest.mark.parametrize("readout", READOUTS)
@pytest.mark.parametrize("pooling", ["sum", "mean"])
@pytest.mark.parametrize("shortcut", [False, True])
@pytest.mark.parametrize("kind", ["gaussian", "multi"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_readout_outputs_and_gradients(readout, pooling, shortcut, kind, dtype):
    cfg = config(
        kind, layers=3, shortcut=shortcut, readout=f"readout.{readout}", global_pooling=pooling
    )
    old, new = models(kind, cfg, dtype=dtype)
    data = graphs(kind, dtype=dtype)
    expected, actual = old(data), new(data)
    assert_close(actual, expected)
    loss(expected).backward()
    loss(actual).backward()
    assert_gradients(old, new)
    no_targets = data.clone()
    del no_targets.y
    unlabeled = new(no_targets)
    assert_close(unlabeled[0], actual[0])
    assert unlabeled[2][1] is None
    assert_close(new.encode(no_targets), actual[1])


@pytest.mark.parametrize("readout", [r for r in READOUTS if r != READOUTS[2]])
def test_gaussian_graph_emission(readout):
    cfg = config(readout=f"readout.{readout}", graph_emission_class="model.GSPNGaussianEmission")
    old, new = models("gaussian", cfg, target_dim=1)
    data = graphs()
    data.y = torch.tensor([[0.5], [1.5]], dtype=torch.float64)
    assert_close(new(data), old(data))


def loss(outputs):
    predictions, embeddings, extras = outputs
    value = -extras[0].mean() + embeddings.square().mean() + extras[4].square().mean()
    if predictions is not None:
        value = value + predictions.square().mean()
    if extras[1] is not None:
        value = value - extras[1].mean()
    return value


def assert_gradients(old, new):
    old_parameters, new_parameters = dict(old.named_parameters()), dict(new.named_parameters())
    mapping = new.reference_state_mapping()
    for key, parameter in old_parameters.items():
        gradient = new_parameters[mapping[key]].grad
        assert (gradient is None) == (parameter.grad is None), key
        if gradient is not None:
            assert torch.isfinite(gradient).all(), key
            assert_close(gradient, parameter.grad, f"gradient.{key}")


@pytest.mark.parametrize("kind", EMISSIONS)
@pytest.mark.parametrize("layers", [1, 2, 4])
@pytest.mark.parametrize("shortcut", [False, True])
@pytest.mark.parametrize("supervised", [False, True])
@pytest.mark.parametrize("pooling", ["sum", "mean"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradients_and_optimizer_update(kind, layers, shortcut, supervised, pooling, dtype):
    old, new = models(
        kind, config(kind, layers, shortcut, global_readout=pooling), supervised, dtype
    )
    data = graphs(kind, dtype=dtype)
    old_optimizer = torch.optim.Adam(old.parameters(), lr=0.01)
    new_optimizer = torch.optim.Adam(new.parameters(), lr=0.01)
    loss(old(data)).backward()
    loss(new(data)).backward()
    assert_gradients(old, new)
    old_optimizer.step()
    new_optimizer.step()
    for old_key, new_key in new.reference_state_mapping().items():
        assert_close(new.state_dict()[new_key], old.state_dict()[old_key], f"state.{old_key}")
    assert_close(new(data), old(data))


# GSPN-GPT-FIXED: Shared RNG states prove initialization parity rather than bypassing it.
@pytest.mark.parametrize("samples", [1, 2, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_kmeans_equivalence_and_training_guard(samples, dtype):
    cfg = config(init_kmeans=True, init_max_variance=4.0, num_mixtures=4, seed=27)
    old, new = models("gaussian", cfg, dtype=dtype)
    x = torch.arange(samples * 2, dtype=dtype).reshape(samples, 2)
    observed = torch.ones_like(x, dtype=torch.bool)
    observed[:, 1] = False
    x[:, 1] = 123456
    data = Batch.from_data_list(
        [Data(x=x, mask=observed, edge_index=torch.empty(2, 0, dtype=torch.long))]
    )
    before = data.clone()
    old.eval()
    new.eval()
    assert_close(new(data), old(data))
    assert not old.initialized.item() and not new.initialized.item()
    old.train()
    new.train()
    rng = torch.get_rng_state()
    expected = old(data)
    torch.set_rng_state(rng)
    actual = new(data)
    assert_close(actual, expected)
    for old_key, new_key in new.reference_state_mapping().items():
        assert_close(new.state_dict()[new_key], old.state_dict()[old_key])
    assert old.initialized.item() and new.initialized.item()
    for emission in new.emissions:
        assert emission.mean.shape == (4, 2)
        assert torch.all(emission.mean[:, 1] == 0)
        assert emission.distribution_parameters().stddev.square().max() <= 4.0001
    state = deepcopy(new.state_dict())
    new.initialize_from_evidence(x + 100, observed)
    for key in state:
        assert_close(new.state_dict()[key], state[key])
    restored = refactored.GSPN((2, 0), 2, cfg).to(dtype)
    restored.load_reference_state_dict(old.state_dict())
    assert_close(restored(data), actual)
    assert_unchanged(data, before)


def test_explicit_initialization_and_invalid_emission():
    cfg = config(init_kmeans=True)
    old, new = models("gaussian", cfg)
    data = graphs()
    rng = torch.get_rng_state()
    old(data)
    torch.set_rng_state(rng)
    new.initialize_from_evidence(data.x, data.mask)
    assert_close(new(data), old(data))
    with pytest.raises(ValueError, match="Gaussian"):
        refactored.GSPN((3, 0), 2, config("integer", init_kmeans=True))


# GSPN-GPT-FIXED: Fast paths must actually skip expensive diagnostic and head operations.
def test_encode_skips_optional_work(monkeypatch):
    cfg = config(readout="readout.ProbabilisticGraphReadout")
    _, new = models("gaussian", cfg)
    data = graphs()
    complete = new.infer(data)

    def fail(*args, **kwargs):
        raise AssertionError("Optional work was executed")

    monkeypatch.setattr(new.emissions[-1], "predictive_mean", fail)
    monkeypatch.setattr(new, "_evaluate_missing_features", fail)
    without_optional = new.infer(data, include_imputation=False, include_diagnostics=False)
    assert without_optional.imputation is None and without_optional.conditional_log_prob is None
    assert_close(without_optional.predictions, complete.predictions)
    monkeypatch.setattr(new.head, "forward", fail)
    embeddings = new.encode(data)
    assert_close(embeddings, complete.embeddings)
    embeddings.square().sum().backward()
    assert any(p.grad is not None for p in new.emissions[0].parameters())


# GSPN-GPT-FIXED: One-loop policy is deliberate; non-self parallel edges remain intact.
def test_self_loop_normalization_and_reference():
    cfg = config(layers=3)
    old, new = models("gaussian", cfg)
    data = graphs()
    loops = torch.arange(data.num_nodes).repeat(2, 1)
    data.edge_index = torch.cat((data.edge_index, data.edge_index[:, :1], loops, loops), dim=1)
    before = data.clone()
    prepared = new._prepare_evidence(data)
    own = prepared.edge_index[0] == prepared.edge_index[1]
    assert_close(prepared.edge_index[:, own], loops)
    original_edges, _ = remove_self_loops(data.edge_index)
    assert_close(prepared.edge_index[:, ~own], original_edges)
    single_loop_reference = data.clone()
    single_loop_reference.edge_index = original_edges
    assert_close(new(data), old(single_loop_reference))
    assert_unchanged(data, before)


def test_duplicate_loop_weighting_changes_connected_nodes_only():
    cfg = config("integer", add_self_loops=True)
    old = reference.GSPN((2, 0), 0, cfg).double()
    with torch.no_grad():
        for emission in old.emissions:
            emission.categorical_probs.copy_(torch.tensor([[0.9, 0.1], [0.1, 0.9]]).log())
        old.transitions[0].transition_table.fill_(0.0)
        old.transitions[1].transition_table.copy_(torch.tensor([[0.9, 0.1], [0.1, 0.9]]).log())
    new = refactored.GSPN(
        (2, 0), 0, {k: v for k, v in cfg.items() if k != "add_self_loops"}
    ).double()
    new.load_reference_state_dict(old.state_dict())
    data = Batch.from_data_list(
        [
            Data(
                x=torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float64),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
            )
        ]
    )
    doubled, single = old(data), new(data)
    assert not torch.allclose(doubled[2][7][:2], single[2][7][:2])
    assert_close(doubled[2][7][2], single[2][7][2])
    old.config["add_self_loops"] = False
    old.add_self_loops = False
    assert_close(new(data), old(data))


@pytest.mark.parametrize("value", [False, True])
def test_removed_loop_option(value):
    with pytest.raises(ValueError, match="Remove add_self_loops"):
        refactored.GSPN((2, 0), 2, config(add_self_loops=value))


# GSPN-GPT-FIXED: Invalid state transfers fail atomically, before any parameter is copied.
@pytest.mark.parametrize("invalid", ["missing", "unexpected", "shape", "layers"])
def test_invalid_reference_state(invalid):
    old, new = models("gaussian", config())
    state = deepcopy(old.state_dict())
    if invalid == "missing":
        state.pop("initialized")
    elif invalid == "unexpected":
        state["unrecognized"] = torch.tensor(0.0)
    elif invalid == "shape":
        state["emissions.1.normal_params"] = torch.zeros(1, 1, 2)
    else:
        state = reference.GSPN((2, 0), 2, config(layers=3)).state_dict()
    before = deepcopy(new.state_dict())
    with pytest.raises(ValueError, match="reference"):
        new.load_reference_state_dict(state)
    for key in before:
        assert_close(new.state_dict()[key], before[key])


def test_extension_contracts_and_new_paths():
    class CustomGaussian(refactored.GSPNGaussianEmission):
        pass

    class CustomTransition(refactored.GSPNBaseConv):
        pass

    class CustomHead(refactored.ProbabilisticGraphReadout):
        pass

    cfg = config(
        emission_class=CustomGaussian, convolution_class=CustomTransition, readout=CustomHead
    )
    output = refactored.GSPN((2, 0), 2, cfg).double()(graphs())
    assert output[0].shape == (2, 2)
    cfg = config(
        emission_class="model_refactored.GSPNGaussianEmission",
        convolution_class="model_refactored.GSPNBaseConv",
    )
    assert refactored.GSPN((2, 0), 2, cfg).encode(graphs()).shape == (5, 4)
    with pytest.raises(TypeError, match="contract"):
        refactored.GSPN((2, 0), 2, config(emission_class="torch.nn.Linear"))
    with pytest.raises(TypeError):
        refactored.GSPNEmission(2, 2)
    with pytest.raises(ValueError, match="at least one node"):
        refactored.GSPN((2, 0), 2, config())(
            Data(x=torch.empty(0, 2), edge_index=torch.empty(2, 0, dtype=torch.long))
        )


# GSPN-GPT-FIXED: Independent analytic checks complement comparisons with the reference code.
def test_component_densities_and_shortcut_parameters():
    emission = refactored.GSPNMultiCategoricalEmission(2, 2, [2, 3]).double()
    params = refactored.MultiCategoricalParameters(
        (
            refactored.CategoricalParameters(
                torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float64)
            ),
            refactored.CategoricalParameters(
                torch.tensor([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]], dtype=torch.float64)
            ),
        )
    )
    x = torch.cartesian_prod(torch.arange(2), torch.arange(3))
    prior = torch.tensor([[0.4, 0.6]], dtype=torch.float64).expand(6, -1)
    components = emission.component_log_prob(x, params=params)
    log_prob, posterior = refactored.infer_mixture(components, prior)
    assert_close(log_prob.exp().sum(), torch.tensor(1.0, dtype=torch.float64))
    assert_close(
        log_prob[0].exp(), torch.tensor(0.4 * 0.8 * 0.1 + 0.6 * 0.1 * 0.7, dtype=torch.float64)
    )
    masked = emission.component_log_prob(x, params=params, observed_mask=torch.tensor([False] * 6))
    missing_ll, missing_posterior = refactored.infer_mixture(masked, prior)
    assert_close(missing_ll, torch.zeros(6, dtype=torch.float64))
    assert_close(missing_posterior, prior)
    assert_close(posterior.sum(-1), torch.ones(6, dtype=torch.float64))
    imputation = emission.predictive_mean(params, weights=posterior)
    assert_close(imputation[:, :2].sum(-1), torch.ones(6, dtype=torch.float64))
    assert_close(imputation[:, 2:].sum(-1), torch.ones(6, dtype=torch.float64))
    categorical = emission.emissions[0]
    combined = categorical.combine_shortcut_parameters([params.blocks[0], params.blocks[0]])
    assert_close(combined.probabilities, params.blocks[0].probabilities)

    gaussian = refactored.GSPNGaussianEmission(2, 2).double()
    gparams = refactored.GaussianParameters(
        torch.tensor([[0.0, 4.0], [5.0, 8.0]], dtype=torch.float64),
        torch.tensor([[2.0, 3.0], [1.0, 2.0]], dtype=torch.float64),
    )
    data = torch.tensor([[1.0, float("nan")]], dtype=torch.float64)
    before = data.clone()
    components = gaussian.component_log_prob(
        data, params=gparams, observed_mask=torch.tensor([[True, False]])
    )
    analytic = torch.distributions.Normal(gparams.mean[:, 0], gparams.stddev[:, 0]).log_prob(
        data[0, 0]
    )
    assert_close(components[0], analytic)
    assert_close(data, before)
    combined = gaussian.combine_shortcut_parameters([gparams, gparams])
    assert_close(combined.mean, gparams.mean)
    assert_close(combined.stddev, gparams.stddev / 2**0.5)


def test_invalid_observed_categories_and_partial_onehot_mask():
    emission = refactored.GSPNCategoricalEmission(3, 2)
    params = emission.distribution_parameters()
    with pytest.raises(ValueError, match="integer category IDs"):
        emission.component_log_prob(torch.tensor([[3.0]]), params=params)
    with pytest.raises(ValueError, match="masked as a whole"):
        emission.component_log_prob(
            torch.tensor([[1.0, 0.0, 0.0]]),
            params=params,
            observed_mask=torch.tensor([[True, False, True]]),
        )


def test_supervised_without_head_and_dictionary_categories():
    cfg = config(
        "multi", num_graph_mixtures=None, dim_categorical_features={"first": 2, "second": 3}
    )
    old, new = models("multi", cfg, supervised=True)
    assert new(graphs("multi"))[0] is None
    assert_close(new(graphs("multi")), old(graphs("multi")))


# GSPN-GPT-FIXED: Standard-engine CPU smoke uses isolated temporary artifacts and real losses.
@pytest.mark.parametrize("supervised", [False, True])
def test_cpu_training_engine(tmp_path, supervised):
    cfg = Grid(yaml.safe_load((ROOT / "configs/smoke_supervised.yml").read_text()))[0]
    cfg.update(
        model=f"model_refactored.{'SupGSPN' if supervised else 'GSPN'}", epochs=1, checkpoint=False
    )
    if not supervised:
        cfg["loss"] = "metric.GSPNNodeLogLikelihood"
        cfg["scorer"] = "metric.GSPNNodeLogLikelihood"
    experiment = Experiment(cfg, str(tmp_path), 42)
    model = experiment.create_model((2, 0), 2, experiment.model_config)
    engine = experiment.create_engine(experiment.model_config, model)
    data = graphs(dtype=torch.float32).to_data_list()
    loader = DataLoader([(graph, graph.y) for graph in data], batch_size=2)
    metrics = engine.train(
        train_loader=loader,
        validation_loader=loader,
        test_loader=None,
        max_epochs=1,
        logger=Logger(str(tmp_path / "training.log"), "a", debug=False),
        training_timeout_seconds=30,
        progress_callback=None,
        should_terminate=lambda: False,
    )
    assert len(metrics) == 6
    assert any(p.grad is not None for p in model.parameters())
    assert torch.isfinite(model.encode(graphs(dtype=torch.float32))).all()
