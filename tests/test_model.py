# GSPN-GPT-FIXED: Frozen numerical regression replaces the retired implementation as oracle.
import gzip
from copy import deepcopy
from functools import lru_cache
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

import model

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


# GSPN-GPT-FIXED: Load tensors only; the oracle contains no executable legacy model code.
@lru_cache(maxsize=1)
def snapshots():
    with gzip.open(ROOT / "tests/fixtures/gspn_regression.pt.gz", "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=True)


def cases(group):
    return [
        pytest.param(case, id=f"{group}-{i}-{case['kind']}-L{case['config']['num_layers']}")
        for i, case in enumerate(snapshots()["groups"][group])
    ]


def replay(case):
    model_type = model.SupGSPN if case["supervised"] else model.GSPN
    new = model_type(case["input_dimensions"], case["target_dim"], case["config"])
    new.to(case["data"]["x"].dtype)
    new.load_reference_state_dict(case["initial_state"])
    return new, Batch(**deepcopy(case["data"]))


def assert_state(new, expected):
    for old_key, new_key in new.reference_state_mapping().items():
        assert_close(new.state_dict()[new_key], expected[old_key], f"state.{old_key}")


def assert_gradients(new, expected):
    parameters = dict(new.named_parameters())
    mapping = new.reference_state_mapping()
    assert {mapping[key] for key in expected} == set(parameters)
    for key, gradient in expected.items():
        actual = parameters[mapping[key]].grad
        assert (actual is None) == (gradient is None), key
        if actual is not None:
            assert torch.isfinite(actual).all(), key
            assert_close(actual, gradient, f"gradient.{key}")


# GSPN-GPT-FIXED: Preserve all prior output comparisons against captured independent evidence.
@pytest.mark.parametrize("case", cases("outputs"))
def test_all_outputs_regression(case):
    new, data = replay(case)
    before = data.clone()
    actual = new(data)
    assert len(actual[2]) == 10
    assert_close(actual, case["outputs"])
    assert_unchanged(data, before)
    result = new.infer(data)
    assert_close(result.to_reference_outputs(), actual)
    assert_close(new.encode(data), actual[1])
    assert_close(result.node_posteriors.sum(-1), torch.ones_like(result.node_posteriors[..., 0]))
    if case["kind"] == "multi":
        for block in result.imputation.split([2, 3], dim=-1):
            assert_close(block.sum(-1), torch.ones_like(block[:, 0]))


@pytest.mark.parametrize("case", cases("topology"))
def test_missing_truth_and_topology_regression(case):
    new, data = replay(case)
    before = data.clone()
    assert_close(new(data), case["outputs"])
    assert_unchanged(data, before)


@pytest.mark.parametrize("case", cases("readouts"))
def test_readout_outputs_and_gradients_regression(case):
    new, data = replay(case)
    actual = new(data)
    assert_close(actual, case["outputs"])
    loss(actual).backward()
    assert_gradients(new, case["gradients"])
    no_targets = data.clone()
    del no_targets.y
    unlabeled = new(no_targets)
    assert_close(unlabeled[0], actual[0])
    assert unlabeled[2][1] is None
    assert_close(new.encode(no_targets), actual[1])


@pytest.mark.parametrize("case", cases("gaussian_readouts"))
def test_gaussian_graph_emission_regression(case):
    new, data = replay(case)
    assert_close(new(data), case["outputs"])


@pytest.mark.parametrize("case", cases("updates"))
def test_gradients_and_optimizer_update_regression(case):
    new, data = replay(case)
    optimizer = torch.optim.Adam(new.parameters(), lr=0.01)
    actual = new(data)
    assert_close(actual, case["outputs"])
    loss(actual).backward()
    assert_gradients(new, case["gradients"])
    optimizer.step()
    assert_state(new, case["updated_state"])
    assert_close(new(data), case["updated_outputs"])


# GSPN-GPT-FIXED: Replay original initialization with its captured RNG state and observed evidence.
@pytest.mark.parametrize("case", cases("initialization"))
def test_kmeans_regression_and_training_guard(case):
    new, data = replay(case)
    before = data.clone()
    new.eval()
    assert_close(new(data), case["evaluation_outputs"])
    assert not new.initialized.item()
    new.train()
    torch.set_rng_state(case["rng_state"])
    actual = new(data)
    assert_close(actual, case["outputs"])
    assert_state(new, case["initialized_state"])
    assert new.initialized.item()
    for emission in new.emissions:
        assert emission.mean.shape == (4, 2)
        assert torch.all(emission.mean[:, 1] == 0)
        assert emission.distribution_parameters().stddev.square().max() <= 4.0001
    state = deepcopy(new.state_dict())
    new.initialize_from_evidence(data.x + 100, data.mask)
    restored, _ = replay(case)
    restored.load_state_dict(state)
    assert_close(restored(data), actual)
    for key in state:
        assert_close(new.state_dict()[key], state[key])
    assert_unchanged(data, before)


def test_explicit_initialization_ignores_hidden_truth():
    case = snapshots()["groups"]["initialization"][-1]
    new, data = replay(case)
    torch.set_rng_state(case["rng_state"])
    new.initialize_from_evidence(data.x, data.mask)
    assert_close(new(data), case["outputs"])
    changed, _ = replay(case)
    hidden = data.x.clone()
    hidden[~data.mask] = float("nan")
    torch.set_rng_state(case["rng_state"])
    changed.initialize_from_evidence(hidden, data.mask)
    assert_state(changed, case["initialized_state"])
    with pytest.raises(ValueError, match="Gaussian"):
        model.GSPN((3, 0), 2, config("integer", init_kmeans=True))


# GSPN-GPT-FIXED: Keep operation-skipping and self-loop tests independent of fixture data.
def test_encode_skips_optional_work(monkeypatch):
    new = model.GSPN((2, 0), 2, config(readout="model.ProbabilisticGraphReadout")).double()
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


def test_self_loop_normalization():
    new = model.GSPN((2, 0), 2, config(layers=3)).double()
    data = graphs()
    loops = torch.arange(data.num_nodes).repeat(2, 1)
    data.edge_index = torch.cat((data.edge_index, data.edge_index[:, :1], loops, loops), dim=1)
    before = data.clone()
    prepared = new._prepare_evidence(data)
    own = prepared.edge_index[0] == prepared.edge_index[1]
    assert_close(prepared.edge_index[:, own], loops)
    original_edges, _ = remove_self_loops(data.edge_index)
    assert_close(prepared.edge_index[:, ~own], original_edges)
    loop_free = data.clone()
    loop_free.edge_index = original_edges
    assert_close(new(data), new(loop_free))
    assert_unchanged(data, before)


def test_isolated_node_message_and_neighbor_average():
    transition = model.GSPNBaseConv(2, use_prior=False).double()
    with torch.no_grad():
        transition.transition_logits.copy_(torch.tensor([[0.9, 0.1], [0.1, 0.9]]).log())
    posteriors = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4]], dtype=torch.float64)
    graph = Data(x=torch.zeros(3, 1), edge_index=torch.tensor([[0, 1], [1, 0]]))
    prepared = model.GSPN((1, 0), 0, config())._prepare_evidence(graph)
    actual = transition(posteriors, edge_index=prepared.edge_index, num_nodes=3)
    transformed = posteriors @ torch.softmax(transition.transition_logits, dim=0).T
    assert_close(actual[:2], ((transformed[0] + transformed[1]) / 2).expand(2, -1))
    assert_close(actual[2], transformed[2])
    # GSPN-GPT-FIXED: A second loop changes connected-node weights but leaves isolated nodes equal.
    loops = torch.arange(3).repeat(2, 1)
    duplicated = transition(
        posteriors, edge_index=torch.cat((prepared.edge_index, loops), dim=1), num_nodes=3
    )
    assert_close(duplicated[0], (2 * transformed[0] + transformed[1]) / 3)
    assert_close(duplicated[1], (transformed[0] + 2 * transformed[1]) / 3)
    assert_close(duplicated[2], actual[2])
    assert not torch.allclose(duplicated[:2], actual[:2])


@pytest.mark.parametrize("value", [False, True])
def test_removed_loop_option(value):
    with pytest.raises(ValueError, match="Remove add_self_loops"):
        model.GSPN((2, 0), 2, config(add_self_loops=value))


# GSPN-GPT-FIXED: Legacy tensor conversion remains useful without any executable old models.
@pytest.mark.parametrize("invalid", ["missing", "unexpected", "shape", "layers"])
def test_invalid_reference_state(invalid):
    case = snapshots()["groups"]["initialization"][0]
    new, _ = replay(case)
    state = deepcopy(case["initial_state"])
    if invalid == "missing":
        state.pop("initialized")
    elif invalid == "unexpected":
        state["unrecognized"] = torch.tensor(0.0)
    elif invalid == "shape":
        state["emissions.1.normal_params"] = torch.zeros(1, 1, 2)
    else:
        state["emissions.2.normal_params"] = state["emissions.1.normal_params"].clone()
    before = deepcopy(new.state_dict())
    with pytest.raises(ValueError, match="reference"):
        new.load_reference_state_dict(state)
    for key in before:
        assert_close(new.state_dict()[key], before[key])


def test_extension_contracts_and_canonical_paths():
    class CustomGaussian(model.GSPNGaussianEmission):
        pass

    class CustomTransition(model.GSPNBaseConv):
        pass

    class CustomHead(model.ProbabilisticGraphReadout):
        pass

    cfg = config(
        emission_class=CustomGaussian, convolution_class=CustomTransition, readout=CustomHead
    )
    output = model.GSPN((2, 0), 2, cfg).double()(graphs())
    assert output[0].shape == (2, 2)
    with pytest.raises(TypeError, match="contract"):
        model.GSPN((2, 0), 2, config(emission_class="torch.nn.Linear"))
    with pytest.raises(TypeError):
        model.GSPNEmission(2, 2)
    with pytest.raises(ValueError, match="at least one node"):
        model.GSPN((2, 0), 2, config())(
            Data(x=torch.empty(0, 2), edge_index=torch.empty(2, 0, dtype=torch.long))
        )


def test_supervised_without_head_and_dictionary_categories():
    cfg = config(
        "multi", num_graph_mixtures=None, dim_categorical_features={"first": 2, "second": 3}
    )
    new = model.SupGSPN((2, 0), 2, cfg).double()
    outputs = new(graphs("multi"))
    assert outputs[0] is None
    assert outputs[1].shape == (5, 4)


def test_fixture_provenance_and_retired_modules():
    fixture = snapshots()
    assert fixture["format_version"] == 1
    assert fixture["source_revision"] == "a9e33246bd0745ea874c7c6074a5b6a28a46090b"
    assert sum(len(group) for group in fixture["groups"].values()) == 569
    assert all(len(digest) == 64 for digest in fixture["source_sha256"].values())
    assert not (ROOT / "model_refactored.py").exists()
    assert not (ROOT / "sup_model.py").exists()


# GSPN-GPT-FIXED: Standard-engine smoke covers the canonical public model paths.
@pytest.mark.parametrize("supervised", [False, True])
def test_cpu_training_engine(tmp_path, supervised):
    cfg = Grid(yaml.safe_load((ROOT / "configs/smoke_supervised.yml").read_text()))[0]
    cfg.update(model=f"model.{'SupGSPN' if supervised else 'GSPN'}", epochs=1, checkpoint=False)
    if not supervised:
        cfg["loss"] = "metric.GSPNNodeLogLikelihood"
        cfg["scorer"] = "metric.GSPNNodeLogLikelihood"
    experiment = Experiment(cfg, str(tmp_path), 42)
    new = experiment.create_model((2, 0), 2, experiment.model_config)
    engine = experiment.create_engine(experiment.model_config, new)
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
    assert any(p.grad is not None for p in new.parameters())
    assert torch.isfinite(new.encode(graphs(dtype=torch.float32))).all()


def config(kind="gaussian", layers=2, shortcut=False, **kwargs):
    return {
        "num_layers": layers,
        "num_mixtures": 2,
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


def assert_unchanged(data, before):
    assert set(data.keys()) == set(before.keys())
    for key in sorted(before.keys()):
        assert_close(data[key], before[key], f"input.{key}")


def loss(outputs):
    predictions, embeddings, extras = outputs
    value = -extras[0].mean() + embeddings.square().mean() + extras[4].square().mean()
    if predictions is not None:
        value = value + predictions.square().mean()
    if extras[1] is not None:
        value = value - extras[1].mean()
    return value


# GSPN-GPT-FIXED: Independent analytic checks complement the frozen regression oracle.
def test_component_densities_and_shortcut_parameters():
    emission = model.GSPNMultiCategoricalEmission(2, 2, [2, 3]).double()
    params = model.MultiCategoricalParameters(
        (
            model.CategoricalParameters(
                torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float64)
            ),
            model.CategoricalParameters(
                torch.tensor([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]], dtype=torch.float64)
            ),
        )
    )
    x = torch.cartesian_prod(torch.arange(2), torch.arange(3))
    prior = torch.tensor([[0.4, 0.6]], dtype=torch.float64).expand(6, -1)
    components = emission.component_log_prob(x, params=params)
    log_prob, posterior = model.infer_mixture(components, prior)
    assert_close(log_prob.exp().sum(), torch.tensor(1.0, dtype=torch.float64))
    assert_close(
        log_prob[0].exp(), torch.tensor(0.4 * 0.8 * 0.1 + 0.6 * 0.1 * 0.7, dtype=torch.float64)
    )
    masked = emission.component_log_prob(x, params=params, observed_mask=torch.tensor([False] * 6))
    missing_ll, missing_posterior = model.infer_mixture(masked, prior)
    assert_close(missing_ll, torch.zeros(6, dtype=torch.float64))
    assert_close(missing_posterior, prior)
    assert_close(posterior.sum(-1), torch.ones(6, dtype=torch.float64))
    imputation = emission.predictive_mean(params, weights=posterior)
    assert_close(imputation[:, :2].sum(-1), torch.ones(6, dtype=torch.float64))
    assert_close(imputation[:, 2:].sum(-1), torch.ones(6, dtype=torch.float64))
    categorical = emission.emissions[0]
    combined = categorical.combine_shortcut_parameters([params.blocks[0], params.blocks[0]])
    assert_close(combined.probabilities, params.blocks[0].probabilities)

    gaussian = model.GSPNGaussianEmission(2, 2).double()
    gparams = model.GaussianParameters(
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
    emission = model.GSPNCategoricalEmission(3, 2)
    params = emission.distribution_parameters()
    with pytest.raises(ValueError, match="integer category IDs"):
        emission.component_log_prob(torch.tensor([[3.0]]), params=params)
    with pytest.raises(ValueError, match="masked as a whole"):
        emission.component_log_prob(
            torch.tensor([[1.0, 0.0, 0.0]]),
            params=params,
            observed_mask=torch.tensor([[True, False, True]]),
        )
