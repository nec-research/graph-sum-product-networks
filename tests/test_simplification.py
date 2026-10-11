# Independent regressions verify readability changes preserve behavior.
import gzip
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

import readout
from model import CategoricalParameters, GSPNCategoricalEmission

ROOT = Path(__file__).resolve().parents[1]
with gzip.open(ROOT / "tests/fixtures/predictor_regression.pt.gz", "rb") as stream:
    PREDICTOR_FIXTURE = torch.load(stream, map_location="cpu", weights_only=True)


def assert_close(actual, expected):
    tolerance = (1e-8, 1e-10) if expected.dtype == torch.float64 else (1e-5, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["integer", "column", "onehot"])
@pytest.mark.parametrize("mask_kind", ["absent", "partial", "missing"])
def test_categorical_indexing_outputs_and_gradients(dtype, layout, mask_kind):
    labels = torch.tensor([2, 0, 2, 1, 0, 1, 2])
    observed = torch.ones(7, dtype=torch.bool)
    if mask_kind == "partial":
        observed[1::2] = False
    elif mask_kind == "missing":
        observed[:] = False
    if layout == "onehot":
        backing = torch.zeros(7, 6, dtype=dtype)
        x = backing[:, ::2]
        x.scatter_(1, labels[:, None], 1)
        x[~observed] = float("nan")
    else:
        backing = torch.empty(14, dtype=dtype)
        x = backing[::2]
        x.copy_(labels)
        x[~observed] = float("nan")
        if layout == "column":
            x = x[:, None]
    assert not x.is_contiguous()
    before = x.clone()
    logits = torch.tensor([[0.2, -0.1, 0.8], [-0.3, 0.4, 0.1]], dtype=dtype, requires_grad=True)
    probabilities = logits.softmax(1)
    safe_labels = torch.where(observed, labels, 0)
    gathered = (
        probabilities[None]
        .expand(7, -1, -1)
        .gather(2, safe_labels[:, None, None].expand(-1, 2, 1))
        .squeeze(2)
        .log()
    )
    expected = torch.where(observed[:, None], gathered, torch.zeros_like(gathered))
    actual = GSPNCategoricalEmission(3, 2).component_log_prob(
        x,
        params=CategoricalParameters(probabilities),
        observed_mask=None if mask_kind == "absent" else observed,
    )
    assert_close(actual, expected)
    weights = torch.arange(1, 15, dtype=dtype).reshape(7, 2)
    old_gradient = torch.autograd.grad((expected * weights).sum(), logits, retain_graph=True)[0]
    new_gradient = torch.autograd.grad((actual * weights).sum(), logits)[0]
    assert_close(new_gradient, old_gradient)
    assert torch.isfinite(new_gradient).all()
    torch.testing.assert_close(x, before, equal_nan=True)


@pytest.mark.parametrize("invalid", [-1.0, 3.0, 0.5, float("nan"), float("inf")])
def test_categorical_indexing_preserves_invalid_label_error(invalid):
    emission = GSPNCategoricalEmission(3, 2)
    with pytest.raises(
        ValueError, match="Observed categorical labels must be valid integer category IDs"
    ):
        emission.component_log_prob(
            torch.tensor([invalid]), params=emission.distribution_parameters()
        )


@pytest.mark.parametrize(
    "case",
    PREDICTOR_FIXTURE["cases"],
    ids=[
        f"{case['model']}-{case['config']['global_pooling']}-{case['data']['x'].dtype}"
        for case in PREDICTOR_FIXTURE["cases"]
    ],
)
def test_predictor_outputs_gradients_and_update_regression(case):
    # Restore capture RNG consumption to verify native initialization as well.
    torch.manual_seed(451)
    inputs = torch.cat([torch.randn(n, 3, dtype=case["data"]["x"].dtype) for n in (3, 1, 2)])
    assert_close(inputs, case["data"]["x"])
    predictor = getattr(readout, case["model"])((3, 0), 2, case["config"]).to(
        case["data"]["x"].dtype
    )
    for name, value in predictor.state_dict().items():
        assert_close(value, case["state"][name])
    predictor.load_state_dict(case["state"])
    data = Batch(**deepcopy(case["data"]))
    before = data.clone()
    outputs = predictor(data)
    assert len(outputs) == 2
    for actual, expected in zip(outputs, case["outputs"]):
        assert_close(actual, expected)
    sum(value.square().mean() for value in outputs).backward()
    assert set(dict(predictor.named_parameters())) == set(case["gradients"])
    for name, parameter in predictor.named_parameters():
        expected = case["gradients"][name]
        assert (parameter.grad is None) == (expected is None)
        if expected is not None:
            assert_close(parameter.grad, expected)
            assert torch.isfinite(parameter.grad).all()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=0.01)
    optimizer.step()
    for name, value in predictor.state_dict().items():
        assert_close(value, case["updated_state"][name])
    for actual, expected in zip(predictor(data), case["updated_outputs"]):
        assert_close(actual, expected)
    torch.testing.assert_close(data.x, before.x)
    torch.testing.assert_close(data.edge_index, before.edge_index)


@pytest.mark.parametrize(
    "model_name",
    [
        "LinearGraphClassifier_GlobalReadout",
        "MLPGraphClassifier_GlobalReadout",
        "MLPGraphClassifier_GraphEmbedding",
    ],
)
@pytest.mark.parametrize("pooling", ["invalid", None, []])
def test_predictor_pooling_preserves_error(model_name, pooling):
    with pytest.raises(NotImplementedError, match="^Global pooling operator not recognized$"):
        getattr(readout, model_name)((3, 0), 2, {"global_pooling": pooling, "hidden_units": 4})


# Configuration cleanup preserves precedence, references, and source dictionaries.
@pytest.mark.parametrize("stage", ["encoder", "predictor"])
def test_stage_configuration_merge(tmp_path, stage):
    from pipeline import EmbeddingPipeline

    config = {
        "device": "cpu",
        "batch_size": 9,
        "seed": 1,
        "checkpoint": False,
        "shared": {"nested": [1, 2]},
        "encoder": {"model": "model.GSPN", "batch_size": 3, "seed": 5, "checkpoint": False},
        "predictor": {
            "model": "readout.MLPGraphClassifier_GlobalReadout",
            "batch_size": 4,
            "optimizer": {"lr": 0.01},
        },
    }
    before = deepcopy(config)
    experiment = EmbeddingPipeline(config, str(tmp_path), 42)
    actual = experiment._stage_config(stage)
    expected = dict(config)
    selected = dict(expected.pop(stage))
    expected.pop("encoder", None)
    expected.pop("predictor", None)
    expected.update(selected)
    expected.update(seed=42, checkpoint=True)
    assert actual == expected
    assert config == before
    assert actual is not config
    assert actual["shared"] is config["shared"]
    if stage == "predictor":
        assert actual["optimizer"] is config[stage]["optimizer"]
    actual["batch_size"] = 123
    assert config == before


@pytest.mark.parametrize("stage", ["encoder", "predictor"])
def test_stage_configuration_missing_stage_error(tmp_path, stage):
    from pipeline import EmbeddingPipeline

    experiment = EmbeddingPipeline({"device": "cpu"}, str(tmp_path), 42)
    with pytest.raises(KeyError) as error:
        experiment._stage_config(stage)
    assert error.value.args == (stage,)


# Lifecycle forwarding retains every explicit callback and distributed argument.
@pytest.mark.parametrize("final", [False, True])
def test_pipeline_lifecycle_argument_forwarding(tmp_path, monkeypatch, final):
    from pipeline import EmbeddingPipeline

    experiment = EmbeddingPipeline({}, str(tmp_path), 42)
    provider, logger = object(), object()
    progress, termination = lambda value: None, lambda: False
    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return "result"

    monkeypatch.setattr(experiment, "_run_pipeline", run)
    entry = experiment._run_test_impl if final else experiment._run_valid_impl
    assert entry(provider, 17, logger, progress, termination, 0, 1) == "result"
    assert captured == {
        "dataset_getter": provider,
        "training_timeout_seconds": 17,
        "logger": logger,
        "progress_callback": progress,
        "should_terminate": termination,
        "final": final,
        "ddp_rank": 0,
        "ddp_world_size": 1,
    }


@pytest.mark.parametrize("deadline,expected_timeout", [(None, -1), (121.2, 22)])
def test_training_stage_forwards_controls(tmp_path, monkeypatch, deadline, expected_timeout):
    import pipeline

    experiment = pipeline.EmbeddingPipeline({}, str(tmp_path), 42)
    config = {"epochs": 3}
    model, train, validation, test, logger = [object() for _ in range(5)]
    captured, events, stop_checks = {}, [], []
    metrics = tuple({"value": i} for i in range(6))

    def progress(value):
        events.append(value)

    def termination():
        stop_checks.append(True)
        return False

    class Engine:
        def train(self, **kwargs):
            captured.update(kwargs)
            kwargs["progress_callback"]({"epoch": 1})
            return metrics

    class Stage:
        model_config = config

        def __init__(self, settings, path, seed):
            assert settings is config
            assert path == str(tmp_path / "encoder")
            assert seed == 42

        def create_model(self, dims, target_dim, settings):
            assert dims == (2, 0) and target_dim == 3 and settings is config
            return model

        def create_engine(self, settings, created_model):
            assert settings is config and created_model is model
            return Engine()

    monkeypatch.setattr(pipeline, "Experiment", Stage)
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: 100.0)
    assert experiment._train(
        config=config,
        name="encoder",
        dims=(2, 0),
        target_dim=3,
        train=train,
        validation=validation,
        test=test,
        logger=logger,
        deadline=deadline,
        progress_callback=progress,
        should_terminate=termination,
    ) == (model, metrics)
    assert captured == {
        "train_loader": train,
        "validation_loader": validation,
        "test_loader": test,
        "max_epochs": 3,
        "logger": logger,
        "training_timeout_seconds": expected_timeout,
        "progress_callback": progress,
        "should_terminate": termination,
    }
    assert events == [{"epoch": 1}]
    assert len(stop_checks) == 2


@pytest.mark.parametrize("reason", ["timeout", "termination"])
def test_pipeline_stops_before_loading_evidence(tmp_path, monkeypatch, reason):
    import yaml
    from mlwiz.evaluation.grid import Grid
    from mlwiz.experiment.experiment import ExperimentTerminated
    from test_migration_pipeline import SpyProvider

    import pipeline
    from dataset import SmokeGraphDataset

    config = Grid(yaml.safe_load((ROOT / "configs/smoke_pipeline.yml").read_text()))[0]
    config["embeddings_folder"] = str(tmp_path / "embeddings")
    experiment = pipeline.EmbeddingPipeline(config, str(tmp_path / "run"), 42)
    provider = SpyProvider(SmokeGraphDataset(str(tmp_path / "data")))
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: 100.0)
    error = TimeoutError if reason == "timeout" else ExperimentTerminated
    with pytest.raises(error):
        experiment._run_pipeline(
            provider,
            0 if reason == "timeout" else -1,
            None,
            should_terminate=lambda: reason == "termination",
        )
    assert provider.calls == []
    assert not list((tmp_path / "embeddings").rglob("embeddings.pkl"))


@pytest.mark.parametrize("shuffle", [False, True])
def test_predictor_loaders_preserve_subsets_order_and_settings(tmp_path, shuffle):
    from torch.utils.data import RandomSampler, SequentialSampler
    from torch_geometric.data import Data

    from pipeline import EmbeddingPipeline

    embeddings = {
        name: [(Data(x=torch.ones(1, 2), sample_id=i), torch.tensor(i)) for i in indices]
        for name, indices in {
            "train": [5, 1, 3, 2],
            "validation": [8, 6],
            "test": [11, 9, 10],
        }.items()
    }
    experiment = EmbeddingPipeline({}, str(tmp_path), 42)
    loaders = experiment._predictor_loaders(embeddings, {"batch_size": 2, "shuffle": shuffle}, 0.5)
    assert {
        name: [int(graph.sample_id) for graph, _ in loader.dataset]
        for name, loader in loaders.items()
    } == {
        "train": [5, 1],
        "validation": [8],
        "test": [11, 9, 10],
    }
    for name, loader in loaders.items():
        assert loader.batch_size == 2 and loader.num_workers == 0
        sampler = RandomSampler if name == "train" and shuffle else SequentialSampler
        assert isinstance(loader.sampler, sampler)
    assert [int(graph.sample_id) for graph, _ in embeddings["train"]] == [5, 1, 3, 2]


@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_predictor_empty_partition_error(tmp_path, partition):
    from torch_geometric.data import Data

    from pipeline import EmbeddingPipeline

    sample = (Data(x=torch.ones(1, 2)), torch.tensor(0))
    embeddings = {name: [sample, sample] for name in ("train", "validation", "test")}
    embeddings[partition] = [] if partition == "test" else [sample]
    experiment = EmbeddingPipeline({}, str(tmp_path), 42)
    with pytest.raises(
        ValueError, match=f"Supervision fraction leaves an empty {partition} partition"
    ):
        experiment._predictor_loaders(embeddings, {"batch_size": 2}, 0.5)


# Preserve the helper's existing behavior for any explicitly selected stage block.
def test_stage_configuration_excludes_selected_custom_block(tmp_path):
    from pipeline import EmbeddingPipeline

    config = {
        "device": "cpu",
        "encoder": {},
        "predictor": {},
        "custom": {"batch_size": 7, "seed": 2},
    }
    experiment = EmbeddingPipeline(config, str(tmp_path), 42)
    assert experiment._stage_config("custom") == {
        "device": "cpu",
        "batch_size": 7,
        "seed": 42,
        "checkpoint": True,
    }
    assert config["custom"] == {"batch_size": 7, "seed": 2}
