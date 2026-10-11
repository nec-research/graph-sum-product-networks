# Exercise framework adapters, readouts, and edge-case metrics.
import json
from pathlib import Path

import pytest
import torch
import yaml
from mlwiz.data.provider import SubsetTrainEval
from mlwiz.data.splitter import Splitter
from mlwiz.data.util import preprocess_data
from torch_geometric.loader import DataLoader

from baselines import DGI, GIN, GAE_Adj
from dataset import SmokeGraphDataset
from metric import (
    OGBGROCAUC,
    BCEWithLogits,
    ConditionalMeanImputationLikelihood,
    MissingFeaturesMSE,
)
from migration import OfficialOGBSplitter, PreservedSplitter

# GSPN heads now share the canonical model module and named result interface.
from model import (
    GSPN,
    ProbabilisticGraphReadout,
    ProbabilisticGraphReadoutNoLayerAttention,
    ProbabilisticGraphReadoutNoLayerAttentionMLP,
    ProbabilisticGraphReadoutNoLayerAttentionMLPVersion2,
)
from transform import DGICorruption, NegativeSampling


@pytest.mark.parametrize(
    "readout",
    [
        ProbabilisticGraphReadout,
        ProbabilisticGraphReadoutNoLayerAttention,
        ProbabilisticGraphReadoutNoLayerAttentionMLP,
        ProbabilisticGraphReadoutNoLayerAttentionMLPVersion2,
    ],
)
def test_probabilistic_readouts(readout):
    config = {
        "num_mixtures": 2,
        "num_layers": 2,
        "global_pooling": "mean",
        "graph_emission_class": "model.GSPNCategoricalEmission",
    }
    model = readout(2, config)
    data = torch.softmax(torch.randn(6, 2, 2), dim=-1)
    result = model(data, torch.tensor([0, 0, 0, 1, 1, 1]), targets=torch.tensor([0, 1]))
    assert torch.isfinite(result.log_prob).all()
    assert result.predictions.shape == (2, 2)
    (-result.log_prob.mean()).backward()


def test_missing_metrics_empty_and_unknown():
    mse = MissingFeaturesMSE()
    empty = torch.tensor([])
    assert mse.compute_metric(empty, empty) == 0
    target = torch.tensor([float("nan"), 2.0])
    assert mse.compute_metric(target, torch.tensor([99.0, 3.0])) == 1
    conditional = ConditionalMeanImputationLikelihood()
    assert conditional.compute_metric(empty, empty) == 0
    assert BCEWithLogits().compute_metric(empty, empty) == 0
    auc = OGBGROCAUC()
    predictions, targets = auc.get_predictions_and_targets(
        torch.tensor([[0.0], [1.0]]), torch.tensor([[0.1], [0.9]])
    )
    assert targets.shape == (2, 1)
    assert auc.compute_metric(targets, predictions) == 1


def test_real_data_cli_transform_contract(tmp_path):
    options = yaml.safe_load(Path("configs/smoke_data.yml").read_text())
    options["dataset"]["args"]["storage_folder"] = str(tmp_path / "data")
    options["dataset"]["args"]["pre_transform"] = {
        "class_name": "transform.Compose",
        "args": {
            "transforms": [
                {
                    "class_name": "transform.GammaRandomNodeFeaturesMask",
                    "args": {"alpha": 1.5, "beta": 2},
                }
            ]
        },
    }
    options["dataset"]["args"]["transform_train"] = {"class_name": "transform.DGICorruption"}
    options["dataset"]["args"]["transform_eval"] = {"class_name": "transform.DGICorruption"}
    options["splitter"]["splits_folder"] = str(tmp_path / "splits")
    options["skip_splits_check"] = False
    preprocess_data(options)
    from mlwiz.data.util import load_dataset

    dataset = load_dataset(str(tmp_path / "data"), SmokeGraphDataset)
    train = SubsetTrainEval(dataset, [0], False)[0]
    evaluation = SubsetTrainEval(dataset, [0], True)[0]
    torch.testing.assert_close(train[0].x_corrupted, evaluation[0].x_corrupted)
    assert not hasattr(dataset[0][0], "x_corrupted")


def test_preserved_splitter_checks_against_prepared_dataset(tmp_path):
    dataset = SmokeGraphDataset(str(tmp_path / "data"))
    args = {
        "n_outer_folds": 1,
        "n_inner_folds": 1,
        "seed": 42,
        "stratify": True,
        "shuffle": True,
        "inner_val_ratio": 0.2,
        "outer_val_ratio": 0.2,
        "test_ratio": 0.2,
    }
    standard = Splitter(**args)
    standard.split(dataset, torch.tensor([i % 2 for i in range(30)]).numpy())
    data = {
        "splitter_args": args,
        "outer_folds": [f.todict() for f in standard.outer_folds],
        "inner_folds": [[f.todict() for f in fs] for fs in standard.inner_folds],
    }
    source = tmp_path / "legacy.splits"
    torch.save(data, source)
    preserved = PreservedSplitter(str(source), **args)
    preserved.split(dataset)
    destination = tmp_path / "new.splits"
    preserved.save(str(destination))
    assert json.loads(destination.with_suffix(".provenance.json").read_text())["dataset_size"] == 30
    assert Splitter.load(str(destination)).outer_folds[0].todict() == data["outer_folds"][0]
    official = {"train": list(range(20)), "valid": list(range(20, 24)), "test": list(range(24, 30))}
    (dataset.dataset_folder / "official_splits.json").write_text(json.dumps(official))
    splitter = OfficialOGBSplitter(**args)
    splitter.split(dataset)
    assert splitter.outer_folds[0].test_idxs == official["test"]
    official_path = tmp_path / "official.splits"
    splitter.save(str(official_path))
    provenance = json.loads(official_path.with_suffix(".provenance.json").read_text())
    assert provenance["dataset_size"] == 30
    assert provenance["kind"] == "official_ogb"


@pytest.mark.parametrize("model_type", [GAE_Adj, DGI, GIN])
def test_baseline_models_are_valid_mlwiz_inputs(tmp_path, model_type):
    dataset = SmokeGraphDataset(str(tmp_path))
    samples = [NegativeSampling()(DGICorruption()(dataset[i])) for i in range(2)]
    graphs, _ = next(iter(DataLoader(samples, batch_size=2)))
    config = {
        "num_layers": 2,
        "dim_embedding": 4,
        "train_eps": False,
        "concat_out_across_layers": True,
        "dropout": 0.1,
        "global_aggregation": "mean",
    }
    model = model_type((2, 0), 2, config)
    output = model(graphs)
    assert output[1].shape[0] in (2, 8)


def test_unavailable_categorical_truth_is_excluded():
    from torch_geometric.data import Batch, Data

    config = {
        "num_layers": 2,
        "num_mixtures": 2,
        "emission_class": "model.GSPNMultiCategoricalEmission",
        "dim_categorical_features": [2, 3],
    }
    data = Batch.from_data_list(
        [
            Data(
                x=torch.tensor([[1.0, -1.0], [float("nan"), 2.0]]),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                mask=torch.tensor([[True, False], [False, True]]),
            )
        ]
    )
    model = GSPN((2, 0), 0, config)
    _, _, extras = model(data)
    assert torch.isnan(extras[9]).all()
    assert torch.isfinite(extras[0]).all()


def test_synthetic_node_labels_do_not_break_graph_batching(tmp_path):
    # Preserve node-level community labels while batching graph targets.
    from torch_geometric.data import Data

    from dataset import SyntheticDataset

    raw = tmp_path / "raw"
    raw.mkdir()
    graphs = [
        Data(x=torch.randn(n, 2), edge_index=torch.empty(2, 0, dtype=torch.long), y=torch.arange(n))
        for n in [2, 3]
    ]
    torch.save(graphs, raw / "data_list_100.pt")
    # The original reader consumes only data_list_100.pt, even if other chunks exist.
    torch.save([graphs[0]], raw / "data_list_200.pt")
    dataset = SyntheticDataset(str(tmp_path / "data"), raw_dir=str(raw))
    assert len(dataset) == 2
    assert dataset.dim_target == 0
    batch, targets = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch.num_nodes == 5
    assert targets.shape == (2, 1)
    torch.testing.assert_close(dataset[1][0].y, graphs[1].y)
