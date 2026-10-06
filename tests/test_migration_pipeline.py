# GSPN-GPT-FIXED: Verify partition provenance, transform isolation, and test-blind orchestration.
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from mlwiz.data.provider import SubsetTrainEval
from mlwiz.data.splitter import Splitter
from mlwiz.evaluation.grid import Grid
from mlwiz.util import s2c
from torch_geometric.loader import DataLoader

from dataset import SmokeGraphDataset
from migration import convert_split, sha256, validate_splits
from pipeline import EmbeddingPipeline, cache_identity
from transform import Compose, DGICorruption, GammaRandomNodeFeaturesMask

ROOT = Path(__file__).resolve().parents[1]


def test_all_saved_partitions_convert_exactly(tmp_path):
    for source in (ROOT / "DATA_SPLITS").rglob("*.splits"):
        before = sha256(source)
        original = torch.load(source, weights_only=False)
        target = tmp_path / source.relative_to(ROOT / "DATA_SPLITS")
        convert_split(source, target)
        size = target.stat().st_size
        convert_split(source, target)
        assert target.stat().st_size == size
        actual = Splitter.load(str(target))
        assert [f.todict() for f in actual.outer_folds] == original["outer_folds"]
        assert [[f.todict() for f in fs] for fs in actual.inner_folds] == original["inner_folds"]
        assert sha256(source) == before


def test_invalid_splits_are_rejected():
    source = next((ROOT / "DATA_SPLITS").rglob("*.splits"))
    data = torch.load(source, weights_only=False)
    invalid = copy.deepcopy(data)
    invalid["outer_folds"][0]["train"].append(invalid["outer_folds"][0]["train"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_splits(invalid)
    invalid = copy.deepcopy(data)
    invalid["inner_folds"][0][0]["train"][0] = invalid["outer_folds"][0]["test"][0]
    with pytest.raises(ValueError):
        validate_splits(invalid)
    with pytest.raises(ValueError, match="bounds"):
        validate_splits(data, 1)


def test_transforms_clone_and_are_deterministic(tmp_path):
    data = SmokeGraphDataset(str(tmp_path))
    original = data[0]
    x = original[0].x.clone()
    transform = Compose(
        [
            {"class_name": "transform.ContinuousAttributesTUDatasetChemical"},
            {
                "class_name": "transform.GammaRandomNodeFeaturesMask",
                "args": {"alpha": 1.5, "beta": 2},
            },
        ]
    )
    transformed = transform(original)
    assert transformed[0] is not original[0]
    torch.testing.assert_close(original[0].x, x)
    a = DGICorruption()(original)
    b = DGICorruption()(original)
    torch.testing.assert_close(a[0].x_corrupted, b[0].x_corrupted)
    assert not hasattr(original[0], "x_corrupted")
    data[0][0].x.zero_()
    torch.testing.assert_close(data[0][0].x, x)
    other = SmokeGraphDataset(str(tmp_path), pre_transform=GammaRandomNodeFeaturesMask(1.5, 2))
    assert data.dataset_filepath != other.dataset_filepath
    again = SmokeGraphDataset(str(tmp_path), pre_transform=GammaRandomNodeFeaturesMask(1.5, 2))
    torch.testing.assert_close(again[0][0].mask, other[0][0].mask)


def test_all_active_configs_and_dotted_paths():
    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in (
                    "model",
                    "class_name",
                    "convolution_class",
                    "emission_class",
                    "loss",
                    "scorer",
                    "readout",
                    "experiment",
                    "dataset_class",
                    "dataset_getter",
                ) and isinstance(item, str):
                    assert s2c(item) is not None, item
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    for directory in ["MODEL_CONFIGS", "WEAK_SUP_MODEL_CONFIGS", "LAYERING_CONFIGS"]:
        for path in (ROOT / directory).glob("*.yml"):
            config = yaml.safe_load(path.read_text())
            grid = Grid(config)
            assert len(grid) > 0
            for concrete in grid:
                check(concrete)
    for directory in ["DATA_CONFIGS", "DGI_DATA_CONFIGS"]:
        for path in (ROOT / directory).glob("*.yml"):
            check(yaml.safe_load(path.read_text()))


class SpyProvider:
    outer_k = 0
    inner_k = 0

    def __init__(self, dataset):
        self.dataset = dataset
        self.calls = []
        self.splitter = SimpleNamespace(
            outer_folds=[
                SimpleNamespace(
                    train_idxs=list(range(16)),
                    val_idxs=list(range(16, 20)),
                    test_idxs=list(range(20, 30)),
                )
            ],
            inner_folds=[
                [SimpleNamespace(train_idxs=list(range(12)), val_idxs=list(range(12, 20)))]
            ],
        )

    def _get_dataset(self):
        return self.dataset

    def _get_splitter(self):
        return self.splitter

    def _get_loader(self, indices, is_eval, **kwargs):
        self.calls.append((list(indices), is_eval))
        return DataLoader(SubsetTrainEval(self.dataset, indices, is_eval), num_workers=0, **kwargs)

    def get_outer_test(self, **kwargs):
        raise AssertionError("Test loader requested during selection")


def test_pipeline_reuse_order_supervision_and_fresh_outer_encoder(tmp_path):
    data = SmokeGraphDataset(str(tmp_path / "data"))
    provider = SpyProvider(data)
    config = Grid(yaml.safe_load((ROOT / "configs/smoke_pipeline.yml").read_text()))[0]
    config["embeddings_folder"] = str(tmp_path / "embeddings")
    config["encoder"]["epochs"] = 1
    config["predictor"]["epochs"] = 1
    config["weak_supervision_percentage"] = 0.5
    experiment = EmbeddingPipeline(config, str(tmp_path / "run"), 42)
    original = experiment._train
    stages = []

    def train(config, name, dims, target, loader, val, test, *args):
        stages.append(
            (
                name,
                len(loader.dataset),
                len(val.dataset),
                None if test is None else len(test.dataset),
            )
        )
        return original(config, name, dims, target, loader, val, test, *args)

    experiment._train = train
    result = experiment.run_valid(provider, -1, None)
    assert len(result) == 2
    assert stages == [("encoder", 12, 8, None), ("predictor", 6, 4, None)]
    assert all(not set(indices) & set(range(20, 30)) for indices, _ in provider.calls)
    from mlwiz.util import dill_load

    cache = next((tmp_path / "embeddings").rglob("embeddings.pkl"))
    saved = dill_load(str(cache))
    assert [int(g.sample_id) for g, _ in saved["embeddings"]["train"]] == list(range(12))
    stages.clear()
    provider.calls.clear()
    experiment.run_valid(provider, -1, None)
    assert stages == [("predictor", 6, 4, None)]
    assert provider.calls == []
    stages.clear()
    experiment.run_test(provider, -1, None)
    assert stages == [("encoder", 16, 4, None), ("predictor", 8, 2, 10)]
    assert len(list((tmp_path / "embeddings").rglob("embeddings.pkl"))) == 2
    _identity, key = cache_identity(data, {"train": [1, 2]}, config["encoder"], 42, "inner")
    _, changed = cache_identity(data, {"train": [2, 1]}, config["encoder"], 42, "inner")
    assert key != changed
    _, changed = cache_identity(
        data, {"train": [1, 2]}, {**config["encoder"], "epochs": 7}, 42, "inner"
    )
    assert key != changed
