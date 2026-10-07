# Convert saved partitions to MLWiz without changing any index/order.
import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from mlwiz.data.splitter import InnerFold, OuterFold, Splitter


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_splits(data, dataset_size=None):
    """Check bounds, duplicates, disjoint partitions, and containment in the outer non-test pool."""
    outer = data["outer_folds"]
    inner = data["inner_folds"]
    args = data["splitter_args"]
    if len(outer) != args["n_outer_folds"] or len(inner) != len(outer):
        raise ValueError("Inconsistent outer-fold count")
    for outer_fold, inner_folds in zip(outer, inner):
        if len(inner_folds) != args["n_inner_folds"]:
            raise ValueError("Inconsistent inner-fold count")
        for fold in [outer_fold, *inner_folds]:
            sets = []
            for indices in fold.values():
                if indices is None:
                    continue
                if any(
                    not isinstance(i, int)
                    or i < 0
                    or (dataset_size is not None and i >= dataset_size)
                    for i in indices
                ):
                    raise ValueError("Split index is out of bounds")
                if len(set(indices)) != len(indices):
                    raise ValueError("Duplicate indices in a partition")
                sets.append(set(indices))
            if any(a & b for i, a in enumerate(sets) for b in sets[i + 1 :]):
                raise ValueError("Partitions overlap")
        # Inner selection uses the non-test pool before final holdout.
        training = set(outer_fold["train"]).union(outer_fold["val"])
        for fold in inner_folds:
            if not set(fold["train"]).union(fold["val"]).issubset(training):
                raise ValueError("Inner fold is not contained in the outer non-test pool")
    return data


def convert_split(source, destination, dataset_size=None):
    """Convert indices without reordering; reject conflicting existing artifacts or provenance."""
    source, destination = Path(source), Path(destination)
    data = validate_splits(torch.load(source, weights_only=False), dataset_size)
    splitter = Splitter(**data["splitter_args"])
    splitter.outer_folds = [OuterFold(f["train"], f["val"], f["test"]) for f in data["outer_folds"]]
    splitter.inner_folds = [
        [InnerFold(f["train"], f["val"]) for f in folds] for folds in data["inner_folds"]
    ]
    splitter.check_splits_overlap()
    provenance = destination.with_suffix(".provenance.json")
    identity = {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "dataset_size": dataset_size,
        "mlwiz": "1.7.6",
    }
    if destination.exists():
        loaded = Splitter.load(str(destination))
        current = {
            "outer_folds": [f.todict() for f in loaded.outer_folds],
            "inner_folds": [[f.todict() for f in folds] for folds in loaded.inner_folds],
        }
        if current != {k: data[k] for k in current}:
            raise ValueError(f"Existing converted split differs: {destination}")
        if (
            not provenance.exists()
            or json.loads(provenance.read_text())["source_sha256"] != identity["source_sha256"]
        ):
            raise ValueError(f"Converted split provenance differs: {destination}")
        if dataset_size is not None:
            provenance.write_text(json.dumps(identity, indent=2) + "\n")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    splitter.save(str(destination))
    provenance.write_text(json.dumps(identity, indent=2) + "\n")
    return destination


class PreservedSplitter(Splitter):
    """mlwiz-data uses the legacy partitions, with bounds checked against the dataset."""

    def __init__(self, legacy_splits_file, **kwargs):
        super().__init__(**kwargs)
        self.legacy_splits_file = legacy_splits_file

    def split(self, dataset, targets=None):
        # Dataset processing must never silently regenerate research folds.
        data = validate_splits(
            torch.load(self.legacy_splits_file, weights_only=False), len(dataset)
        )
        if any(data["splitter_args"][k] != self._splitter_args()[k] for k in data["splitter_args"]):
            raise ValueError("Configured split arguments differ from the saved research splits")
        self.outer_folds = [OuterFold(f["train"], f["val"], f["test"]) for f in data["outer_folds"]]
        self.inner_folds = [
            [InnerFold(f["train"], f["val"]) for f in folds] for folds in data["inner_folds"]
        ]
        self.dataset_size = len(dataset)

    def save(self, path):
        # Save a standard Splitter so no legacy adapter is needed while training.
        return convert_split(self.legacy_splits_file, path, self.dataset_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", help="Migrated data YAML containing the legacy split path")
    parser.add_argument("--source")
    parser.add_argument("--destination")
    parser.add_argument("--dataset-size", type=int)
    args = parser.parse_args()
    if args.config_file:
        from mlwiz.data.util import load_dataset
        from mlwiz.util import s2c

        config = yaml.safe_load(Path(args.config_file).read_text())
        info = config["dataset"]
        cls = s2c(info["class_name"])
        root = info["args"]["storage_folder"]
        dataset = load_dataset(root, cls)
        split = config["splitter"]
        source = split["args"].get("legacy_splits_file")
        destination = (
            Path(split["splits_folder"])
            / cls.__name__
            / f"{cls.__name__}_outer{split['args']['n_outer_folds']}_inner{split['args']['n_inner_folds']}.splits"
        )
        if source is not None:
            convert_split(source, destination, len(dataset))
        else:
            # Revalidate already prepared official OGB partitions.
            loaded = Splitter.load(str(destination))
            official = OfficialOGBSplitter(**split["args"])
            official.split(dataset)
            if [f.todict() for f in loaded.outer_folds] != [
                f.todict() for f in official.outer_folds
            ]:
                raise ValueError("Saved partitions differ from the official OGB split")
            official.check_splits_overlap()

    elif args.source and args.destination:
        convert_split(args.source, args.destination, args.dataset_size)
    else:
        parser.error("Supply --config-file or --source and --destination")


# When no legacy artifact exists, retain official OGB partitions exactly.
class OfficialOGBSplitter(Splitter):
    def split(self, dataset, targets=None):
        self.source_path = dataset.dataset_folder / "official_splits.json"
        self.dataset_size = len(dataset)
        split = json.loads(self.source_path.read_text())
        self.outer_folds = [OuterFold(split["train"], split["valid"], split["test"])]
        self.inner_folds = [[InnerFold(split["train"], split["valid"])]]
        data = {
            "outer_folds": [f.todict() for f in self.outer_folds],
            "inner_folds": [[f.todict() for f in fs] for fs in self.inner_folds],
            "splitter_args": self._splitter_args(),
        }
        validate_splits(data, len(dataset))

    def save(self, path):
        # Retain official-partition provenance alongside the MLWiz artifact.
        super().save(path)
        identity = {
            "source": str(self.source_path.resolve()),
            "source_sha256": sha256(self.source_path),
            "dataset_size": self.dataset_size,
            "mlwiz": "1.7.6",
            "kind": "official_ogb",
        }
        Path(path).with_suffix(".provenance.json").write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
    main()
