# GSPN-GPT-FIXED: One MLWiz experiment owns both stages and keeps outer test blind.
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import torch
from mlwiz.experiment import Experiment
from mlwiz.static import LOSS, SCORE
from mlwiz.util import atomic_dill_save, dill_load
from torch_geometric.loader import DataLoader

from migration import sha256


def code_fingerprint():
    files = sorted(Path(__file__).parent.glob("*.py"))
    return hashlib.sha256("".join(p.name + sha256(p) for p in files).encode()).hexdigest()


def cache_identity(dataset, indices, config, seed, mode):
    # GSPN-GPT-FIXED: Fold, evidence, configuration, implementation, and versions key reuse.
    path = getattr(dataset, "dataset_filepath", None)
    dataset_identity = {
        "class": type(dataset).__module__ + "." + type(dataset).__name__,
        "size": len(dataset),
        "file": str(path),
        "sha256": sha256(path) if path is not None else None,
    }
    identity = {
        "dataset": dataset_identity,
        "indices": indices,
        "encoder": config,
        "seed": seed,
        "mode": mode,
        "code": code_fingerprint(),
        "versions": {
            p: importlib.metadata.version(p) for p in ("mlwiz", "torch", "torch-geometric")
        },
        "transforms": {
            "train": _transform_spec(dataset, "transform_train"),
            "eval": _transform_spec(dataset, "transform_eval"),
        },
    }
    return identity, hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()


def _transform_spec(dataset, name):
    from dataset import transform_identity

    return transform_identity(getattr(dataset, name, None))


def extract_embeddings(model, loader, device, should_terminate=None, deadline=None):
    # GSPN-GPT-FIXED: Explicit deterministic inference replaces PyDGN's removed data-list return.
    result = []
    model.eval()
    with torch.no_grad():
        for graphs, targets in loader:
            check_stop(should_terminate, deadline)
            graphs = graphs.to(device)
            embeddings = model(graphs)[1]
            if embeddings is None or embeddings.shape[0] != graphs.num_nodes:
                raise ValueError("Encoder must return one embedding per node")
            individual = graphs.to_data_list()
            for i, graph in enumerate(individual):
                graph.x = embeddings[graphs.ptr[i] : graphs.ptr[i + 1]].detach().cpu()
                graph = graph.cpu()
                result.append((graph, targets[i].detach().cpu()))
    return result


def check_stop(should_terminate, deadline):
    if should_terminate is not None and should_terminate():
        from mlwiz.experiment.experiment import ExperimentTerminated

        raise ExperimentTerminated("Embedding pipeline terminated")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Embedding pipeline exceeded its total training timeout")


class EmbeddingPipeline(Experiment):
    def _stage_config(self, key):
        config = dict(self.model_config.items())
        stage = dict(config.pop(key))
        config.pop("encoder", None)
        config.pop("predictor", None)
        config.update(stage)
        config["seed"] = self.exp_seed
        config["checkpoint"] = True
        return config

    def _train(
        self,
        config,
        name,
        dims,
        target_dim,
        train,
        validation,
        test,
        logger,
        deadline,
        progress_callback,
        should_terminate,
    ):
        check_stop(should_terminate, deadline)
        stage = Experiment(config, str(Path(self.exp_path) / name), self.exp_seed)
        model = stage.create_model(dims, target_dim, stage.model_config)
        engine = stage.create_engine(stage.model_config, model)
        remaining = -1 if deadline is None else max(1, math.ceil(deadline - time.monotonic()))
        metrics = engine.train(
            train_loader=train,
            validation_loader=validation,
            test_loader=test,
            max_epochs=config["epochs"],
            logger=logger,
            training_timeout_seconds=remaining,
            progress_callback=progress_callback,
            should_terminate=should_terminate,
        )
        check_stop(should_terminate, deadline)
        return model, metrics

    def _partitions(self, provider, final):
        splitter = provider._get_splitter()
        if final:
            fold = splitter.outer_folds[provider.outer_k]
            return {
                "train": list(fold.train_idxs),
                "validation": list(fold.val_idxs),
                "test": list(fold.test_idxs),
            }
        fold = splitter.inner_folds[provider.outer_k][provider.inner_k]
        return {"train": list(fold.train_idxs), "validation": list(fold.val_idxs)}

    def _run_pipeline(
        self,
        dataset_getter,
        training_timeout_seconds,
        logger,
        progress_callback=None,
        should_terminate=None,
        final=False,
        ddp_rank=None,
        ddp_world_size=1,
    ):
        if ddp_world_size != 1:
            raise ValueError(
                "The embedding pipeline supports one device per run; parallelize folds/configurations with MLWiz"
            )
        deadline = (
            None if training_timeout_seconds < 0 else time.monotonic() + training_timeout_seconds
        )
        encoder_config = self._stage_config("encoder")
        partitions = self._partitions(dataset_getter, final)
        dataset = dataset_getter._get_dataset()
        identity, digest = cache_identity(
            dataset, partitions, encoder_config, self.exp_seed, "outer" if final else "inner"
        )
        cache = (
            Path(self.model_config.get("embeddings_folder", "EMBEDDINGS_MLWIZ"))
            / digest
            / "embeddings.pkl"
        )
        check_stop(should_terminate, deadline)
        if cache.exists():
            saved = dill_load(str(cache))
            if saved["identity"] != identity:
                raise ValueError("Embedding cache provenance does not match")
            embeddings = saved["embeddings"]
        else:
            # Entire training fold is unlabeled encoder input, independent of supervision fraction.
            train = dataset_getter._get_loader(
                partitions["train"],
                is_eval=False,
                batch_size=encoder_config["batch_size"],
                shuffle=encoder_config.get("shuffle", True),
            )
            validation = dataset_getter._get_loader(
                partitions["validation"],
                is_eval=True,
                batch_size=encoder_config["batch_size"],
                shuffle=False,
            )
            model, _ = self._train(
                encoder_config,
                "encoder",
                dataset.dim_input_features,
                dataset.dim_target,
                train,
                validation,
                None,
                logger,
                deadline,
                progress_callback,
                should_terminate,
            )
            embeddings = {}
            for name, indices in partitions.items():
                loader = dataset_getter._get_loader(
                    indices, is_eval=True, batch_size=encoder_config["batch_size"], shuffle=False
                )
                embeddings[name] = extract_embeddings(
                    model, loader, encoder_config["device"], should_terminate, deadline
                )
                if len(embeddings[name]) != len(indices):
                    raise ValueError("Embedding count does not match partition indices")
            cache.parent.mkdir(parents=True, exist_ok=True)
            atomic_dill_save(
                {
                    "identity": identity,
                    "embeddings": embeddings,
                    "encoder_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                },
                str(cache),
            )
        fraction = self.model_config.get("weak_supervision_percentage", 1.0)
        if not 0 < fraction <= 1:
            raise ValueError("weak_supervision_percentage must be in (0, 1]")
        config = self._stage_config("predictor")
        loaders = {}
        for name, samples in embeddings.items():
            selected = samples if name == "test" else samples[: math.floor(len(samples) * fraction)]
            if not selected:
                raise ValueError(f"Supervision fraction leaves an empty {name} partition")
            loaders[name] = DataLoader(
                selected,
                batch_size=config["batch_size"],
                shuffle=name == "train" and config.get("shuffle", True),
                num_workers=0,
            )
        dims = (embeddings["train"][0][0].x.shape[1], 0)
        _, metrics = self._train(
            config,
            "predictor",
            dims,
            dataset.dim_target,
            loaders["train"],
            loaders["validation"],
            loaders.get("test"),
            logger,
            deadline,
            progress_callback,
            should_terminate,
        )
        results = [{LOSS: metrics[i], SCORE: metrics[i + 1]} for i in range(0, 6, 2)]
        return tuple(results if final else results[:2])

    def _run_valid_impl(
        self,
        dataset_getter,
        training_timeout_seconds,
        logger,
        progress_callback=None,
        should_terminate=None,
        ddp_rank=None,
        ddp_world_size=1,
    ):
        return self._run_pipeline(
            dataset_getter,
            training_timeout_seconds,
            logger,
            progress_callback,
            should_terminate,
            False,
            ddp_rank,
            ddp_world_size,
        )

    def _run_test_impl(
        self,
        dataset_getter,
        training_timeout_seconds,
        logger,
        progress_callback=None,
        should_terminate=None,
        ddp_rank=None,
        ddp_world_size=1,
    ):
        return self._run_pipeline(
            dataset_getter,
            training_timeout_seconds,
            logger,
            progress_callback,
            should_terminate,
            True,
            ddp_rank,
            ddp_world_size,
        )


class EmbeddingTask(Experiment):
    """Optional encoder-only experiment with honest MLWiz validation and assessment."""

    # GSPN-GPT-FIXED: Remove dummy final scores and premature outer-test extraction.
