# Load MLWiz final-run artifacts using the configuration's actual paths.
import json
from pathlib import Path

import torch
import yaml
from mlwiz.data.splitter import Splitter
from mlwiz.evaluation.grid import Grid
from mlwiz.util import s2c


def assessment_folder(config_file):
    """Resolve the assessment directory from the dataset-specific MLWiz experiment name."""
    config = yaml.safe_load(Path(config_file).read_text())
    grid = Grid(config)
    return Path(config["experiment"]["result_folder"]) / grid.exp_name / "MODEL_ASSESSMENT"


def load_run(config_file, outer_fold=1, final_run=1, stage=None, device="cpu"):
    """Load a selected final-run model, its provider, and configuration for notebook analysis."""
    config = yaml.safe_load(Path(config_file).read_text())
    outer = assessment_folder(config_file) / f"OUTER_FOLD_{outer_fold}"
    winner = json.loads((outer / "MODEL_SELECTION/winner_config.json").read_text())
    selected = winner["config"]
    run = outer / f"final_run{final_run}"
    model_config = {**selected, **selected[stage]} if stage else selected
    if stage:
        run = run / stage
    info = config["dataset"]
    splits = Splitter.load(info["data_splits_file"])
    loader_info = config["data_loading"]["data_loader"]
    provider = s2c(config["data_loading"]["dataset_getter"])(
        storage_folder=info["storage_folder"],
        splits_filepath=info["data_splits_file"],
        dataset_class=s2c(info["dataset_class"]),
        data_loader_class=s2c(loader_info["class_name"]),
        data_loader_args=loader_info.get("args", {}),
        outer_folds=splits.n_outer_folds,
        inner_folds=splits.n_inner_folds,
    )
    provider.set_outer_k(outer_fold - 1)
    provider.set_inner_k(0)
    provider.set_exp_seed(config["reproducibility"]["seed"])
    dataset = provider._get_dataset()
    dims = dataset.dim_input_features
    if stage == "predictor":
        manifest = json.loads((run / "model_manifest.json").read_text())
        dims = manifest["dim_input_features"]
    model = s2c(model_config["model"])(dims, dataset.dim_target, model_config)
    checkpoint = run / "best_checkpoint.pth"
    if not checkpoint.exists():
        checkpoint = run / "last_checkpoint.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    return model, provider, selected
