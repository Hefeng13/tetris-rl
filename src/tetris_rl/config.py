"""Configuration loading and run provenance."""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(path=None):
    """Expand shared settings and experiment files into a self-contained config.

    A single method file uses the adjacent common.yaml. Saved run configs and
    legacy full configs already contain their settings and load directly.
    """
    path = Path(path or ROOT / "configs/experiments.yaml")
    config = _read_yaml(path)
    if "id" in config:
        return {**_read_yaml(path.parent / "common.yaml"), "experiments": [config]}
    if "shared_config" in config:
        shared = _read_yaml(path.parent / config.pop("shared_config"))
        config = {**shared, **config}
    if "experiment_files" in config:
        config["experiments"] = [
            _read_yaml(path.parent / filename)
            for filename in config.pop("experiment_files")
        ]
    return config


def experiment_config(config, name):
    for experiment in config["experiments"]:
        if experiment["id"] == name:
            return experiment
    raise ValueError(f"Unknown experiment {name}")


def make_env(config, board_dim=3, piece_cap=None, shaping_enabled=False):
    from .env import PlacementEnv
    settings = config["environment"]
    reward = config["reward"]
    return PlacementEnv(board_dim=board_dim,
                        piece_cap=piece_cap or settings["training_piece_cap"],
                        width=settings["width"], height=settings["height"],
                        alife=reward["alife"], clear_line=reward["clear_line"],
                        game_over=reward["game_over"], reward_scale=config["training"]["reward_scale"],
                        gamma=config["training"]["gamma"],
                        shaping=config["shaping"] if shaping_enabled else None)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    os.replace(temporary, path)


def resolve_run_directory(batch, recorded_path):
    """Locate a batch's run after cloning or moving Windows/POSIX records."""
    name = Path(str(recorded_path).replace("\\", "/")).name
    return Path(batch) / name


def provenance(device="cpu"):
    import torch
    digest = hashlib.sha256()
    for source in sorted((ROOT / "src").rglob("*.py")):
        digest.update(source.relative_to(ROOT).as_posix().encode())
        digest.update(source.read_bytes())
    packages = ["torch", "numpy", "gymnasium", "tetris-gymnasium", "matplotlib", "PyYAML"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in packages},
            "source_sha256": digest.hexdigest(), "device": str(device), "torch_threads": 1,
            "gpu": torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None,
            "cuda_runtime": torch.version.cuda}
