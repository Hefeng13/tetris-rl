"""Configuration loading and run provenance."""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    with Path(path or ROOT / "configs/experiments.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def provenance():
    digest = hashlib.sha256()
    for source in sorted((ROOT / "src").rglob("*.py")):
        digest.update(str(source.relative_to(ROOT)).encode())
        digest.update(source.read_bytes())
    packages = ["torch", "numpy", "gymnasium", "tetris-gymnasium", "matplotlib", "PyYAML"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in packages},
            "source_sha256": digest.hexdigest(), "device": "cpu", "torch_threads": 1}
