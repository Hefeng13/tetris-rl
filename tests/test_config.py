from pathlib import Path
import sys

import pytest
import yaml

from tetris_rl.config import ROOT, load_config, resolve_run_directory


@pytest.mark.parametrize("recorded", [
    r"runs\formal_v1\dqn_seed0_attempt1",
    r"C:\other-computer\tetris-rl\runs\formal_v1\dqn_seed0_attempt1",
    "/home/teammate/tetris-rl/runs/formal_v1/dqn_seed0_attempt1",
])
def test_recorded_runs_follow_the_cloned_batch_location(tmp_path, recorded):
    batch = tmp_path / "runs" / "formal_v1"
    assert resolve_run_directory(batch, recorded) == Path(batch) / "dqn_seed0_attempt1"


def test_batch_configuration_expands_shared_settings_and_method_files():
    directory = ROOT / "configs"
    manifest = yaml.safe_load((directory / "experiments.yaml").read_text(encoding="utf-8"))
    expected = yaml.safe_load((directory / manifest["shared_config"]).read_text(encoding="utf-8"))
    expected["experiments"] = [
        yaml.safe_load((directory / name).read_text(encoding="utf-8"))
        for name in manifest["experiment_files"]
    ]
    assert load_config() == expected


@pytest.mark.parametrize("name", ["dqn", "double_dqn", "dqn_shaped", "dqn_features13"])
def test_single_method_uses_shared_settings_and_saves_standalone(tmp_path, name):
    full = load_config()
    single = load_config(ROOT / "configs" / f"{name}.yaml")
    assert single["experiments"] == [item for item in full["experiments"] if item["id"] == name]
    assert {key: value for key, value in single.items() if key != "experiments"} == {
        key: value for key, value in full.items() if key != "experiments"
    }
    saved = tmp_path / "config.yaml"
    saved.write_text(yaml.safe_dump(single), encoding="utf-8")
    assert load_config(saved) == single


@pytest.mark.parametrize("name", ["dqn", "double_dqn", "dqn_shaped", "dqn_features13"])
def test_train_entry_point_selects_the_single_config_method(monkeypatch, tmp_path, name):
    from tetris_rl import train

    calls = []
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(ROOT / "configs" / f"{name}.yaml"),
                                     "--output", str(tmp_path / "run")])
    monkeypatch.setattr(train, "run_training", lambda *args: calls.append(args))
    train.main()
    assert calls[0][1] == name
    assert calls[0][0]["experiments"][0]["id"] == name


def test_legacy_full_config_loads_without_other_files(tmp_path):
    source = ROOT / "configs/pilot.yaml"
    copied = tmp_path / "legacy.yaml"
    copied.write_bytes(source.read_bytes())
    assert load_config(copied) == yaml.safe_load(source.read_text(encoding="utf-8"))
