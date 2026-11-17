from pathlib import Path

import pytest

from tetris_rl.config import resolve_run_directory


@pytest.mark.parametrize("recorded", [
    r"runs\formal_v1\dqn_seed0_attempt1",
    r"C:\other-computer\tetris-rl\runs\formal_v1\dqn_seed0_attempt1",
    "/home/teammate/tetris-rl/runs/formal_v1/dqn_seed0_attempt1",
])
def test_recorded_runs_follow_the_cloned_batch_location(tmp_path, recorded):
    batch = tmp_path / "runs" / "formal_v1"
    assert resolve_run_directory(batch, recorded) == Path(batch) / "dqn_seed0_attempt1"
