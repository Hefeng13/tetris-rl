"""Shared validation and held-out evaluation for all policies."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from .agents.dqn_agent import DQNAgent
from .agents.heuristic_agent import HeuristicAgent
from .agents.random_agent import RandomAgent
from .config import ROOT, load_config, make_env, write_json


EPISODE_FIELDS = ["run_id", "experiment", "training_seed", "test_seed", "policy_seed",
                  "lines", "pieces", "return", "terminated", "truncated", "piece_cap"]


def evaluate_policy(agent, config, board_dim, seeds, piece_cap, run_id="", experiment="",
                    training_seed="", random_policy_seed=None):
    shaping_enabled = any(item["id"] == experiment and item["shaping_enabled"]
                          for item in config["experiments"])
    env = make_env(config, board_dim, piece_cap, shaping_enabled)
    rows = []
    started = time.perf_counter()
    for seed in seeds:
        policy_seed = ""
        if random_policy_seed is not None:
            policy_seed = random_policy_seed + int(seed)
            agent = RandomAgent(policy_seed)
        candidates = env.reset(seed)
        lines, raw_return = 0, 0.0
        while True:
            action = agent.act(candidates)
            candidates, reward, terminated, truncated, info = env.step(action)
            lines += int(info["lines_cleared"])
            raw_return += reward
            if terminated or truncated:
                break
        rows.append(dict(zip(EPISODE_FIELDS, [run_id, experiment, training_seed, int(seed),
                    policy_seed, lines, env.pieces, raw_return, terminated, truncated, piece_cap])))
    env.close()
    summary = {"mean_lines": float(np.mean([row["lines"] for row in rows])),
               "episode_std_lines": float(np.std([row["lines"] for row in rows], ddof=1))
               if len(rows) > 1 else 0.0,
               "mean_pieces": float(np.mean([row["pieces"] for row in rows])),
               "cap_hit_rate": float(np.mean([row["truncated"] for row in rows])),
               "elapsed_seconds": time.perf_counter() - started, "episodes": len(rows)}
    return rows, summary


def write_episodes(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--policy", choices=["random", "heuristic", "dqn"], default="heuristic")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--piece-cap", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "results/baselines")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.config)
    settings = config["evaluation"]
    count = args.episodes or settings["test_episodes"]
    cap = args.piece_cap or config["environment"]["test_piece_cap"]
    board_dim, training_seed, random_seed = 3, "", None
    experiment = args.policy
    if args.policy == "dqn":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for dqn")
        agent, metadata = DQNAgent.load(args.checkpoint)
        board_dim = metadata["board_feature_dim"]
        training_seed = metadata["seed"]
        experiment = metadata["experiment"]
    elif args.policy == "random":
        random_seed = settings["random_baseline_policy_seed"]
        agent = RandomAgent(random_seed)
    else:
        agent = HeuristicAgent(**config["heuristic"])
    args.output.mkdir(parents=True, exist_ok=True)
    rows, summary = evaluate_policy(agent, config, board_dim,
        range(settings["test_seed_start"], settings["test_seed_start"] + count),
        cap, args.output.name, experiment, training_seed, random_seed)
    write_episodes(args.output / "episodes.csv", rows)
    write_json(args.output / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
