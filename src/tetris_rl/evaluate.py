"""Shared validation and held-out evaluation for all policies."""

import argparse
import csv
from datetime import datetime
import time
from pathlib import Path

import numpy as np
import torch

from .agents.dqn_agent import DQNAgent
from .agents.heuristic_agent import HeuristicAgent
from .agents.lookahead import LookaheadAgent
from .agents.potential_agent import PotentialAgent
from .agents.random_agent import RandomAgent
from .config import ROOT, load_config, make_env, write_json


CLASSIC_LINE_SCORES = (0, 40, 100, 300, 1200)
EPISODE_FIELDS = ["run_id", "experiment", "training_seed", "test_seed", "policy_seed",
                  "lines", "pieces", "return", "classic_score",
                  "clears_1", "clears_2", "clears_3", "clears_4",
                  "terminated", "truncated", "piece_cap"]


def evaluate_policy(agent, config, board_dim, seeds, piece_cap, run_id="", experiment="",
                    training_seed="", random_policy_seed=None, lookahead=0, experiment_config=None):
    if lookahead not in (0, 1):
        raise ValueError("lookahead must be 0 or 1")
    if lookahead and random_policy_seed is not None:
        raise ValueError("lookahead is not supported for random policy")
    base_experiment = experiment[:-10] if experiment.endswith("_lookahead") else experiment
    exp_cfg = experiment_config or next(
        (item for item in config["experiments"] if item["id"] == base_experiment), None)
    shaping_enabled = (base_experiment == "potential") or bool(exp_cfg and exp_cfg["shaping_enabled"])
    env = make_env(config, board_dim, piece_cap, shaping_enabled, experiment=exp_cfg)
    decision_agent = LookaheadAgent(agent, env) if lookahead == 1 else agent
    rows = []
    started = time.perf_counter()
    for seed in seeds:
        policy_seed = ""
        if random_policy_seed is not None:
            policy_seed = random_policy_seed + int(seed)
            decision_agent = RandomAgent(policy_seed)
        candidates = env.reset(seed)
        lines, raw_return, classic_total = 0, 0.0, 0
        clears = [0, 0, 0, 0, 0]
        while True:
            action = decision_agent.act(candidates)
            candidates, reward, terminated, truncated, info = env.step(action)
            cleared = int(info["lines_cleared"])
            lines += cleared
            clears[cleared] += 1
            classic_total += CLASSIC_LINE_SCORES[cleared]
            raw_return += reward
            if terminated or truncated:
                break
        rows.append(dict(zip(EPISODE_FIELDS, [run_id, experiment, training_seed, int(seed),
                    policy_seed, lines, env.pieces, raw_return, classic_total,
                    clears[1], clears[2], clears[3], clears[4],
                    terminated, truncated, piece_cap])))
    env.close()
    summary = {"mean_lines": float(np.mean([row["lines"] for row in rows])),
               "episode_std_lines": float(np.std([row["lines"] for row in rows], ddof=1))
               if len(rows) > 1 else 0.0,
               "mean_pieces": float(np.mean([row["pieces"] for row in rows])),
               "mean_return": float(np.mean([row["return"] for row in rows])),
               "mean_classic_score": float(np.mean([row["classic_score"] for row in rows])),
               "episode_std_classic_score": float(np.std([row["classic_score"] for row in rows], ddof=1))
               if len(rows) > 1 else 0.0,
               "mean_clears_1": float(np.mean([row["clears_1"] for row in rows])),
               "mean_clears_2": float(np.mean([row["clears_2"] for row in rows])),
               "mean_clears_3": float(np.mean([row["clears_3"] for row in rows])),
               "mean_clears_4": float(np.mean([row["clears_4"] for row in rows])),
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
    parser.add_argument("--policy", choices=["random", "heuristic", "potential", "dqn"], default="heuristic")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--piece-cap", type=int)
    parser.add_argument("--lookahead", type=int, choices=[0, 1], default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    args.output = args.output or ROOT / "results" / f"{args.policy}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    torch.set_num_threads(1)
    config = load_config(args.config)
    settings = config["evaluation"]
    count = args.episodes or settings["test_episodes"]
    cap = args.piece_cap or config["environment"]["test_piece_cap"]
    board_dim, training_seed, random_seed, exp_cfg = 3, "", None, None
    experiment = args.policy
    if args.policy == "dqn":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for dqn")
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        agent, metadata = DQNAgent.load(args.checkpoint, device=device)
        board_dim = metadata["board_feature_dim"]
        training_seed = metadata["seed"]
        experiment = metadata["experiment"]
        exp_cfg = metadata.get("experiment_config")
    elif args.policy == "random":
        random_seed = settings["random_baseline_policy_seed"]
        agent = RandomAgent(random_seed)
    elif args.policy == "potential":
        agent = PotentialAgent()
    else:
        agent = HeuristicAgent(**config["heuristic"])
    if args.lookahead == 1:
        experiment = f"{experiment}_lookahead"
    args.output.mkdir(parents=True, exist_ok=True)
    rows, summary = evaluate_policy(agent, config, board_dim,
        range(settings["test_seed_start"], settings["test_seed_start"] + count),
        cap, args.output.name, experiment, training_seed, random_seed,
        lookahead=args.lookahead, experiment_config=exp_cfg)
    write_episodes(args.output / "episodes.csv", rows)
    write_json(args.output / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
