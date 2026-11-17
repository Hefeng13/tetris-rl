"""Single-configuration training, validation and final held-out evaluation."""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from .agents.dqn_agent import DQNAgent
from .config import ROOT, experiment_config, load_config, make_env, provenance, write_json
from .evaluate import evaluate_policy, write_episodes
from .replay import ReplayBuffer


def run_training(config, experiment_name, seed, output, steps=None,
                 validation_episodes=None, test_episodes=None, evaluation_cap=None, device=None):
    torch.set_num_threads(1)
    device = device or config["training"].get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment. Use CUDA-enabled PyTorch.")
    experiment = experiment_config(config, experiment_name)
    settings = config["training"]
    evaluation = config["evaluation"]
    steps = steps or settings["total_steps_per_seed"]
    validation_episodes = validation_episodes or evaluation["validation_episodes"]
    test_episodes = test_episodes or evaluation["test_episodes"]
    board_dim = experiment["board_feature_dim"]
    env = make_env(config, board_dim, shaping_enabled=experiment["shaping_enabled"])
    candidates = env.reset(1_000_000 + seed * 1_000_000)
    agent = DQNAgent(board_dim + 2, seed, settings["learning_rate"], settings["gamma"],
                     double=experiment["algorithm"] == "double_dqn", device=device)
    replay = ReplayBuffer(settings["replay_capacity"], board_dim + 2, seed=seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    metadata = {**provenance(device), "experiment": experiment_name, "seed": seed,
                "board_feature_dim": board_dim, "steps": steps,
                "value_definition": "Q=known_scaled_reward+gamma*V_afterstate;terminal_V=0",
                "created_at": datetime.now().isoformat(),
                "test_episodes": test_episodes, "validation_episodes": validation_episodes,
                "evaluation_cap_override": evaluation_cap}
    write_json(output / "metadata.json", metadata)
    validation_cap = evaluation_cap or config["environment"]["validation_piece_cap"]
    test_cap = evaluation_cap or config["environment"]["test_piece_cap"]
    started = time.perf_counter()
    best_lines = -1.0
    episode = 0
    last_loss = 0.0
    episode_lines, episode_return = 0, 0.0
    validations = []
    train_fields = ["env_steps", "episode", "episode_lines", "episode_pieces", "return",
                    "epsilon", "td_loss", "elapsed_seconds"]
    validation_fields = ["env_steps", "mean_lines", "mean_pieces", "cap_hit_rate"]
    with (output / "train.csv").open("w", newline="", encoding="utf-8") as train_handle, \
            (output / "validation.csv").open("w", newline="", encoding="utf-8") as val_handle:
        train_writer = csv.DictWriter(train_handle, fieldnames=train_fields)
        val_writer = csv.DictWriter(val_handle, fieldnames=validation_fields)
        train_writer.writeheader()
        val_writer.writeheader()

        def validate(step):
            nonlocal best_lines
            _, summary = evaluate_policy(agent, config, board_dim,
                range(evaluation["validation_seed_start"],
                      evaluation["validation_seed_start"] + validation_episodes), validation_cap,
                experiment=experiment_name)
            row = {"env_steps": step, **{name: summary[name] for name in validation_fields[1:]}}
            val_writer.writerow(row)
            val_handle.flush()
            validations.append(row)
            checkpoint_metadata = {**metadata, "checkpoint_step": step}
            # Equal primary performance favors the later checkpoint.
            if summary["mean_lines"] >= best_lines:
                best_lines = summary["mean_lines"]
                agent.save(output / "best.pt", checkpoint_metadata)
            write_json(output / "progress.json", {"status": "training", "step": step,
                "total_steps": steps, "best_validation_lines": best_lines,
                "last_validation": row, "elapsed_seconds": time.perf_counter() - started})
            print(f"{experiment_name} seed={seed} step={step}/{steps} "
                  f"validation_lines={summary['mean_lines']:.2f}", flush=True)

        validate(0)
        for step in range(1, steps + 1):
            epsilon = max(settings["epsilon_final"], 1.0 - (1.0 - settings["epsilon_final"])
                          * step / (settings["exploration_fraction"] * steps))
            action = agent.act(candidates, epsilon)
            chosen = candidates.features[action].copy()
            learning_reward = float(candidates.learning_rewards[action])
            next_candidates, reward, terminated, truncated, info = env.step(action)
            if reward != candidates.rewards[action] or terminated != candidates.terminated[action]:
                raise AssertionError("Candidate simulation diverged from the game engine")
            replay.add(chosen, learning_reward, next_candidates, terminated)
            if step >= settings["learning_starts"] and step % settings["train_frequency"] == 0:
                last_loss = agent.update(replay, settings["batch_size"])
            if step % settings["target_update_interval"] == 0:
                agent.sync_target()
            episode_lines += int(info["lines_cleared"])
            episode_return += reward
            if terminated or truncated:
                train_writer.writerow(dict(zip(train_fields, [step, episode, episode_lines,
                    env.pieces, episode_return, epsilon, last_loss, time.perf_counter() - started])))
                train_handle.flush()
                episode += 1
                episode_lines, episode_return = 0, 0.0
                candidates = env.reset(1_000_000 + seed * 1_000_000 + episode)
            else:
                candidates = next_candidates
            if step % settings["validation_interval_steps"] == 0 or step == steps:
                validate(step)
        agent.save(output / "last.pt", {**metadata, "checkpoint_step": steps})
    training_seconds = time.perf_counter() - started
    env.close()
    best_agent, best_metadata = DQNAgent.load(output / "best.pt", device=device)
    rows, summary = evaluate_policy(best_agent, config, board_dim,
        range(evaluation["test_seed_start"], evaluation["test_seed_start"] + test_episodes),
        test_cap, output.name, experiment_name, seed)
    write_episodes(output / "episodes.csv", rows)
    summary.update({"experiment": experiment_name, "training_seed": seed,
                    "training_seconds": training_seconds, "steps": steps,
                    "best_checkpoint_step": best_metadata["checkpoint_step"],
                    "initial_validation_lines": validations[0]["mean_lines"],
                    "best_validation_lines": best_lines})
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", {"status": "complete", **summary})
    print(f"COMPLETE {output.name} test_lines={summary['mean_lines']:.2f}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--experiment", default="dqn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--validation-episodes", type=int)
    parser.add_argument("--test-episodes", type=int)
    parser.add_argument("--evaluation-cap", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()
    output = args.output or ROOT / "runs" / f"{args.experiment}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    run_training(load_config(args.config), args.experiment, args.seed, output, args.steps,
                 args.validation_episodes, args.test_episodes, args.evaluation_cap, args.device)


if __name__ == "__main__":
    main()
