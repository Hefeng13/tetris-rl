"""Parallel v2 evaluation with classic Tetris scoring, clear breakdowns, and lookahead."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from tetris_rl.agents.dqn_agent import DQNAgent
from tetris_rl.agents.heuristic_agent import HeuristicAgent
from tetris_rl.agents.potential_agent import PotentialAgent
from tetris_rl.config import ROOT, load_config, resolve_run_directory, write_json
from tetris_rl.evaluate import EPISODE_FIELDS, evaluate_policy, write_episodes


def _run_chunk(task):
    torch.set_num_threads(1)
    config = load_config(task["config_path"])
    policy = task["policy"]
    board_dim = task["board_dim"]
    random_seed = None
    exp_cfg = None
    if policy == "dqn":
        agent, metadata = DQNAgent.load(task["checkpoint"], device="cpu")
        board_dim = metadata["board_feature_dim"]
        exp_cfg = metadata.get("experiment_config")
    elif policy == "random":
        agent = None
        random_seed = config["evaluation"]["random_baseline_policy_seed"]
    elif policy == "potential":
        agent = PotentialAgent()
    elif policy == "heuristic":
        agent = HeuristicAgent(**config["heuristic"])
    else:
        raise ValueError(f"Unsupported policy {policy}")
    rows, _ = evaluate_policy(
        agent,
        config,
        board_dim,
        task["seeds"],
        task["piece_cap"],
        run_id=task["run_id"],
        experiment=task["experiment"],
        training_seed=task["training_seed"],
        random_policy_seed=random_seed,
        lookahead=task["lookahead"],
        experiment_config=exp_cfg,
    )
    return task["job_id"], rows


def summarize_episodes(rows):
    return {
        "episodes": len(rows),
        "mean_lines": float(np.mean([int(r["lines"]) for r in rows])),
        "episode_std_lines": float(np.std([int(r["lines"]) for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
        "mean_pieces": float(np.mean([int(r["pieces"]) for r in rows])),
        "mean_return": float(np.mean([float(r["return"]) for r in rows])),
        "mean_classic_score": float(np.mean([int(r["classic_score"]) for r in rows])),
        "episode_std_classic_score": float(np.std([int(r["classic_score"]) for r in rows], ddof=1))
        if len(rows) > 1 else 0.0,
        "mean_clears_1": float(np.mean([int(r["clears_1"]) for r in rows])),
        "mean_clears_2": float(np.mean([int(r["clears_2"]) for r in rows])),
        "mean_clears_3": float(np.mean([int(r["clears_3"]) for r in rows])),
        "mean_clears_4": float(np.mean([int(r["clears_4"]) for r in rows])),
        "cap_hit_rate": float(np.mean([bool(r["truncated"]) for r in rows])),
    }


def evaluate_jobs(jobs, config_path, piece_cap, test_seeds, workers=8, chunk_size=20):
    tasks = []
    for job in jobs:
        for start in range(0, len(test_seeds), chunk_size):
            tasks.append({
                "job_id": job["job_id"],
                "config_path": str(job.get("config_path") or config_path),
                "policy": job["policy"],
                "checkpoint": str(job["checkpoint"]) if job.get("checkpoint") else None,
                "board_dim": job.get("board_dim", 3),
                "run_id": job["run_id"],
                "experiment": job["experiment"],
                "training_seed": job.get("training_seed", ""),
                "lookahead": job.get("lookahead", 0),
                "piece_cap": piece_cap,
                "seeds": list(test_seeds[start:start + chunk_size]),
            })
    grouped = {job["job_id"]: [] for job in jobs}
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_chunk, task) for task in tasks]
        completed = 0
        for future in as_completed(futures):
            job_id, rows = future.result()
            grouped[job_id].extend(rows)
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(f"PROGRESS {completed}/{len(tasks)} chunks ({time.perf_counter() - started:.1f}s)",
                      flush=True)
    results = {}
    for job in jobs:
        rows = sorted(grouped[job["job_id"]], key=lambda row: int(row["test_seed"]))
        results[job["job_id"]] = {"job": job, "rows": rows, "summary": summarize_episodes(rows)}
    return results


def build_formal_v1_jobs(batch):
    manifest = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Batch must be complete")
    jobs = []
    for item in manifest["jobs"]:
        folder = resolve_run_directory(batch, item["output"])
        name = item["job"]
        if name in ("random", "heuristic"):
            jobs.append({
                "job_id": name,
                "policy": name,
                "run_id": folder.name,
                "experiment": name,
                "board_dim": 3,
                "training_seed": "",
                "lookahead": 0,
                "expected_mean_lines": item["summary"]["mean_lines"],
                "expected_mean_pieces": item["summary"]["mean_pieces"],
            })
        else:
            ckpt = folder / "best.pt"
            _, meta = DQNAgent.load(ckpt, device="cpu")
            jobs.append({
                "job_id": name,
                "policy": "dqn",
                "checkpoint": ckpt,
                "run_id": folder.name,
                "experiment": meta["experiment"],
                "board_dim": meta["board_feature_dim"],
                "training_seed": meta["seed"],
                "lookahead": 0,
                "expected_mean_lines": item["summary"]["mean_lines"],
                "expected_mean_pieces": item["summary"]["mean_pieces"],
            })
    return jobs


def write_summary_tables(results, jobs, output):
    all_rows = []
    for job in jobs:
        all_rows.extend(results[job["job_id"]]["rows"])
    write_episodes(output / "episodes.csv", all_rows)

    order = []
    for job in jobs:
        exp = job["experiment"]
        if exp not in order:
            order.append(exp)

    seed_rows = []
    for job in jobs:
        if job["training_seed"] == "":
            continue
        s = results[job["job_id"]]["summary"]
        seed_rows.append({
            "method": job["experiment"],
            "training_seed": job["training_seed"],
            "mean_classic_score": s["mean_classic_score"],
            "mean_lines": s["mean_lines"],
            "mean_pieces": s["mean_pieces"],
            "mean_return": s["mean_return"],
            "mean_clears_1": s["mean_clears_1"],
            "mean_clears_2": s["mean_clears_2"],
            "mean_clears_3": s["mean_clears_3"],
            "mean_clears_4": s["mean_clears_4"],
            "cap_hit_rate": s["cap_hit_rate"],
            "run_id": job["run_id"],
        })
    with (output / "training_seed_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)

    summary_rows = []
    for exp in order:
        group = [results[j["job_id"]] for j in jobs if j["experiment"] == exp]
        is_baseline = group[0]["job"]["training_seed"] == ""
        scores = [g["summary"]["mean_classic_score"] for g in group]
        lines = [g["summary"]["mean_lines"] for g in group]
        pieces = [g["summary"]["mean_pieces"] for g in group]
        returns = [g["summary"]["mean_return"] for g in group]
        c1 = [g["summary"]["mean_clears_1"] for g in group]
        c2 = [g["summary"]["mean_clears_2"] for g in group]
        c3 = [g["summary"]["mean_clears_3"] for g in group]
        c4 = [g["summary"]["mean_clears_4"] for g in group]
        caps = [g["summary"]["cap_hit_rate"] for g in group]
        std_score = (group[0]["summary"]["episode_std_classic_score"] if is_baseline
                     else (float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0))
        std_lines = (group[0]["summary"]["episode_std_lines"] if is_baseline
                     else (float(np.std(lines, ddof=1)) if len(lines) > 1 else 0.0))
        summary_rows.append({
            "method": exp,
            "training_seeds": 0 if is_baseline else len(group),
            "mean_classic_score": float(np.mean(scores)),
            "std_classic_score": std_score,
            "mean_lines": float(np.mean(lines)),
            "std_lines": std_lines,
            "mean_pieces": float(np.mean(pieces)),
            "mean_return": float(np.mean(returns)),
            "mean_clears_1": float(np.mean(c1)),
            "mean_clears_2": float(np.mean(c2)),
            "mean_clears_3": float(np.mean(c3)),
            "mean_clears_4": float(np.mean(c4)),
            "cap_hit_rate": float(np.mean(caps)),
            "test_episodes_per_model": group[0]["summary"]["episodes"],
        })
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, default=ROOT / "runs/formal_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/v2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--piece-cap", type=int)
    parser.add_argument("--include-potential", action="store_true")
    parser.add_argument("--reuse-base-episodes", type=Path)
    parser.add_argument("--final-lookahead-dir", type=Path)
    parser.add_argument("--final-lookahead-label", default="final_classic_double_lookahead")
    parser.add_argument("--include-formal-f13-lookahead", action="store_true")
    args = parser.parse_args()

    config_path = args.batch / "config.yaml"
    config = load_config(config_path)
    settings = config["evaluation"]
    count = args.episodes or settings["test_episodes"]
    cap = args.piece_cap or config["environment"]["test_piece_cap"]
    seeds = list(range(settings["test_seed_start"], settings["test_seed_start"] + count))

    base_results = {}
    base_jobs = []
    if args.reuse_base_episodes:
        with args.reuse_base_episodes.open(encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
        by_key = {}
        for r in existing_rows:
            r["truncated"] = r["truncated"].lower() == "true"
            r["terminated"] = r["terminated"].lower() == "true"
            key = (r["experiment"], r["training_seed"], r["run_id"])
            by_key.setdefault(key, []).append(r)
        for (exp, tseed, run_id), rows in by_key.items():
            job_id = f"{exp}_seed{tseed}" if tseed != "" else exp
            job_spec = {"job_id": job_id, "run_id": run_id, "experiment": exp,
                        "training_seed": int(tseed) if tseed != "" else ""}
            base_jobs.append(job_spec)
            base_results[job_id] = {"job": job_spec, "rows": rows, "summary": summarize_episodes(rows)}
    else:
        base_jobs = build_formal_v1_jobs(args.batch)
        if args.include_potential:
            base_jobs.append({
                "job_id": "potential",
                "policy": "potential",
                "run_id": "potential_baseline",
                "experiment": "potential",
                "board_dim": 3,
                "training_seed": "",
                "lookahead": 0,
            })

    jobs_to_run = [] if args.reuse_base_episodes else list(base_jobs)
    extra_jobs = []
    if args.include_formal_f13_lookahead:
        for item in build_formal_v1_jobs(args.batch):
            if item["experiment"] == "dqn_features13":
                extra_jobs.append({
                    **item,
                    "job_id": f"dqn_features13_lookahead_seed{item['training_seed']}",
                    "run_id": f"{item['run_id']}_lookahead",
                    "experiment": "dqn_features13_lookahead",
                    "lookahead": 1,
                    "expected_mean_lines": None,
                    "expected_mean_pieces": None,
                })
    if args.final_lookahead_dir:
        for seed_dir in sorted(args.final_lookahead_dir.glob("*_seed*_attempt*")):
            ep_csv = seed_dir / "episodes.csv"
            ckpt = seed_dir / "best.pt"
            if not ckpt.exists():
                continue
            _, meta = DQNAgent.load(ckpt, device="cpu")
            if ep_csv.exists():
                with ep_csv.open(encoding="utf-8", newline="") as handle:
                    la0_rows = list(csv.DictReader(handle))
                for r in la0_rows:
                    r["truncated"] = r["truncated"].lower() == "true"
                    r["terminated"] = r["terminated"].lower() == "true"
                la0_id = f"{meta['experiment']}_seed{meta['seed']}"
                la0_spec = {"job_id": la0_id, "run_id": seed_dir.name, "experiment": meta["experiment"],
                            "training_seed": meta["seed"]}
                base_jobs.append(la0_spec)
                base_results[la0_id] = {"job": la0_spec, "rows": la0_rows, "summary": summarize_episodes(la0_rows)}
            extra_jobs.append({
                "job_id": f"{args.final_lookahead_label}_seed{meta['seed']}",
                "config_path": seed_dir / "config.yaml",
                "policy": "dqn",
                "checkpoint": ckpt,
                "run_id": f"{seed_dir.name}_lookahead",
                "experiment": args.final_lookahead_label,
                "board_dim": meta["board_feature_dim"],
                "training_seed": meta["seed"],
                "lookahead": 1,
            })

    jobs_to_run.extend(extra_jobs)
    args.output.mkdir(parents=True, exist_ok=True)
    run_results = evaluate_jobs(jobs_to_run, config_path, cap, seeds, workers=args.workers) if jobs_to_run else {}
    all_jobs = base_jobs + extra_jobs if args.reuse_base_episodes else jobs_to_run
    results = {**base_results, **run_results}
    for job in all_jobs:
        exp_lines = job.get("expected_mean_lines")
        if exp_lines is not None and count == settings["test_episodes"] and cap == config["environment"]["test_piece_cap"]:
            got = results[job["job_id"]]["summary"]["mean_lines"]
            if not np.isclose(got, exp_lines):
                raise ValueError(f"{job['job_id']} diverged from formal_v1: {got} != {exp_lines}")
    summary_rows = write_summary_tables(results, all_jobs, args.output)
    write_json(args.output / "summary.json", summary_rows)
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
