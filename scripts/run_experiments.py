"""Run the frozen experiment matrix with bounded CPU concurrency.

Each process owns a run directory. The parent owns the batch manifest.
Re-running the same batch skips successful jobs and retries incomplete jobs in
a fresh attempt directory, preserving earlier diagnostics.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import yaml
import torch

from tetris_rl.config import ROOT, load_config, provenance, write_json


def run_job(job, config_path, batch, config):
    name, policy, seed = job
    log_dir = batch / "logs"
    log_dir.mkdir(exist_ok=True)
    for attempt in range(1, 3):
        output = batch / f"{name}_attempt{attempt}"
        result_path = output / "summary.json"
        if result_path.exists():
            summary = json.loads(result_path.read_text(encoding="utf-8"))
            if summary["episodes"] == config["evaluation"]["test_episodes"]:
                return {"job": name, "status": "complete", "output": str(output),
                        "summary": summary}
        if output.exists():
            continue
        if policy in ("random", "heuristic"):
            command = [sys.executable, "-u", "-m", "tetris_rl.evaluate", "--config", str(config_path),
                       "--policy", policy, "--output", str(output)]
        else:
            command = [sys.executable, "-u", "-m", "tetris_rl.train", "--config", str(config_path),
                       "--experiment", policy, "--seed", str(seed), "--output", str(output)]
        with (log_dir / f"{name}_attempt{attempt}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if completed.returncode == 0 and result_path.exists():
            summary = json.loads(result_path.read_text(encoding="utf-8"))
            return {"job": name, "status": "complete", "output": str(output), "summary": summary}
    return {"job": name, "status": "failed", "log_directory": str(log_dir)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments.yaml")
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    batch = (args.batch or ROOT / "runs" / f"formal_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    batch.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    device = config["training"].get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Use a CUDA-enabled Python environment for --device cuda")
    config_path = batch / "config.yaml"
    if config_path.exists():
        if load_config(config_path) != config:
            raise ValueError("Existing batch has a different frozen configuration")
    else:
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    jobs = [("random", "random", None), ("heuristic", "heuristic", None)]
    jobs.extend((f"{experiment['id']}_seed{seed}", experiment["id"], seed)
                for experiment in config["experiments"] for seed in config["training"]["seeds"])
    current_provenance = provenance(device)
    manifest_path = batch / "batch.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["provenance"]["source_sha256"] != current_provenance["source_sha256"]:
            raise ValueError("Source changed since this batch started. Use a new batch directory.")
        if existing["provenance"]["packages"] != current_provenance["packages"]:
            raise ValueError("Dependency versions changed. Use a new batch directory.")
    metadata = {"status": "running", "total_jobs": len(jobs), "workers": args.workers,
                "provenance": current_provenance, "jobs": []}
    write_json(batch / "batch.json", metadata)
    print(f"BATCH {batch} jobs={len(jobs)} workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job, config_path, batch, config) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            metadata["jobs"].append(result)
            write_json(batch / "batch.json", metadata)
            score = result.get("summary", {}).get("mean_lines", "failed")
            print(f"FINISHED {len(metadata['jobs'])}/{len(jobs)} {result['job']} lines={score}", flush=True)
    failed = [job for job in metadata["jobs"] if job["status"] != "complete"]
    metadata["status"] = "failed" if failed else "complete"
    write_json(batch / "batch.json", metadata)
    if failed:
        raise RuntimeError(f"Failed jobs: {[job['job'] for job in failed]}")
    print("ALL EXPERIMENTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
