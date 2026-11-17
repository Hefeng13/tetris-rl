"""Aggregate recorded results at the training-seed level and plot them."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ROOT, write_json
from .evaluate import write_episodes


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate(batch, output):
    manifest = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Aggregate a fully completed batch")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    episode_rows, groups = [], {}
    for job in manifest["jobs"]:
        folder = Path(job["output"])
        rows = read_csv(folder / "episodes.csv")
        episode_rows.extend(rows)
        experiment = rows[0]["experiment"]
        groups.setdefault(experiment, []).append((folder, job["summary"]))
    write_episodes(output / "episodes.csv", episode_rows)
    order = [name for name in ("random", "heuristic", "dqn", "double_dqn", "dqn_shaped", "dqn_features13")
             if name in groups]
    summary_rows = []
    for name in order:
        models = groups[name]
        values = [summary["mean_lines"] for _, summary in models]
        summary_rows.append({"method": name, "training_seeds": len(models) if name not in ("random", "heuristic") else 0,
            "mean_lines": float(np.mean(values)), "std_across_training_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "mean_pieces": float(np.mean([s["mean_pieces"] for _, s in models])),
            "cap_hit_rate": float(np.mean([s["cap_hit_rate"] for _, s in models])),
            "training_seconds": float(sum(s.get("training_seconds", 0) for _, s in models)),
            "test_episodes_per_model": models[0][1]["episodes"]})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    colors = ["#8b96a5", "#dfad5b", "#3677ad", "#3d9d87", "#9b6db0", "#d67e65"]
    with plt.rc_context({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
        positions = np.arange(len(summary_rows))
        ax.bar(positions, [r["mean_lines"] for r in summary_rows],
               yerr=[r["std_across_training_seeds"] for r in summary_rows], color=colors[:len(positions)], capsize=4)
        ax.set_xticks(positions, [r["method"].replace("_", "\n") for r in summary_rows])
        ax.set_ylabel("Mean cleared lines per test game")
        ax.set_title("Held-out test performance")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        fig.savefig(figures / "test_comparison.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
        for name in order:
            if name in ("random", "heuristic"):
                continue
            runs = [read_csv(folder / "validation.csv") for folder, _ in groups[name]]
            x = np.array([float(row["env_steps"]) for row in runs[0]])
            values = np.array([[float(row["mean_lines"]) for row in run] for run in runs])
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
            ax.plot(x, mean, label=name)
            ax.fill_between(x, np.maximum(mean - std, 0), mean + std, alpha=0.12)
        ax.set_xlabel("Training piece placements")
        ax.set_ylabel("Validation mean cleared lines")
        ax.set_title("Learning curves across training seeds")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.savefig(figures / "learning_curves.png", dpi=180)
        plt.close(fig)
    write_json(output / "provenance.json", {"batch": str(batch), "manifest": manifest})
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/final")
    args = parser.parse_args()
    aggregate(args.batch, args.output)


if __name__ == "__main__":
    main()
