"""Plot seed stability, evaluation-cap counts and matched board snapshots."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import numpy as np
import torch

from tetris_rl.agents.dqn_agent import DQNAgent
from tetris_rl.agents.heuristic_agent import HeuristicAgent
from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.config import load_config, make_env, resolve_run_directory, write_json
from tetris_rl.plot import read_csv


METHODS = ("random", "heuristic", "dqn", "double_dqn", "dqn_shaped", "dqn_features13")
LABELS = {
    "random": "随机放置", "heuristic": "人工固定规则", "dqn": "3 维 DQN",
    "double_dqn": "Double DQN", "dqn_shaped": "奖励塑形 DQN", "dqn_features13": "13 维 DQN",
}
COLORS = dict(zip(METHODS, ("#8b96a5", "#dfad5b", "#3677ad", "#3d9d87", "#9b6db0", "#d67e65")))


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=200)
    fig.savefig(directory / f"{name}.pdf")
    plt.close(fig)


def read_runs(batch):
    manifest = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Plot a fully completed batch")
    runs = []
    for job in manifest["jobs"]:
        folder = resolve_run_directory(batch, job["output"])
        rows = read_csv(folder / "episodes.csv")
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if not np.isclose(np.mean([int(row["lines"]) for row in rows]), summary["mean_lines"]):
            raise ValueError(f"Summary and episodes disagree in {folder.name}")
        runs.append({"folder": folder, "method": rows[0]["experiment"], "rows": rows, "summary": summary})
    return runs


def plot_seed_scores(runs, output):
    rows = []
    for run in runs:
        if run["method"] in ("random", "heuristic"):
            continue
        summary = run["summary"]
        rows.append({"method": run["method"], "training_seed": summary["training_seed"],
                     "mean_lines": summary["mean_lines"], "test_episodes": summary["episodes"],
                     "checkpoint_step": summary["best_checkpoint_step"], "run_id": run["folder"].name})
    rows.sort(key=lambda row: (METHODS.index(row["method"]), row["training_seed"]))
    write_table(output / "training_seed_scores.csv", rows)
    methods = [method for method in METHODS[2:] if any(row["method"] == method for row in rows)]
    fig, ax = plt.subplots(figsize=(12, 6.6))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.83, bottom=0.20)
    for position, method in enumerate(methods):
        selected = [row for row in rows if row["method"] == method]
        offsets = np.linspace(-0.24, 0.24, len(selected))
        values = [row["mean_lines"] for row in selected]
        for offset, row in zip(offsets, selected):
            ax.scatter(position + offset, row["mean_lines"], s=90,
                       facecolor="white" if row["checkpoint_step"] == 0 else COLORS[method],
                       edgecolor=COLORS[method], linewidth=2, zorder=3)
        mean = float(np.mean(values))
        ax.hlines(mean, position - 0.32, position + 0.32, color="#222222", linewidth=1.5, zorder=2)
        ax.annotate(f"均值 {mean:.2f}", (position, max(values)), xytext=(0, 16),
                    textcoords="offset points", ha="center", fontsize=12)
    heuristic = next(run for run in runs if run["method"] == "heuristic")
    ax.axhline(heuristic["summary"]["mean_lines"], color=COLORS["heuristic"], linestyle="--", linewidth=1.5)
    ax.set_xticks(range(len(methods)), [LABELS[method] for method in methods], fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(5, max(row["mean_lines"] for row in rows) * 2)
    ax.set_yticks([5, 10, 20, 50, 100, 300, 1000, 2000])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel("每份模型的平均消行数（对数刻度）", fontsize=12)
    ax.grid(axis="y", alpha=0.20)
    ax.legend(handles=[Line2D([], [], color=COLORS["heuristic"], linestyle="--", label="人工规则的平均成绩"),
                       Line2D([], [], color="#222222", linewidth=1.5, label="五份模型的平均成绩")],
              loc="upper left", frameon=False, fontsize=11)
    fig.suptitle("重复训练五次，成绩有多稳定？", fontsize=21, y=0.97)
    fig.text(0.5, 0.89, "每个点是一份模型在相同 100 个测试序列上的平均成绩", ha="center", fontsize=12)
    fig.text(0.10, 0.07, "同组从左到右为训练种子 0–4。空心点为选中的初始化模型（训练步数为 0）。", fontsize=11)
    fig.text(0.10, 0.025, "每局上限 3000 次落子。点的位置表示成绩，横向间距用于区分五个训练种子。", fontsize=10, color="#555555")
    save_figure(fig, output / "figures", "training_seed_scores")


def plot_cap_hits(runs, output):
    rows = []
    for method in METHODS:
        episodes = [row for run in runs if run["method"] == method for row in run["rows"]]
        if not episodes:
            continue
        caps = {int(row["piece_cap"]) for row in episodes}
        if len(caps) != 1:
            raise ValueError("Cap-hit comparisons require one common placement cap")
        hits = sum(row["truncated"].lower() == "true" for row in episodes)
        rows.append({"method": method, "cap_hits": hits, "test_episodes": len(episodes),
                     "cap_hit_percent": hits / len(episodes) * 100, "piece_cap": caps.pop()})
    if len({row["piece_cap"] for row in rows}) != 1:
        raise ValueError("Methods have different placement caps")
    write_table(output / "cap_hit_rates.csv", rows)
    fig, ax = plt.subplots(figsize=(12, 6.6))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.80, bottom=0.16)
    for position, row in enumerate(rows):
        value = row["cap_hit_percent"]
        ax.barh(position, value, height=0.55, color=COLORS[row["method"]], zorder=2)
        ax.plot(value, position, "o", color=COLORS[row["method"]], markersize=5, zorder=3)
        ax.text(value + 1.5, position, f"{value:.1f}%（{row['cap_hits']}/{row['test_episodes']} 局）",
                va="center", fontsize=12)
    ax.set_yticks(range(len(rows)), [LABELS[row["method"]] for row in rows], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(-1, 135)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("达到评估落子上限的测试局比例", fontsize=12)
    ax.grid(axis="x", alpha=0.20)
    fig.suptitle(f"有多少局能坚持到 {rows[0]['piece_cap']} 次落子？", fontsize=21, y=0.97)
    fig.text(0.5, 0.88, "学习方法各汇总 5 份模型、500 局测试。两种基线各测试 100 局。", ha="center", fontsize=12)
    fig.text(0.18, 0.045, "括号中为达到上限的局数 / 测试局数。达到上限的游戏由评估程序结束。", fontsize=11)
    save_figure(fig, output / "figures", "cap_hit_rates")


def replay_snapshots(run, config, test_seed, targets):
    method = run["method"]
    shaping = False
    checkpoint = None
    if method == "random":
        agent = RandomAgent(config["evaluation"]["random_baseline_policy_seed"] + test_seed)
        board_dim = 3
    elif method == "heuristic":
        agent = HeuristicAgent(**config["heuristic"])
        board_dim = 3
    else:
        agent, metadata = DQNAgent.load(run["folder"] / "best.pt")
        board_dim = metadata["board_feature_dim"]
        shaping = next(item["shaping_enabled"] for item in config["experiments"] if item["id"] == method)
        checkpoint = f"{run['folder'].name}/best.pt"
    recorded = next(row for row in run["rows"] if int(row["test_seed"]) == test_seed)
    env = make_env(config, board_dim, int(recorded["piece_cap"]), shaping)
    try:
        candidates = env.reset(test_seed)
        palette = {piece.id: list(piece.color_rgb) for piece in env.engine.tetrominoes}
        lines, total_return = 0, 0.0
        snapshots, piece_sequence = {}, []
        while True:
            if env.pieces < max(targets):
                piece_sequence.append(int(env.engine.active_tetromino.id))
            candidates, reward, terminated, truncated, info = env.step(agent.act(candidates))
            lines += int(info["lines_cleared"])
            total_return += reward
            if env.pieces in targets or terminated or truncated:
                current = {"actual_move": env.pieces, "lines": lines, "board": env.board.tolist(),
                           "terminated": terminated, "truncated": truncated}
                if env.pieces in targets:
                    snapshots[env.pieces] = current
            if terminated or truncated:
                break
        if (env.pieces != int(recorded["pieces"]) or lines != int(recorded["lines"])
                or not np.isclose(total_return, float(recorded["return"]))
                or terminated != (recorded["terminated"].lower() == "true")
                or truncated != (recorded["truncated"].lower() == "true")):
            raise ValueError(f"Replay disagrees with formal evaluation for {method}")
        return {"method": method, "checkpoint": checkpoint, "run_id": run["folder"].name,
                "piece_sequence": piece_sequence, "palette": palette,
                "snapshots": [{"requested_move": move, **snapshots.get(move, current)} for move in targets],
                "formal_test_replay_verified": True, "final_moves": env.pieces, "final_lines": lines}
    finally:
        env.close()


def plot_board_comparison(runs, config, output, test_seed, training_seed):
    targets = (10, 20, 50)
    examples = []
    for method in METHODS:
        run = next(run for run in runs if run["method"] == method and (
            method in ("random", "heuristic") or run["summary"]["training_seed"] == training_seed))
        examples.append(replay_snapshots(run, config, test_seed, targets))
    common_length = min(len(example["piece_sequence"]) for example in examples)
    reference = examples[0]["piece_sequence"][:common_length]
    if any(example["piece_sequence"][:common_length] != reference for example in examples):
        raise ValueError("Methods did not receive matching piece-sequence prefixes")
    write_json(output / "board_comparison.json", {"test_seed": test_seed, "training_seed": training_seed,
        "requested_moves": targets, "matched_piece_prefix_length": common_length, "examples": examples})
    fig, axes = plt.subplots(len(targets), len(examples), figsize=(14, 12))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.075, wspace=0.23, hspace=0.24)
    for column, example in enumerate(examples):
        palette = example["palette"]
        color_list = ["#eef1f5"] * (max(palette) + 1)
        for piece_id, color in palette.items():
            color_list[piece_id] = np.array(color) / 255
        for row, snapshot in enumerate(example["snapshots"]):
            ax = axes[row, column]
            board = np.array(snapshot["board"])
            ax.imshow(board, cmap=ListedColormap(color_list), vmin=0, vmax=len(color_list) - 1,
                      interpolation="nearest", origin="upper")
            ax.set_xticks(np.arange(-0.5, board.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, board.shape[0], 1), minor=True)
            ax.grid(which="minor", color="#cbd2dc", linewidth=0.35)
            ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
            ending = "结束" if snapshot["terminated"] else ""
            title = f"第 {snapshot['actual_move']} 步{ending} · 消行 {snapshot['lines']}"
            if row == 0:
                title = LABELS[example["method"]] + "\n" + title
            ax.set_title(title, fontsize=10.5, pad=8)
            if column == 0:
                ax.set_ylabel(f"第 {targets[row]} 次落子", fontsize=13, labelpad=12)
    fig.suptitle("同一方块序列下的棋盘变化", fontsize=23, y=0.98)
    fig.text(0.5, 0.935, f"测试种子 {test_seed}，学习模型使用训练种子 {training_seed} 的选中检查点", ha="center", fontsize=13)
    fig.text(0.06, 0.035, "提前结束的方法保留最终棋盘，标题标明实际结束步数。棋盘仅显示已经落下的方块。", fontsize=12)
    fig.text(0.06, 0.01, "这是一个固定序列的行为示例。整体表现以全部测试序列的统计为依据。", fontsize=11, color="#555555")
    save_figure(fig, output / "figures", "board_comparison")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-seed", type=int, default=40000)
    parser.add_argument("--training-seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    runs = read_runs(args.batch)
    config = load_config(args.batch / "config.yaml")
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}):
        plot_seed_scores(runs, args.output)
        plot_cap_hits(runs, args.output)
        plot_board_comparison(runs, config, args.output, args.test_seed, args.training_seed)
    print(f"Saved three figures and their source data to {args.output}", flush=True)


if __name__ == "__main__":
    main()
