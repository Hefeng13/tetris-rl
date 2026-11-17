"""Plot formal model scores, game distributions, survival, runtime and boards."""

import argparse
import csv
import inspect
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


def plot_episode_distribution(runs, output):
    methods, observations, summary_rows = [], [], []
    for method in METHODS:
        values = np.array([int(row["lines"]) for run in runs if run["method"] == method
                           for row in run["rows"]])
        if not len(values):
            continue
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        methods.append(method)
        observations.append(values)
        summary_rows.append({"method": method, "test_episodes": len(values),
            "minimum_lines": int(values.min()), "q25_lines": float(q25), "median_lines": float(median),
            "q75_lines": float(q75), "maximum_lines": int(values.max()), "mean_lines": float(values.mean()),
            "zero_line_games": int((values == 0).sum())})
    write_table(output / "episode_distribution_summary.csv", summary_rows)
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.subplots_adjust(left=0.20, right=0.80, top=0.82, bottom=0.22)
    orientation = {"vert": False}
    if "orientation" in inspect.signature(ax.boxplot).parameters:
        orientation = {"orientation": "horizontal"}
    boxes = ax.boxplot(observations, positions=range(len(methods)), **orientation,
                       widths=0.40, whis=(0, 100), showfliers=False, patch_artist=True,
                       medianprops={"color": "#222222", "linewidth": 2, "zorder": 4},
                       whiskerprops={"color": "#555555"}, capprops={"color": "#555555"})
    rng = np.random.default_rng(42)
    for position, (method, values, box, summary) in enumerate(zip(methods, observations, boxes["boxes"], summary_rows)):
        box.set_facecolor(COLORS[method])
        box.set_alpha(0.35)
        ax.scatter(values, position + rng.uniform(-0.18, 0.18, len(values)),
                   s=10, color=COLORS[method], alpha=0.18, edgecolors="none", zorder=1)
        ax.text(1.03, position, f"中位数 {summary['median_lines']:g}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=12)
    ax.set_xscale("symlog", linthresh=1, linscale=0.5)
    ax.set_xlim(-0.08, max(float(values.max()) for values in observations) * 1.2 + 1)
    ax.set_xticks([0, 1, 5, 10, 50, 100, 500, 1200])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks(range(len(methods)), [f"{LABELS[method]}\n{len(values)} 局" for method, values in zip(methods, observations)], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("每局消行数", fontsize=12)
    ax.grid(axis="x", alpha=0.20)
    fig.suptitle("每局成绩分布，平均值背后是什么？", fontsize=21, y=0.97)
    fig.text(0.5, 0.89, "每个淡点代表一局测试，学习方法合并五份模型的测试记录", ha="center", fontsize=12)
    fig.text(0.20, 0.105, "盒子覆盖中间 50% 的成绩，盒内黑线为中位数，横线覆盖最低到最高成绩。", fontsize=11)
    fig.text(0.20, 0.04, "横轴 0–1 区间为线性刻度，1 以上为对数刻度。每局上限 3000 次落子。", fontsize=11, color="#555555")
    save_figure(fig, output / "figures", "episode_lines_distribution")


def survival_points(episodes):
    """Empirical natural survival with a common cap and endpoint censoring."""
    caps = {int(row["piece_cap"]) for row in episodes}
    if len(caps) != 1:
        raise ValueError("A survival curve requires one common placement cap")
    cap = caps.pop()
    terminations, censored = [], 0
    for row in episodes:
        pieces = int(row["pieces"])
        terminated = row["terminated"].lower() == "true"
        truncated = row["truncated"].lower() == "true"
        if terminated == truncated or not 0 < pieces <= cap:
            raise ValueError("Each completed episode needs a valid ending")
        if truncated:
            if pieces != cap:
                raise ValueError("This curve supports censoring at the common cap")
            censored += 1
        else:
            terminations.append(pieces)
    terminations = np.sort(terminations)
    moves = np.unique(np.concatenate(([0, cap], terminations))).astype(int)
    remaining = len(episodes) - np.searchsorted(terminations, moves, side="right")
    return moves, remaining, cap, censored


def plot_survival_curves(runs, output):
    curves, rows = [], []
    for method in METHODS:
        episodes = [row for run in runs if run["method"] == method for row in run["rows"]]
        if not episodes:
            continue
        moves, remaining, cap, censored = survival_points(episodes)
        curves.append((method, moves, remaining / len(episodes) * 100, cap))
        rows.extend({"method": method, "placements": int(move), "remaining_games": int(count),
                     "test_episodes": len(episodes), "remaining_percent": float(count / len(episodes) * 100),
                     "cap_censored_here": censored if move == cap else 0}
                    for move, count in zip(moves, remaining))
    caps = {curve[3] for curve in curves}
    if len(caps) != 1:
        raise ValueError("Methods have different placement caps")
    cap = caps.pop()
    write_table(output / "survival_curve_data.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.73, bottom=0.22, wspace=0.10)
    styles = (":", "--", "-", "-.", (0, (4, 1, 1, 1)), "-")
    for method, moves, remaining_percent, _ in curves:
        for ax in axes:
            ax.step(moves, remaining_percent, where="post", label=LABELS[method],
                    color=COLORS[method], linestyle=styles[METHODS.index(method)], linewidth=2)
        axes[1].plot(cap, remaining_percent[-1], marker="|", color=COLORS[method], markersize=10)
    axes[0].set_xlim(0, min(300, cap))
    axes[1].set_xlim(0, cap * 1.01)
    axes[0].set_title("开局阶段", fontsize=13)
    axes[1].set_title("完整评估长度", fontsize=13)
    for ax in axes:
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("已经进行的落子决策次数", fontsize=12)
        ax.grid(alpha=0.20)
    axes[0].set_ylabel("尚未自然结束的测试局比例", fontsize=12)
    feature_curve = next(curve for curve in curves if curve[0] == "dqn_features13")
    axes[1].annotate(f"13 维 DQN\n上限处 {feature_curve[2][-1]:.1f}%", (cap, feature_curve[2][-1]),
                     xytext=(-18, -28), textcoords="offset points", ha="right", va="top", fontsize=12)
    fig.suptitle("每放下一块，还有多少局能继续？", fontsize=21, y=0.97)
    fig.text(0.5, 0.89, "学习方法各汇总 500 局，两种基线各汇总 100 局，使用相同的测试方块序列", ha="center", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.85), ncol=6, frameon=False, fontsize=10)
    fig.text(0.07, 0.10, f"达到 {cap} 次落子的局在上限处计为仍存活，随后由评估程序结束。曲线展示到该上限。", fontsize=11)
    fig.text(0.07, 0.04, "落子次数包含导致游戏结束的最后一次尝试。", fontsize=11, color="#555555")
    save_figure(fig, output / "figures", "survival_curves")


def plot_training_time(runs, output):
    rows, summaries = [], []
    for method in METHODS[2:]:
        models = [run for run in runs if run["method"] == method]
        if not models:
            continue
        for run in sorted(models, key=lambda run: run["summary"]["training_seed"]):
            summary = run["summary"]
            rows.append({"method": method, "training_seed": summary["training_seed"],
                         "training_elapsed_seconds": summary["training_seconds"],
                         "mean_test_lines": summary["mean_lines"], "training_steps": summary["steps"],
                         "selected_checkpoint_step": summary["best_checkpoint_step"], "run_id": run["folder"].name})
        summaries.append({"method": method, "training_runs": len(models),
            "mean_elapsed_minutes": float(np.mean([run["summary"]["training_seconds"] for run in models]) / 60),
            "mean_test_lines": float(np.mean([run["summary"]["mean_lines"] for run in models])),
            "training_steps_per_run": models[0]["summary"]["steps"]})
    write_table(output / "training_time_scores.csv", rows)
    write_table(output / "training_time_summary.csv", summaries)
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.25)
    offsets = {"dqn": (-160, 10), "double_dqn": (55, 8), "dqn_shaped": (20, 10), "dqn_features13": (-20, -45)}
    for summary in summaries:
        method = summary["method"]
        minutes, score = summary["mean_elapsed_minutes"], summary["mean_test_lines"]
        ax.scatter(minutes, score, s=150, color=COLORS[method], edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(f"{LABELS[method]}\n{minutes:.2f} 分钟 · {score:.2f} 行", (minutes, score),
                    xytext=offsets[method], textcoords="offset points", fontsize=12,
                    ha="right" if method == "dqn_features13" else "left",
                    arrowprops={"arrowstyle": "-", "color": "#888888", "linewidth": 0.7})
    heuristic = next(run for run in runs if run["method"] == "heuristic")
    ax.axhline(heuristic["summary"]["mean_lines"], color=COLORS["heuristic"], linestyle="--", linewidth=1.5,
               label="人工规则的平均成绩")
    ax.set_xlim(0, max(row["mean_elapsed_minutes"] for row in summaries) * 1.18)
    ax.set_yscale("log")
    ax.set_ylim(min(row["mean_test_lines"] for row in summaries) / 2,
                max(row["mean_test_lines"] for row in summaries) * 2)
    ax.set_yticks([10, 20, 50, 100, 300, 1000, 2000])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("单份模型的平均运行时间（分钟，覆盖训练和定期验证）", fontsize=12)
    ax.set_ylabel("平均测试消行数（对数刻度）", fontsize=12)
    ax.grid(alpha=0.20)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    fig.suptitle("这次运行花了多久，取得什么成绩？", fontsize=21, y=0.97)
    fig.text(0.5, 0.89, "每个点汇总一种方法的五次运行，每次训练 100000 次落子", ha="center", fontsize=12)
    fig.text(0.10, 0.12, "本机 CPU、四任务并行时的实际记录。时间覆盖训练、定期验证和检查点保存。", fontsize=11)
    fig.text(0.10, 0.055, "奖励塑形组包含两份选中的初始化模型，用时按完整训练流程统计。", fontsize=11, color="#555555")
    save_figure(fig, output / "figures", "training_time_vs_score")


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
    plotters = {"seed_scores": plot_seed_scores, "cap_hits": plot_cap_hits,
                "distribution": plot_episode_distribution, "survival": plot_survival_curves,
                "training_time": plot_training_time}
    parser.add_argument("--figures", nargs="+", choices=[*plotters, "boards"],
                        default=["seed_scores", "cap_hits", "boards", "distribution", "survival", "training_time"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    runs = read_runs(args.batch)
    config = load_config(args.batch / "config.yaml")
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}):
        for name in args.figures:
            if name == "boards":
                plot_board_comparison(runs, config, args.output, args.test_seed, args.training_seed)
            else:
                plotters[name](runs, args.output)
    print(f"Saved {len(args.figures)} figures and their source data to {args.output}", flush=True)


if __name__ == "__main__":
    main()
