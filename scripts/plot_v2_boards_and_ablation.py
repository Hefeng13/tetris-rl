"""Plot 7-method board comparison, 13D vs Final long-horizon boards, and final ablation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
import torch

from tetris_rl.agents.dqn_agent import DQNAgent
from tetris_rl.agents.heuristic_agent import HeuristicAgent
from tetris_rl.agents.lookahead import LookaheadAgent
from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.config import ROOT, load_config, make_env, resolve_run_directory
from tetris_rl.evaluate import CLASSIC_LINE_SCORES


def save_fig(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=200)
    fig.savefig(directory / f"{name}.pdf")
    plt.close(fig)


def rollout_board_snapshots(agent, config, board_dim, test_seed, targets, cap=500,
                            shaping=False, lookahead=0, exp_cfg=None):
    env = make_env(config, board_dim, cap, shaping, experiment=exp_cfg)
    decision_agent = LookaheadAgent(agent, env) if lookahead == 1 else agent
    candidates = env.reset(test_seed)
    palette = {piece.id: list(piece.color_rgb) for piece in env.engine.tetrominoes}
    lines, classic_score = 0, 0
    snapshots = {}
    last = None
    while True:
        candidates, _, terminated, truncated, info = env.step(decision_agent.act(candidates))
        cleared = int(info["lines_cleared"])
        lines += cleared
        classic_score += CLASSIC_LINE_SCORES[cleared]
        if env.pieces in targets or terminated or truncated:
            last = {
                "actual_move": env.pieces,
                "lines": lines,
                "classic_score": classic_score,
                "board": env.board.copy(),
                "terminated": terminated,
            }
            if env.pieces in targets:
                snapshots[env.pieces] = last
        if terminated or truncated or env.pieces >= max(targets):
            break
    env.close()
    return palette, [snapshots.get(t, last) for t in targets]


def render_board_grid(examples, targets, title, subtitle, footnote, out_dir, filename, figsize):
    fig, axes = plt.subplots(len(targets), len(examples), figsize=figsize)
    if len(examples) == 1:
        axes = axes[:, None]
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.08, wspace=0.22, hspace=0.26)
    for col, (label, palette, snaps) in enumerate(examples):
        color_list = ["#eef1f5"] * (max(palette) + 1)
        for pid, rgb in palette.items():
            color_list[pid] = np.array(rgb) / 255.0
        cmap = ListedColormap(color_list)
        for row, snap in enumerate(snaps):
            ax = axes[row, col]
            board = snap["board"]
            ax.imshow(board, cmap=cmap, vmin=0, vmax=len(color_list) - 1,
                      interpolation="nearest", origin="upper")
            ax.set_xticks(np.arange(-0.5, board.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, board.shape[0], 1), minor=True)
            ax.grid(which="minor", color="#cbd2dc", linewidth=0.35)
            ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
            end_tag = "结束" if snap["terminated"] else ""
            sub = f"第{snap['actual_move']}步{end_tag} · {snap['lines']}行 · {snap['classic_score']:,}分"
            if row == 0:
                sub = f"{label}\n{sub}"
            ax.set_title(sub, fontsize=9.3, pad=7)
            if col == 0:
                ax.set_ylabel(f"第 {targets[row]} 次落子", fontsize=12, labelpad=10)
    fig.suptitle(title, fontsize=20, y=0.965)
    fig.text(0.5, 0.915, subtitle, ha="center", fontsize=11.5)
    fig.text(0.055, 0.025, footnote, fontsize=10.5, color="#555555")
    save_fig(fig, out_dir, filename)


def plot_final_ablation(summary_path, figures_dir):
    with Path(summary_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {r["method"]: r for r in rows}
    methods = ["dqn_features13", "final_classic_double", "final_classic_double_lookahead"]
    labels = ["13 维 DQN\n(基准)", "+ 长视野经典奖励\n(不开前瞻)", "+ 1 步前瞻\n(最终方法)"]
    colors = ["#d67e65", "#e08d4f", "#b83227"]

    scores = [float(by_name[m]["mean_classic_score"]) for m in methods]
    caps = [float(by_name[m]["cap_hit_rate"]) * 100.0 for m in methods]
    tetrises = [float(by_name[m]["mean_clears_4"]) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.2))
    fig.subplots_adjust(left=0.08, right=0.91, top=0.73, bottom=0.17, wspace=0.30)
    x = np.arange(len(methods))

    b1 = ax1.bar(x, scores, width=0.50, color=colors, zorder=3)
    ax1.set_ylim(0, max(scores) * 1.20)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    for bar, val in zip(b1, scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + max(scores) * 0.025,
                 f"{val:,.0f} 分", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax1.set_xticks(x, labels, fontsize=10.5)
    ax1.set_ylabel("平均经典得分", fontsize=11)
    ax1.set_title("① 经典得分跃升", fontsize=12.5, pad=18, fontweight="bold")
    ax1.grid(axis="y", alpha=0.20)

    width = 0.34
    b2 = ax2.bar(x - width / 2, caps, width=width, color="#4f7cac", label="3000 块存活率 (%)", zorder=3)
    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    b3 = ax2_r.bar(x + width / 2, tetrises, width=width, color="#c83e37", label="平均四连消次数 (次/局)", zorder=3)
    ax2.set_ylim(0, 118)
    ax2_r.set_ylim(0, max(tetrises) * 1.22)
    for bar, val in zip(b2, caps):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 2.2, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#2f5782")
    for bar, val in zip(b3, tetrises):
        ax2_r.text(bar.get_x() + bar.get_width() / 2, val + max(tetrises) * 0.025, f"{val:.1f}次",
                   ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#9c2721")
    ax2.set_xticks(x, labels, fontsize=10.5)
    ax2.set_ylabel("3000 块存活率 (%)", fontsize=11, color="#2f5782")
    ax2_r.set_ylabel("每局四连消次数", fontsize=11, color="#9c2721", labelpad=10)
    ax2.set_title("② 为什么必须结合两者？（存活率 × 四连消）", fontsize=12.5, pad=26, fontweight="bold")
    ax2.grid(axis="y", alpha=0.20)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.005),
               ncol=2, frameon=False, fontsize=9.5)

    fig.suptitle("最终方法内部消融：长视野经典奖励与 1 步前瞻如何协同", fontsize=18.5, y=0.955)
    fig.text(0.5, 0.875, "长视野让模型学会打四连消但容易翻车（42.6% 存活），加上 1 步前瞻把存活救回 93.0% 并让得分翻倍",
             ha="center", fontsize=11)
    save_fig(fig, figures_dir, "final_ablation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, default=ROOT / "runs/formal_v1")
    parser.add_argument("--final-checkpoint", type=Path,
                        default=ROOT / "runs/v2_final/final_classic_double_seed0_attempt1/best.pt")
    parser.add_argument("--results", type=Path, default=ROOT / "results/v2")
    parser.add_argument("--test-seed", type=int, default=40000)
    args = parser.parse_args()

    torch.set_num_threads(1)
    config = load_config(args.batch / "config.yaml")
    manifest = json.loads((args.batch / "batch.json").read_text(encoding="utf-8"))
    figures_dir = args.results / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    def load_formal(job_name):
        job = next(j for j in manifest["jobs"] if j["job"] == job_name)
        ckpt = resolve_run_directory(args.batch, job["output"]) / "best.pt"
        return DQNAgent.load(ckpt, device="cpu")

    dqn3, _ = load_formal("dqn_seed0")
    ddqn3, _ = load_formal("double_dqn_seed0")
    shaped3, _ = load_formal("dqn_shaped_seed0")
    f13, _ = load_formal("dqn_features13_seed0")
    final_agent, final_meta = DQNAgent.load(args.final_checkpoint, device="cpu")

    specs = [
        ("随机放置", RandomAgent(config["evaluation"]["random_baseline_policy_seed"] + args.test_seed), 3, False, 0, None),
        ("人工固定规则", HeuristicAgent(**config["heuristic"]), 3, False, 0, None),
        ("3 维 DQN", dqn3, 3, False, 0, None),
        ("Double DQN", ddqn3, 3, False, 0, None),
        ("奖励塑形 DQN", shaped3, 3, True, 0, None),
        ("13 维 DQN", f13, 13, False, 0, None),
        ("最终方法 (前瞻)", final_agent, final_meta["board_feature_dim"], False, 1, final_meta.get("experiment_config")),
    ]

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }):
        examples_all7 = []
        for label, agent, dim, shaping, la, exp_cfg in specs:
            palette, snaps = rollout_board_snapshots(agent, config, dim, args.test_seed, (10, 20, 50),
                                                     cap=100, shaping=shaping, lookahead=la, exp_cfg=exp_cfg)
            examples_all7.append((label, palette, snaps))
        render_board_grid(
            examples_all7, (10, 20, 50),
            "同一方块序列下的棋盘变化（全部 7 组对比）",
            f"测试种子 {args.test_seed}，学习模型使用训练种子 0；第 10、20、50 次落子对比",
            "提前结束的方法保留最终棋盘，标题标明步数、消行数及经典得分。",
            figures_dir, "board_comparison_all7", figsize=(16.5, 11.0),
        )

        examples_long = []
        for label, agent, dim, shaping, la, exp_cfg in specs[-2:]:
            palette, snaps = rollout_board_snapshots(agent, config, dim, args.test_seed, (100, 200, 400),
                                                     cap=500, shaping=shaping, lookahead=la, exp_cfg=exp_cfg)
            examples_long.append((label, palette, snaps))
        render_board_grid(
            examples_long, (100, 200, 400),
            "中长局棋盘策略差异：13 维 DQN vs 最终方法",
            f"测试种子 {args.test_seed}，训练种子 0；第 100、200、400 次落子对比",
            "左侧 13 维 DQN 压低苟活、主要单/双消；右侧最终方法主动留槽叠层、高频打出三/四连消。",
            figures_dir, "board_13d_vs_final", figsize=(9.5, 11.0),
        )

        plot_final_ablation(args.results / "summary.csv", figures_dir)

    print(f"Saved board comparisons and ablation figures to {figures_dir}")


if __name__ == "__main__":
    main()
