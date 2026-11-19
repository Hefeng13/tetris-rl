"""Generate v2 figures comparing classic Tetris scores and multi-line clear composition."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from tetris_rl.config import ROOT


LABELS = {
    "random": "随机策略",
    "heuristic": "人工启发式",
    "dqn": "3 维 DQN",
    "double_dqn": "Double DQN",
    "dqn_shaped": "奖励塑形 DQN",
    "dqn_features13": "13 维 DQN",
    "dqn_features13_lookahead": "13 维 DQN\n+ 1 步前瞻",
    "final_classic_double": "最终方法\n(无前瞻)",
    "final_classic_double_lookahead": "最终方法\n(13维Double+前瞻)",
}
COLORS = {
    "random": "#8b96a5",
    "heuristic": "#dfad5b",
    "dqn": "#3677ad",
    "double_dqn": "#3d9d87",
    "dqn_shaped": "#9b6db0",
    "dqn_features13": "#d67e65",
    "dqn_features13_lookahead": "#c05640",
    "final_classic_double": "#b84a39",
    "final_classic_double_lookahead": "#b83227",
}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_fig(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=200)
    fig.savefig(directory / f"{name}.pdf")
    plt.close(fig)


def plot_classic_comparison(summary_rows, final_method, figures_dir):
    from matplotlib.ticker import FuncFormatter
    order = ["random", "heuristic", "dqn", "double_dqn", "dqn_shaped", "dqn_features13", final_method]
    rows = [next(r for r in summary_rows if r["method"] == m) for m in order if any(r["method"] == m for r in summary_rows)]
    short_labels_left = {
        "random": "随机策略",
        "heuristic": "人工启发式",
        "dqn": "3维 DQN",
        "double_dqn": "Double DQN",
        "dqn_shaped": "奖励塑形",
    }
    short_labels_right = {
        "random": "随机",
        "heuristic": "启发式",
        "dqn": "3维\nDQN",
        "double_dqn": "Double\nDQN",
        "dqn_shaped": "奖励\n塑形",
        "dqn_features13": "13维\nDQN",
        final_method: "最终方法\n(前瞻)",
    }

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14.0, 6.8), gridspec_kw={"width_ratios": [1.0, 1.28]}
    )
    fig.subplots_adjust(left=0.075, right=0.98, top=0.76, bottom=0.17, wspace=0.22)

    # 左图：前 5 组低分段放大（0–5,000 分）
    left_rows = rows[:5]
    pos_l = np.arange(len(left_rows))
    scores_l = [float(r["mean_classic_score"]) for r in left_rows]
    lines_l = [float(r["mean_lines"]) for r in left_rows]
    colors_l = [COLORS[r["method"]] for r in left_rows]
    bars_l = ax_left.bar(pos_l, scores_l, width=0.56, color=colors_l, zorder=3)
    ymax_l = 5000
    ax_left.set_ylim(0, ymax_l)
    ax_left.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, score, line_val in zip(bars_l, scores_l, lines_l):
        top_y = bar.get_height() + ymax_l * 0.025
        ax_left.text(bar.get_x() + bar.get_width() / 2, top_y,
                     f"{score:,.0f}\n({line_val:.1f}行)",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax_left.set_xticks(pos_l, [short_labels_left[r["method"]] for r in left_rows], fontsize=10)
    ax_left.set_ylabel("平均经典得分（40/100/300/1200）", fontsize=11)
    ax_left.set_title("① 前 5 组局部放大（0–5,000 分）", fontsize=12.5, pad=12, fontweight="bold")
    ax_left.grid(axis="y", alpha=0.20)

    # 右图：全部 7 组完整量程对比（突显 13 维与最终方法）
    pos_r = np.arange(len(rows))
    scores_r = [float(r["mean_classic_score"]) for r in rows]
    lines_r = [float(r["mean_lines"]) for r in rows]
    colors_r = [COLORS[r["method"]] for r in rows]
    bars_r = ax_right.bar(pos_r, scores_r, width=0.58, color=colors_r, zorder=3)
    ax_right.axhline(48000, color="#888888", linestyle="--", linewidth=1.2, zorder=2,
                     label="全部单消 1200 行得分线（48,000）")
    ax_right.axvspan(-0.45, 4.45, color="#f2f4f7", zorder=1)
    ax_right.text(2.0, 23000, "← 左侧放大区间\n(均 < 5,000 分)", ha="center", va="center",
                  fontsize=9.5, color="#555555")
    ymax_r = max(scores_r) * 1.18
    ax_right.set_ylim(0, ymax_r)
    ax_right.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    for idx in (5, 6):
        bar = bars_r[idx]
        top_y = bar.get_height() + ymax_r * 0.02
        ax_right.text(bar.get_x() + bar.get_width() / 2, top_y,
                      f"{scores_r[idx]:,.0f}\n({lines_r[idx]:.1f}行)",
                      ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_right.set_xticks(pos_r, [short_labels_right[r["method"]] for r in rows], fontsize=9.8)
    ax_right.set_title("② 全部 7 组全局对比（含 13 维与最终方法）", fontsize=12.5, pad=12, fontweight="bold")
    ax_right.grid(axis="y", alpha=0.20)
    ax_right.legend(loc="upper left", frameon=False, fontsize=10)

    fig.suptitle("相同 3000 步上限下，各方法的经典得分对比", fontsize=20, y=0.955)
    fig.text(0.5, 0.875, "左图放大看清前 5 组差异，右图展示 13 维与最终方法的真实量级跃升（括号内为平均消行数）",
             ha="center", fontsize=11)
    fig.text(0.075, 0.035, "注：学习方法取 5 个训练种子的测试均值，多种子波动分布见《种子稳定性分布图》（seed_classic_scores）。",
             fontsize=10, color="#555555")
    save_fig(fig, figures_dir, "classic_score_comparison")


def plot_clear_composition(summary_rows, final_method, figures_dir):
    order = ["heuristic", "dqn_shaped", "dqn_features13", final_method]
    rows = [next(r for r in summary_rows if r["method"] == m) for m in order if any(r["method"] == m for r in summary_rows)]
    fig, ax = plt.subplots(figsize=(12.2, 6.5))
    fig.subplots_adjust(left=0.23, right=0.95, top=0.74, bottom=0.16)

    clear_keys = ["mean_clears_1", "mean_clears_2", "mean_clears_3", "mean_clears_4"]
    clear_names = ["单消（1 行 · 40 分）", "双消（2 行 · 100 分）", "三消（3 行 · 300 分）", "四连消（4 行 · 1200 分）"]
    clear_colors = ["#9bb7d4", "#5c9ead", "#e29558", "#c83e37"]
    multipliers = np.array([1, 2, 3, 4], dtype=float)

    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    shares = []
    for r in rows:
        counts = np.array([float(r[k]) for k in clear_keys])
        cleared_lines_by_k = counts * multipliers
        total = max(cleared_lines_by_k.sum(), 1e-9)
        shares.append(cleared_lines_by_k / total * 100.0)
    shares = np.array(shares)

    for idx in range(4):
        vals = shares[:, idx]
        ax.barh(y, vals, left=left, height=0.52, color=clear_colors[idx], label=clear_names[idx], zorder=3)
        for row_idx, val in enumerate(vals):
            if val >= 6.0:
                ax.text(left[row_idx] + val / 2, row_idx, f"{val:.1f}%",
                        ha="center", va="center", color="white" if idx >= 2 else "#111111",
                        fontsize=10, fontweight="bold")
        left += vals

    ax.set_yticks(y, [LABELS.get(r["method"], r["method"]).replace("\n", " ") for r in rows], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("各连消类型占总消行数的比例（%）", fontsize=11.5)
    ax.grid(axis="x", alpha=0.20)
    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.835), ncol=4, frameon=False, fontsize=10)
    fig.suptitle("高分背后：谁学会了多连消？", fontsize=20, y=0.96)
    fig.text(0.5, 0.89, "对比各方法消掉的总行数中，1 / 2 / 3 / 4 行一次消除的占比结构",
             ha="center", fontsize=11.5)
    save_fig(fig, figures_dir, "clear_composition")


def plot_seed_scores_with_potential(seed_rows, summary_rows, final_method, figures_dir):
    order = ["dqn", "double_dqn", "dqn_shaped", "dqn_features13", final_method]
    methods = [m for m in order if any(r["method"] == m for r in seed_rows)]
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.83, bottom=0.18)

    for pos, method in enumerate(methods):
        selected = sorted([r for r in seed_rows if r["method"] == method],
                          key=lambda r: int(r["training_seed"]))
        offsets = np.linspace(-0.22, 0.22, len(selected))
        vals = [float(r["mean_classic_score"]) for r in selected]
        color = COLORS.get(method, "#333333")
        for off, val in zip(offsets, vals):
            ax.scatter(pos + off, val, s=90, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        mean_val = float(np.mean(vals))
        ax.hlines(mean_val, pos - 0.30, pos + 0.30, color="#222222", linewidth=1.6, zorder=4)
        ax.annotate(f"均值 {mean_val:,.0f}", (pos, max(vals)), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=11)

    pot = next((r for r in summary_rows if r["method"] == "potential"), None)
    heu = next((r for r in summary_rows if r["method"] == "heuristic"), None)
    handles = [Line2D([], [], color="#222222", linewidth=1.6, label="五份训练种子的均值")]
    if heu:
        val = float(heu["mean_classic_score"])
        ax.axhline(val, color=COLORS["heuristic"], linestyle="--", linewidth=1.5)
        handles.append(Line2D([], [], color=COLORS["heuristic"], linestyle="--",
                              label=f"人工启发式 ({val:,.0f} 分)"))
    if pot:
        val = float(pot["mean_classic_score"])
        ax.axhline(val, color="#6c5ce7", linestyle=":", linewidth=1.8)
        handles.append(Line2D([], [], color="#6c5ce7", linestyle=":",
                              label=f"纯势函数基线 V≡0 ({val:,.0f} 分)"))

    from matplotlib.ticker import FuncFormatter
    ax.set_xticks(range(len(methods)), [LABELS.get(m, m) for m in methods], fontsize=11)
    ax.set_yscale("log")
    all_vals = [float(r["mean_classic_score"]) for r in seed_rows if r["method"] in methods]
    ax.set_ylim(min(all_vals) * 0.6, max(all_vals) * 2.6)
    ax.set_yticks([500, 2000, 5000, 20000, 60000, 150000])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylabel("平均经典得分（对数刻度）", fontsize=11.5)
    ax.grid(axis="y", alpha=0.20)
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10.5)
    fig.suptitle("五个训练种子的经典得分稳定性与纯势函数拆解", fontsize=20, y=0.96)
    fig.text(0.5, 0.89, "每点为一份模型在相同 100 局测试集上的平均经典得分（对数刻度）",
             ha="center", fontsize=11.5)
    save_fig(fig, figures_dir, "seed_classic_scores")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results/v2")
    parser.add_argument("--final-method", default="final_classic_double_lookahead")
    args = parser.parse_args()

    summary_rows = read_csv(args.results / "summary.csv")
    seed_rows = read_csv(args.results / "training_seed_scores.csv")
    figures_dir = args.results / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }):
        plot_classic_comparison(summary_rows, args.final_method, figures_dir)
        plot_clear_composition(summary_rows, args.final_method, figures_dir)
        plot_seed_scores_with_potential(seed_rows, summary_rows, args.final_method, figures_dir)
    print(f"Saved v2 figures to {figures_dir}")


if __name__ == "__main__":
    main()
