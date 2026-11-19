"""Render v2 gameplay demos with live classic Tetris scores, individual & multi-panel comparisons."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from tetris_rl.agents.dqn_agent import DQNAgent
from tetris_rl.agents.heuristic_agent import HeuristicAgent
from tetris_rl.agents.lookahead import LookaheadAgent
from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.config import ROOT, load_config, make_env, resolve_run_directory, write_json
from tetris_rl.evaluate import CLASSIC_LINE_SCORES


def draw_panel(board, palette, height, width, title, seed, pieces, lines, classic_score, clears, finished, tile=20):
    board_w = width * tile
    panel_w = max(board_w + 36, 208)
    panel_h = height * tile + 95
    x0 = (panel_w - board_w) // 2
    image = Image.new("RGB", (panel_w, panel_h), "#14202e")
    draw = ImageDraw.Draw(image)
    draw.text((10, 8), title, fill="white")
    draw.text((10, 26), f"seed {seed} | mv {pieces} | ln {lines}", fill="#b8cad9")
    draw.text((10, 42), f"score {classic_score:,} | {clears[1]}/{clears[2]}/{clears[3]}/{clears[4]}",
              fill="#f3bb74")
    for row in range(height):
        for column in range(width):
            color = palette.get(int(board[row, column]), (31, 47, 64))
            x, y = x0 + column * tile, 66 + row * tile
            draw.rectangle((x, y, x + tile - 2, y + tile - 2), fill=color)
    if finished:
        draw.text((10, panel_h - 20), "Ended / Cap", fill="#f3bb74")
    return image


def rollout_frames(agent, config, board_dim, seed, title, piece_cap, shaping=False, lookahead=0, exp_cfg=None, tile=20):
    env = make_env(config, board_dim, piece_cap, shaping_enabled=shaping, experiment=exp_cfg)
    decision_agent = LookaheadAgent(agent, env) if lookahead == 1 else agent
    candidates = env.reset(seed)
    palette = {piece.id: tuple(piece.color_rgb) for piece in env.engine.tetrominoes}
    lines, classic_score = 0, 0
    clears = [0, 0, 0, 0, 0]
    frames = [draw_panel(env.board, palette, env.height, env.width, title, seed,
                         env.pieces, lines, classic_score, clears, False, tile=tile)]
    while True:
        candidates, _, terminated, truncated, info = env.step(decision_agent.act(candidates))
        cleared = int(info["lines_cleared"])
        lines += cleared
        clears[cleared] += 1
        classic_score += CLASSIC_LINE_SCORES[cleared]
        frames.append(draw_panel(env.board, palette, env.height, env.width, title, seed,
                                 env.pieces, lines, classic_score, clears, terminated or truncated, tile=tile))
        if terminated or truncated:
            break
    meta = {"policy": title, "seed": seed, "lines": lines, "classic_score": classic_score,
            "clears": {"1": clears[1], "2": clears[2], "3": clears[3], "4": clears[4]},
            "pieces": env.pieces, "frames": len(frames)}
    env.close()
    return frames, meta


def save_video(frames, stem):
    frames[0].save(f"{stem}.gif", save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    video_path = f"{stem}.mp4"
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, frames[0].size)
    if not writer.isOpened():
        raise RuntimeError("MP4 writer could not be initialized")
    for frame in frames:
        writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    writer.release()


def combine_side_by_side(frame_lists, stem, gap=8):
    max_len = max(len(fl) for fl in frame_lists)
    w = sum(fl[0].width for fl in frame_lists) + gap * (len(frame_lists) - 1)
    h = frame_lists[0][0].height
    combined = []
    for idx in range(max_len):
        canvas = Image.new("RGB", (w, h), "#0d1620")
        x = 0
        for fl in frame_lists:
            panel = fl[min(idx, len(fl) - 1)]
            canvas.paste(panel, (x, 0))
            x += panel.width + gap
        combined.append(canvas)
    save_video(combined, stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, default=ROOT / "runs/formal_v1")
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--final-lookahead", type=int, choices=[0, 1], default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "results/v2/videos")
    parser.add_argument("--seed", type=int, default=40000)
    parser.add_argument("--piece-cap", type=int, default=500)
    args = parser.parse_args()

    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.batch / "config.yaml")
    manifest = json.loads((args.batch / "batch.json").read_text(encoding="utf-8"))

    def load_formal(job_name):
        job = next(j for j in manifest["jobs"] if j["job"] == job_name)
        return DQNAgent.load(resolve_run_directory(args.batch, job["output"]) / "best.pt")

    dqn3, _ = load_formal("dqn_seed0")
    ddqn3, _ = load_formal("double_dqn_seed0")
    shaped3, _ = load_formal("dqn_shaped_seed0")
    f13, _ = load_formal("dqn_features13_seed0")
    final_agent, final_meta = DQNAgent.load(args.final_checkpoint)

    specs = [
        ("random_score", "Random", RandomAgent(config["evaluation"]["random_baseline_policy_seed"] + args.seed), 3, False, 0, None),
        ("heuristic_score", "Heuristic", HeuristicAgent(**config["heuristic"]), 3, False, 0, None),
        ("dqn_seed0_score", "3D DQN", dqn3, 3, False, 0, None),
        ("double_dqn_seed0_score", "Double DQN", ddqn3, 3, False, 0, None),
        ("dqn_shaped_seed0_score", "Shaped DQN", shaped3, 3, True, 0, None),
        ("features13_seed0_score", "13D DQN (Baseline)", f13, 13, False, 0, None),
        ("final_method_seed0", "Final Method (13D+Double+LA1)" if args.final_lookahead else "Final Method",
         final_agent, final_meta["board_feature_dim"], False, args.final_lookahead, final_meta.get("experiment_config")),
    ]

    all_frames = {}
    metas = []
    for fname, title, agent, dim, shaping, la, exp_cfg in specs:
        frames, meta = rollout_frames(agent, config, dim, args.seed, title, args.piece_cap,
                                      shaping=shaping, lookahead=la, exp_cfg=exp_cfg, tile=20)
        save_video(frames, str(args.output / fname))
        all_frames[fname] = frames
        metas.append(meta)

    combine_side_by_side(
        [all_frames["features13_seed0_score"], all_frames["final_method_seed0"]],
        str(args.output / "comparison_13d_vs_final"),
    )

    combine_side_by_side(
        [all_frames["random_score"], all_frames["heuristic_score"],
         all_frames["dqn_seed0_score"], all_frames["double_dqn_seed0_score"],
         all_frames["dqn_shaped_seed0_score"]],
        str(args.output / "comparison_first5"),
        gap=8,
    )

    all7_specs = [
        ("Random", RandomAgent(config["evaluation"]["random_baseline_policy_seed"] + args.seed), 3, False, 0, None),
        ("Heuristic", HeuristicAgent(**config["heuristic"]), 3, False, 0, None),
        ("3D DQN", dqn3, 3, False, 0, None),
        ("Double DQN", ddqn3, 3, False, 0, None),
        ("Shaped DQN", shaped3, 3, True, 0, None),
        ("13D DQN", f13, 13, False, 0, None),
        ("Final (13D+Double+LA1)" if args.final_lookahead else "Final",
         final_agent, final_meta["board_feature_dim"], False, args.final_lookahead, final_meta.get("experiment_config")),
    ]
    all7_frames = [
        rollout_frames(agent, config, dim, args.seed, title, args.piece_cap,
                       shaping=shaping, lookahead=la, exp_cfg=exp_cfg, tile=14)[0]
        for title, agent, dim, shaping, la, exp_cfg in all7_specs
    ]
    combine_side_by_side(all7_frames, str(args.output / "comparison_all7"), gap=4)

    write_json(args.output / "demo_metadata.json", metas)
    print(json.dumps(metas, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
