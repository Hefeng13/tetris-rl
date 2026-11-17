"""Render fixed-seed gameplay from completed experiments to GIF and MP4."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from tetris_rl.agents.dqn_agent import DQNAgent
from tetris_rl.agents.heuristic_agent import HeuristicAgent
from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.config import load_config, make_env, write_json


def draw_frame(env, title, seed, lines, finished):
    tile = 20
    image = Image.new("RGB", (240, 470), "#14202e")
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), title, fill="white")
    draw.text((12, 28), f"seed {seed} | moves {env.pieces} | lines {lines}", fill="#b8cad9")
    palette = {piece.id: tuple(piece.color_rgb) for piece in env.engine.tetrominoes}
    board = env.board
    for row in range(env.height):
        for column in range(env.width):
            color = palette.get(int(board[row, column]), (31, 47, 64))
            x, y = 20 + column * tile, 55 + row * tile
            draw.rectangle((x, y, x + tile - 2, y + tile - 2), fill=color)
    if finished:
        draw.text((12, 456), "Game ended or demo cap reached", fill="#f3bb74")
    return image


def render_policy(agent, config, board_dim, seed, name, output, piece_cap):
    env = make_env(config, board_dim, piece_cap)
    candidates = env.reset(seed)
    frames = [draw_frame(env, name, seed, 0, False)]
    lines = 0
    while True:
        candidates, _, terminated, truncated, info = env.step(agent.act(candidates))
        lines += int(info["lines_cleared"])
        frames.append(draw_frame(env, name, seed, lines, terminated or truncated))
        if terminated or truncated:
            break
    frames[0].save(output / f"{name}.gif", save_all=True, append_images=frames[1:],
                   duration=100, loop=0, optimize=True)
    video_path = output / f"{name}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, frames[0].size)
    if not writer.isOpened():
        raise RuntimeError("MP4 writer could not be initialized")
    for frame in frames:
        writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    writer.release()
    check = cv2.VideoCapture(str(video_path))
    frame_count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    if frame_count != len(frames):
        raise RuntimeError("MP4 frame count failed verification")
    env.close()
    return {"policy": name, "seed": seed, "lines": lines, "pieces": env.pieces,
            "demo_piece_cap": piece_cap, "frames": len(frames)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=40000)
    parser.add_argument("--piece-cap", type=int, default=500)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.batch / "config.yaml")
    manifest = json.loads((args.batch / "batch.json").read_text(encoding="utf-8"))
    job = next(job for job in manifest["jobs"] if job["job"] == "dqn_seed0")
    dqn, metadata = DQNAgent.load(Path(job["output"]) / "best.pt")
    feature_job = next(job for job in manifest["jobs"] if job["job"] == "dqn_features13_seed0")
    feature_agent, feature_metadata = DQNAgent.load(Path(feature_job["output"]) / "best.pt")
    args.output.mkdir(parents=True, exist_ok=True)
    policies = [("random", RandomAgent(config["evaluation"]["random_baseline_policy_seed"] + args.seed), 3),
                ("heuristic", HeuristicAgent(**config["heuristic"]), 3),
                ("dqn_seed0", dqn, metadata["board_feature_dim"]),
                ("features13_seed0", feature_agent, feature_metadata["board_feature_dim"])]
    rows = [render_policy(agent, config, dimension, args.seed, name, args.output, args.piece_cap)
            for name, agent, dimension in policies]
    write_json(args.output / "demo_metadata.json", rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
