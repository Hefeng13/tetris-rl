"""Compare CPU and CUDA on the actual small candidate-scoring network."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from tetris_rl.network import QNetwork


def benchmark(device, repetitions):
    torch.manual_seed(7)
    network = QNetwork(15).to(device)
    target = QNetwork(15).to(device)
    target.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    rng = np.random.default_rng(7)
    candidates = rng.normal(size=(40, 15)).astype(np.float32)
    candidates[:, -1] = 0
    current = torch.from_numpy(np.repeat(candidates[:1], 64, axis=0))
    following = torch.from_numpy(np.repeat(candidates[None], 64, axis=0))
    reward = torch.ones(64)

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    def decision():
        with torch.no_grad():
            return int(network(torch.from_numpy(candidates).to(device)).argmax().item())

    def update():
        with torch.no_grad():
            expected = reward.to(device) + 0.95 * target(following.to(device)).max(dim=1).values
        predicted = network(current.to(device))
        loss = F.smooth_l1_loss(predicted, expected)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 10)
        optimizer.step()
        return float(loss.detach().cpu())

    for _ in range(20):
        decision()
        update()
    synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        decision()
    synchronize()
    decision_ms = (time.perf_counter() - started) * 1000 / repetitions
    synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        update()
    synchronize()
    update_ms = (time.perf_counter() - started) * 1000 / repetitions
    return {"device": device, "decision_ms": decision_ms, "update_ms": update_ms,
            "network_cost_ms_per_environment_step": decision_ms + update_ms / 4}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Use a CUDA-enabled PyTorch environment for the paired comparison")
    result = {"torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "repetitions": args.repetitions,
              "measurements": [benchmark(device, args.repetitions) for device in ("cpu", "cuda")]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
