"""Vectorized, float32 board statistics without uint8 arithmetic overflow."""

import numpy as np


def board_statistics(boards):
    """Return column heights, maximum height, holes and bumpiness."""
    occupied = np.asarray(boards) != 0
    height = occupied.shape[-2]
    heights = np.where(occupied.any(axis=-2), height - occupied.argmax(axis=-2), 0)
    holes = heights.sum(axis=-1) - occupied.sum(axis=(-2, -1))
    bumpiness = np.abs(np.diff(heights, axis=-1)).sum(axis=-1)
    return heights, heights.max(axis=-1), holes, bumpiness


def candidate_features(boards, learning_rewards, terminated, board_dim):
    """3/13 board inputs plus known scaled reward and terminal metadata."""
    heights, maximum, holes, bumpiness = board_statistics(boards)
    # A common divisor keeps small, strategically important hole counts visible.
    summary = np.stack((maximum / 10.0, holes / 10.0, bumpiness / 10.0), axis=-1)
    if board_dim == 13:
        summary = np.concatenate((heights / 10.0, summary), axis=-1)
    elif board_dim != 3:
        raise ValueError("board_dim must be 3 or 13 for the 10-column experiments")
    metadata = np.stack((learning_rewards, terminated.astype(float)), axis=-1)
    return np.concatenate((summary, metadata), axis=-1).astype(np.float32)


def potential(boards, maximum_weight, holes_weight):
    _, maximum, holes, _ = board_statistics(boards)
    return -(maximum_weight * maximum + holes_weight * holes)
