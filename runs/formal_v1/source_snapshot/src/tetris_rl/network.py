"""A shared Q scorer for candidate-placement representations."""

from torch import nn


class QNetwork(nn.Module):
    def __init__(self, input_dim, gamma=0.95, known_reward=True):
        super().__init__()
        self.gamma = gamma
        self.known_reward = known_reward
        value_input_dim = input_dim - 2 if known_reward else input_dim
        self.layers = nn.Sequential(
            nn.Linear(value_input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, features):
        # Legacy diagnostic checkpoints used a direct candidate-Q network.
        if not self.known_reward:
            return self.layers(features).squeeze(-1)
        continuation = self.layers(features[..., :-2]).squeeze(-1)
        continuation = continuation.masked_fill(features[..., -1] > 0.5, 0.0)
        return features[..., -2] + self.gamma * continuation
