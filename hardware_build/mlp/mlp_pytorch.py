"""PyTorch SimpleMLP model for reference/validation."""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simple MLP: input -> FC1 -> GELU -> FC2 -> output."""
    
    def __init__(self, input_dim=768, hidden_dim=3072, output_dim=768, activation=None):
        super().__init__()
        if activation is None:
            activation = nn.GELU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = activation
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, S, D) -> treat last dim as feature dimension
        return self.fc2(self.act(self.fc1(x)))
