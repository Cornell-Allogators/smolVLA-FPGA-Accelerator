"""Shared pytest fixtures for hardware_build tests."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add hardware_build to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def mlp_model():
    """PyTorch SimpleMLP model for testing."""
    from mlp.mlp_pytorch import SimpleMLP
    model = SimpleMLP(input_dim=768, hidden_dim=3072, output_dim=768)
    model.eval()
    return model


@pytest.fixture
def test_input():
    """Random test input tensor (batch=1, seq=3, dim=768)."""
    return torch.randn(1, 3, 768, dtype=torch.float32)


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42
