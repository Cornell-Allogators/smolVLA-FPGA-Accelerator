import allo, math
from allo.ir.types import int8, int16, float32, bfloat16

import torch
import torch.nn as nn
import math

# Simple MLP model

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=3072, output_dim=768, activation=None):
        super().__init__()
        # allow passing None to use a default GELU activation
        if activation is None:
            activation = nn.GELU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = activation
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, S, D) -> treat last dim as feature dimension
        return self.fc2(self.act(self.fc1(x)))
    
model = SimpleMLP()
model.eval()

example_inputs = [torch.rand(1, 3, 768)]
# Generate the LLVM backend but keep weights as explicit arguments so the
# HLS generator will emit them as pointer arguments (so they can be streamed
# in over AXI) instead of embedding large static arrays into the C++.
llvm_mod = allo.frontend.from_pytorch(
    model, example_inputs=example_inputs, weights_as_args=True
)

golden = model(*example_inputs)
np_inputs = [x.detach().numpy() for x in example_inputs]
res = llvm_mod(*np_inputs)
# Allow small numerical differences between the PyTorch reference and the
# generated backend (float32 rounding/ordering). Relax tolerances accordingly.
torch.testing.assert_close(
    res, golden.detach().numpy(), rtol=1e-2, atol=1e-2
)
print("Passed!")

# Generate VHLS code with weights passed as function arguments. The
# backend postprocessing will add #pragma HLS interface m_axi for these
# pointer arguments so the host can stream weights over AXI.
mod = allo.frontend.from_pytorch(
    model, example_inputs=example_inputs, target="vhls", weights_as_args=True
)

print("Done")