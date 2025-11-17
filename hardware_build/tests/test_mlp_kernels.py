"""Pytest for comparing PyTorch MLP and Allo kernel MLP outputs."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import allo
from common_kernels.kernels import gemm, add_bias, gelu_approx
from mlp.mlp_pytorch import SimpleMLP


# Linear layer shapes from smolVLA model (in_dim, out_dim) tuples
LINEAR_LAYER_SHAPES = [
    # Top-level action/state projections
    (32, 720),      # top-level: action_in_proj
    (720, 32),      # top-level: action_out_proj
    (1440, 720),    # top-level: action_time_mlp_in
    (720, 720),     # top-level: action_time_mlp_out
    (32, 960),      # top-level: state_proj
    
    # Text model language expert (lm_expert) - 16 layers each
    (720, 2048),    # lm_expert (16x): gate_proj / up_proj
    (2048, 720),    # lm_expert (16x): down_proj
    (720, 960),     # lm_expert (16x): q_proj
    (720, 320),     # lm_expert (16x): k_proj
    (720, 320),     # lm_expert (16x): v_proj
    (960, 720),     # lm_expert (16x): o_proj
    
    # VLM text model - 16 layers each
    (960, 2560),    # vlm.text_model (16x): gate_proj / up_proj
    (2560, 960),    # vlm.text_model (16x): down_proj
    (960, 960),     # vlm.text_model (16x): q_proj
    (960, 320),     # vlm.text_model (16x): k_proj
    (960, 320),     # vlm.text_model (16x): v_proj
    
    # VLM vision model - 12 layers each
    (768, 3072),    # vlm.vision_model (12x): fc1 (MLP)
    (3072, 768),    # vlm.vision_model (12x): fc2 (MLP)
    (768, 768),     # vlm.vision_model (12x): q_proj / k_proj / v_proj / out_proj
    
    # Language head
    (960, 49280),   # vlm: lm_head
]


class TestMLPKernels:
    """Test Allo MLP kernels against PyTorch reference."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set seeds and create model/inputs."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # PyTorch MLP
        self.model = SimpleMLP(input_dim=768, hidden_dim=3072, output_dim=768)
        self.model.eval()
        
        # Test input: (1, 3, 768) - batch=1, seq=3, dim=768
        self.test_input = torch.randn(1, 3, 768, dtype=torch.float32)
        
        # Flatten for Allo (Allo expects 2D: 3x768)
        self.flat_input = self.test_input.squeeze(0)  # (3, 768)

    def test_pytorch_mlp_forward(self):
        """Verify PyTorch MLP runs without error."""
        with torch.no_grad():
            output = self.model(self.test_input)
        assert output.shape == (1, 3, 768)
        assert not torch.isnan(output).any(), "PyTorch MLP produced NaN"

    def test_allo_mlp_module_creation(self):
        """Verify Allo MLP module compiles and prints."""
        # Import float32 outside the Allo function (Allo doesn't support imports inside)
        from allo.ir.types import float32
        
        # Constants for Allo
        M, D_in, H, D_out = 3, 768, 3072, 768
        
        # Define the top-level MLP in Allo (matches mlp.py)
        def mlp_top(A: "float32[M, D_in]", 
                    W1: "float32[D_in, H]", 
                    b1: "float32[H]", 
                    W2: "float32[H, D_out]", 
                    b2: "float32[D_out]") -> "float32[M, D_out]":
            C1 = gemm[M, D_in, H](A, W1)
            C1b = add_bias[M, H](C1, b1)
            A1 = gelu_approx[M, H](C1b)
            C2 = gemm[M, H, D_out](A1, W2)
            Out: float32[M, D_out] = 0
            for i, j in allo.grid(M, D_out):
                Out[i, j] = C2[i, j] + b2[j]
            return Out
        
        # Customize and compose
        s_gemm1 = allo.customize(gemm, instantiate=[M, D_in, H])
        s_gemm1.pipeline("j")
        
        s_gemm2 = allo.customize(gemm, instantiate=[M, H, D_out])
        s_gemm2.reorder("k", "j")
        s_gemm2.pipeline("j")
        
        s_act = allo.customize(gelu_approx, instantiate=[M, H])
        s_act.pipeline("j")
        
        s = allo.customize(mlp_top)
        s.compose([s_gemm1, s_gemm2, s_act])
        
        # Module should exist and be printable
        assert s.module is not None
        module_str = str(s.module)
        assert "gemm" in module_str
        assert "gelu_approx" in module_str
        assert "mlp_top" in module_str

    def test_allo_gemm_kernel(self):
        """Test individual GEMM kernel against matmul."""
        M, K, N = 3, 768, 3072
        
        # Random inputs (ensure C-contiguous)
        A = np.ascontiguousarray(np.random.randn(M, K).astype(np.float32))
        B = np.ascontiguousarray(np.random.randn(K, N).astype(np.float32))
        
        # NumPy reference (C = A @ B)
        expected = A @ B
        
        # Allo GEMM
        s = allo.customize(gemm, instantiate=[M, K, N])
        # Build and execute
        gemm_func = s.build(target="llvm")
        result = gemm_func(A, B)
        
        # Compare with relaxed tolerance for float32 rounding
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4,
                                   err_msg="GEMM kernel output mismatch")

    def test_allo_add_bias_kernel(self):
        """Test add_bias kernel."""
        M, H = 3, 3072
        
        X = np.random.randn(M, H).astype(np.float32)
        b = np.random.randn(H).astype(np.float32)
        
        # NumPy reference: X + b (broadcast)
        expected = X + b[np.newaxis, :]
        
        # Allo add_bias
        s = allo.customize(add_bias, instantiate=[M, H])
        add_bias_func = s.build(target="llvm")
        result = add_bias_func(X, b)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5,
                                   err_msg="add_bias kernel output mismatch")

    def test_allo_gelu_approx_kernel(self):
        """Test approximate GELU activation."""
        M, H = 3, 3072
        
        X = np.random.randn(M, H).astype(np.float32)
        
        # NumPy reference: tanh-based GELU approximation
        a = np.sqrt(2 / np.pi)
        expected = 0.5 * X * (1 + np.tanh(a * (X + 0.044715 * X**3)))
        
        # Allo gelu_approx
        s = allo.customize(gelu_approx, instantiate=[M, H])
        gelu_func = s.build(target="llvm")
        result = gelu_func(X)
        
        # GELU is smooth, so allow slightly larger tolerance
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4,
                                   err_msg="gelu_approx kernel output mismatch")

    def test_mlp_end_to_end_numerical(self):
        """End-to-end test: compare PyTorch and Allo MLP on same weights."""
        M, D_in, H, D_out = 3, 768, 3072, 768
        
        # Extract PyTorch weights and convert to NumPy (ensure C-contiguous)
        W1 = np.ascontiguousarray(self.model.fc1.weight.data.T.numpy())  # (D_in, H)
        b1 = np.ascontiguousarray(self.model.fc1.bias.data.numpy())       # (H,)
        W2 = np.ascontiguousarray(self.model.fc2.weight.data.T.numpy())  # (H, D_out)
        b2 = np.ascontiguousarray(self.model.fc2.bias.data.numpy())       # (D_out,)
        
        # Flatten input for Allo (ensure C-contiguous)
        A_allo = np.ascontiguousarray(self.flat_input.numpy())  # (3, 768)
        
        # PyTorch forward pass
        with torch.no_grad():
            pytorch_out = self.model(self.test_input).squeeze(0).numpy()  # (3, 768)
        
        # Allo forward pass (step-by-step)
        # FC1: (3, 768) @ (768, 3072) -> (3, 3072)
        s_gemm1 = allo.customize(gemm, instantiate=[M, D_in, H])
        gemm1_func = s_gemm1.build(target="llvm")
        C1 = gemm1_func(A_allo, W1)  # (3, 3072)
        C1 = np.ascontiguousarray(C1)
        
        # Add bias: (3, 3072) + (3072,) -> (3, 3072)
        s_bias1 = allo.customize(add_bias, instantiate=[M, H])
        bias1_func = s_bias1.build(target="llvm")
        C1b = bias1_func(C1, b1)  # (3, 3072)
        C1b = np.ascontiguousarray(C1b)
        
        # GELU: (3, 3072) -> (3, 3072)
        s_gelu = allo.customize(gelu_approx, instantiate=[M, H])
        gelu_func = s_gelu.build(target="llvm")
        A1 = gelu_func(C1b)  # (3, 3072)
        A1 = np.ascontiguousarray(A1)
        
        # FC2: (3, 3072) @ (3072, 768) -> (3, 768)
        s_gemm2 = allo.customize(gemm, instantiate=[M, H, D_out])
        gemm2_func = s_gemm2.build(target="llvm")
        C2 = gemm2_func(A1, W2)  # (3, 768)
        C2 = np.ascontiguousarray(C2)
        
        # Add bias: (3, 768) + (768,) -> (3, 768)
        s_bias2 = allo.customize(add_bias, instantiate=[M, D_out])
        bias2_func = s_bias2.build(target="llvm")
        allo_out = bias2_func(C2, b2)  # (3, 768)
        
        # Compare with relaxed tolerances
        # (PyTorch uses different GELU, different op ordering -> larger diff)
        np.testing.assert_allclose(allo_out, pytorch_out, rtol=1e-2, atol=1e-2,
                                   err_msg="MLP end-to-end output mismatch")


    @pytest.mark.parametrize("shape", LINEAR_LAYER_SHAPES)
    def test_linear_layer_shapes(self, shape):
        """Test GEMM kernel for all model linear layer shapes."""
        in_dim, out_dim = shape
        
        # Use batch size of 3 for sequence dimension
        batch_size = 3
        
        # Random inputs (ensure C-contiguous)
        A = np.ascontiguousarray(np.random.randn(batch_size, in_dim).astype(np.float32))
        B = np.ascontiguousarray(np.random.randn(in_dim, out_dim).astype(np.float32))
        
        # NumPy reference (C = A @ B)
        expected = A @ B
        
        # Allo GEMM
        s = allo.customize(gemm, instantiate=[batch_size, in_dim, out_dim])
        gemm_func = s.build(target="llvm")
        result = gemm_func(A, B)
        
        # Compare with relaxed tolerance for float32 rounding
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4,
                                   err_msg=f"GEMM mismatch for shape [{in_dim}, {out_dim}]")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
