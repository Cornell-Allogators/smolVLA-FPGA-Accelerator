"""
Test for MLP - INT8 MLP Kernel Verification with HLS Emulation

Compares Allo mlp_dataflow against PyTorch reference.
Supports Vitis HLS emulation modes: sw_emu, hw_emu, and hw.
"""

import allo
import numpy as np
import torch
import torch.nn.functional as F
from allo.ir.types import int8, int32
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mlp import mlp_dataflow


def pytorch_int8_mlp(
    X: np.ndarray,      # [L, D] int8
    W_1: np.ndarray,    # [D, 4*D] int8
    B_1: np.ndarray,    # [4*D] int8
    W_2: np.ndarray,    # [4*D, D] int8
    B_2: np.ndarray,    # [D] int8
) -> tuple[np.ndarray, dict]:
    """
    PyTorch reference for INT8 MLP (FC1 -> GELU -> FC2).
    
    Matches the numerical flow of mlp_dataflow:
    1. FC1: X @ W_1 + B_1 (int32 accumulator)
    2. GELU approximation (tanh-based)
    3. FC2: GELU_out @ W_2 + B_2 (int32 accumulator)
    """
    # Convert to torch tensors (int32 for matmul precision)
    X_t = torch.from_numpy(X.astype(np.int32))
    W_1_t = torch.from_numpy(W_1.astype(np.int32))
    B_1_t = torch.from_numpy(B_1.astype(np.int32))
    W_2_t = torch.from_numpy(W_2.astype(np.int32))
    B_2_t = torch.from_numpy(B_2.astype(np.int32))
    
    # FC1: [L, D] @ [D, 4*D] -> [L, 4*D]
    FC1_acc = torch.matmul(X_t, W_1_t)
    FC1_out = FC1_acc + B_1_t.unsqueeze(0)  # Add bias
    
    # GELU approximation (matching mlp.py)
    # GELU(x) ≈ 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    x = FC1_out.float()
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    FC1_act_float = 0.5 * x * (1.0 + torch.tanh(inner))
    
    # Cast GELU output back to int32 for FC2
    FC1_act = FC1_act_float.to(torch.int32)
    
    # FC2: [L, 4*D] @ [4*D, D] -> [L, D]
    FC2_acc = torch.matmul(FC1_act, W_2_t)
    FC2_out = FC2_acc + B_2_t.unsqueeze(0)  # Add bias
    
    intermediates = {
        'FC1_acc': FC1_acc.numpy(),
        'FC1_out': FC1_out.numpy(),
        'FC1_act': FC1_act.numpy(),
        'FC2_acc': FC2_acc.numpy(),
        'FC2_out': FC2_out.numpy(),
    }
    
    # Return FC2_out as int32, NOT clamped to int8
    # (The int8 clamping would destroy the values due to overflow)
    return FC2_out.numpy(), intermediates


def test_mlp_hls(allo_dtype=int8, mode="sw_emu", reduced=False):
    """Test MLP with Vitis HLS (sw_emu, hw_emu, or hw)."""
    mode_names = {
        "sw_emu": "Software Emulation",
        "hw_emu": "Hardware Emulation", 
        "hw": "Hardware Synthesis"
    }
    
    print("\n" + "=" * 80)
    print(f"MLP Test: Allo ({mode_names[mode]}) vs PyTorch | dtype={allo_dtype}")
    print("=" * 80)
    
    # Dimension selection based on reduced flag
    if reduced:
        L, D = 16, 128  # Reduced: 16 tokens, 128 features
        print("  Using REDUCED tensor sizes for faster emulation")
    else:
        L, D = 32, 256  # Default: 32 tokens, 256 features
    
    np.random.seed(42)
    X = np.random.randint(-8, 8, size=(L, D), dtype=np.int8)
    W_1 = np.random.randint(-4, 4, size=(D, 4*D), dtype=np.int8)
    B_1 = np.random.randint(-2, 2, size=(4*D,), dtype=np.int8)
    W_2 = np.random.randint(-4, 4, size=(4*D, D), dtype=np.int8)
    B_2 = np.random.randint(-2, 2, size=(D,), dtype=np.int8)
    
    print(f"Config: L={L}, D={D}")
    print(f"  Input size: {X.nbytes / 1024:.1f} KB")
    print(f"  Weight sizes: W_1={W_1.nbytes / 1024:.1f} KB, W_2={W_2.nbytes / 1024:.1f} KB")
    
    # Time estimates
    time_estimates = {
        "sw_emu": "~1 minute",
        "hw_emu": "~5-15 minutes (includes HLS synthesis)",
        "hw": "~2-6 hours (includes P&R and bitstream generation)"
    }
    print(f"  Estimated time: {time_estimates[mode]}")
    
    # PyTorch reference
    print("\nPyTorch reference...")
    pytorch_out, intermediates = pytorch_int8_mlp(X, W_1, B_1, W_2, B_2)
    print(f"  Output range: [{pytorch_out.min()}, {pytorch_out.max()}]")
    
    # Allo implementation (HLS)
    print(f"\nAllo mlp_dataflow ({mode_names[mode]})...")
    print(f"  Building HLS project in '{mode}' mode...")
    s = allo.customize(mlp_dataflow, instantiate=[allo_dtype, D, L])
    
    try:
        mod = s.build(
            target="vitis_hls",
            mode=mode,
            project=f"mlp_test_{mode}_L{L}_D{D}.prj"
        )
        
        print(f"  Running {mode_names[mode].lower()}...")
        allo_out = np.zeros((L, D), dtype=np.int32)
        mod(X, W_1, B_1, W_2, B_2, allo_out)
        print(f"  Output range: [{allo_out.min()}, {allo_out.max()}]")
        
        # Compare
        diff = np.abs(pytorch_out.astype(np.int64) - allo_out.astype(np.int64))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        matches = np.sum(pytorch_out == allo_out)
        
        print(f"\nComparison:")
        print(f"  Max diff: {max_diff}, Mean diff: {mean_diff:.2f}")
        print(f"  Exact matches: {matches}/{pytorch_out.size} ({100*matches/pytorch_out.size:.1f}%)")
        
        print("\nSample (rows 0-3, cols 0-4):")
        print("  PyTorch:", pytorch_out[:4, :4])
        print("  Allo:   ", allo_out[:4, :4])
        
        tolerance = 15
        success = max_diff <= tolerance
        print(f"\n{'✓ PASSED' if success else '✗ FAILED'} (tolerance={tolerance})")
        
        return success
    
    except Exception as e:
        print(f"\n✗ {mode.upper()} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run MLP tests based on configuration."""
    # ============ CONFIGURATION ============
    TEST_TYPE = "hw_emu"     # Options: "sw_emu", "hw_emu", "hw"
    DTYPE = int8             # Data type for test
    REDUCED_SIZE = True     # If True, use smaller tensors (L=16, D=128) for faster emulation
    # =======================================
    
    print("  MLP VERIFICATION TEST")
    print(f"\nTest Type: {TEST_TYPE}")
    print(f"Data Type: {DTYPE}")
    print(f"Reduced Size: {REDUCED_SIZE}")
    print()
    
    try:
        if TEST_TYPE in ["sw_emu", "hw_emu", "hw"]:
            success = test_mlp_hls(allo_dtype=DTYPE, mode=TEST_TYPE, reduced=REDUCED_SIZE)
        else:
            print(f"✗ Unknown test type: {TEST_TYPE}")
            print("   Valid options: 'sw_emu', 'hw_emu', 'hw'")
            return False
        
        if success:
            print("\n✓ TEST PASSED")
        else:
            print("\n✗ TEST FAILED")
        
        return success
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

