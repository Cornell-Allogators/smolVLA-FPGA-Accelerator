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
    gamma: np.ndarray,  # [D] int8 - LayerNorm scale
    beta: np.ndarray,   # [D] int8 - LayerNorm bias
) -> tuple[np.ndarray, dict]:
    """
    PyTorch reference for INT8 MLP (FC1 -> GELU -> FC2 -> LayerNorm).
    
    Matches the numerical flow of mlp_dataflow:
    1. FC1: X @ W_1 + B_1 (int32 accumulator)
    2. GELU approximation (tanh-based)
    3. FC2: GELU_out @ W_2 + B_2 (int32 accumulator)
    4. LayerNorm: normalize and scale back to int8
    """
    # Convert to torch tensors (int32 for matmul precision)
    X_t = torch.from_numpy(X.astype(np.int32))
    W_1_t = torch.from_numpy(W_1.astype(np.int32))
    B_1_t = torch.from_numpy(B_1.astype(np.int32))
    W_2_t = torch.from_numpy(W_2.astype(np.int32))
    B_2_t = torch.from_numpy(B_2.astype(np.int32))
    gamma_t = torch.from_numpy(gamma.astype(np.float32))
    beta_t = torch.from_numpy(beta.astype(np.float32))
    
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
    FC2_out = FC2_acc + B_2_t.unsqueeze(0)  # Add bias (int32)
    
    # LayerNorm: convert int32 to normalized int8
    FC2_float = FC2_out.float()
    mean = FC2_float.mean(dim=-1, keepdim=True)
    var = FC2_float.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (FC2_float - mean) / torch.sqrt(var + 1e-8)
    scaled = normalized * gamma_t.unsqueeze(0) + beta_t.unsqueeze(0)
    FC2_ln = scaled.to(torch.int8)
    
    intermediates = {
        'FC1_acc': FC1_acc.numpy(),
        'FC1_out': FC1_out.numpy(),
        'FC1_act': FC1_act.numpy(),
        'FC2_acc': FC2_acc.numpy(),
        'FC2_out': FC2_out.numpy(),
        'FC2_ln': FC2_ln.numpy(),
    }
    
    # Return normalized int8 output
    return FC2_ln.numpy(), intermediates


def test_mlp_hls(allo_dtype=int8, mode="sw_emu", reduced=False, schedule_spec=None, L_override=None, D_override=None, project_name=None):
    """Test MLP with Vitis HLS (sw_emu, hw_emu, hw, or csyn)."""
    mode_names = {
        "sw_emu": "Software Emulation",
        "hw_emu": "Hardware Emulation", 
        "hw": "Hardware Synthesis",
        "csyn": "C Synthesis"
    }
    
    print("\n" + "=" * 80)
    print(f"MLP Test: Allo ({mode_names[mode]}) vs PyTorch | dtype={allo_dtype}")
    print("=" * 80)
    
    # Dimension selection based on reduced flag or overrides
    if L_override is not None and D_override is not None:
        L, D = L_override, D_override
    else:
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
    # LayerNorm parameters: use values appropriate for int8 quantized outputs
    gamma = np.random.randint(10, 31, size=D, dtype=np.int8)  # Scale factors 10-30
    beta = np.random.randint(-5, 6, size=D, dtype=np.int8)  # Bias -5 to 5
    
    print(f"Config: L={L}, D={D}")
    print(f"  Input size: {X.nbytes / 1024:.1f} KB")
    print(f"  Weight sizes: W_1={W_1.nbytes / 1024:.1f} KB, W_2={W_2.nbytes / 1024:.1f} KB")
    
    # Time estimates
    time_estimates = {
        "sw_emu": "~1 minute",
        "hw_emu": "~5-15 minutes (includes HLS synthesis)",
        "hw": "~2-6 hours (includes P&R and bitstream generation)",
        "csyn": "~2-5 minutes (C synthesis only)"
    }
    print(f"  Estimated time: {time_estimates[mode]}")
    
    # PyTorch reference
    print("\nPyTorch reference...")
    pytorch_out, intermediates = pytorch_int8_mlp(X, W_1, B_1, W_2, B_2, gamma, beta)
    print(f"  Output range: [{pytorch_out.min()}, {pytorch_out.max()}]")
    print(f"  Building HLS project in '{mode}' mode...")
    s = allo.customize(mlp_dataflow, instantiate=[allo_dtype, D, L])
    # Apply schedule overrides if provided
    if schedule_spec is not None:
        kind = schedule_spec[0]
        if kind == "dataflow":
            enable_df = schedule_spec[1]
            if enable_df:
                try:
                    s.dataflow("mlp_dataflow")
                except Exception:
                    pass
        elif kind == "parallel":
            pf = schedule_spec[1]
            try:
                s.dataflow("mlp_dataflow")
                s.pipeline("mlp_dataflow", axis=1)
                s.unroll("mlp_dataflow", axis=1, factor=pf)
            except Exception:
                pass
    
    # Check if dataflow is enabled
    print(f"  Schedule info: {s}")
    
    try:
        proj = project_name if project_name is not None else f"mlp_test_{mode}_L{L}_D{D}.prj"
        
        if mode == "csyn":
            mod = s.build(
                target="vitis_hls",
                mode="csyn",
                project=proj
            )
            print(f"  Running C synthesis...")
            mod()  # csyn mode doesn't execute, just synthesizes
            print(f"  ✓ Synthesis completed")
            return True  # Skip verification for synthesis-only mode
        else:
            # For emulation/hardware modes, use vitis_hls
            mod = s.build(
                target="vitis_hls",
                mode=mode,
                project=proj
            )
        
            print(f"  Running {mode_names[mode].lower()}...")
            allo_out = np.zeros((L, D), dtype=np.int8)
            print(f"  Input X range: [{X.min()}, {X.max()}]")
            print(f"  Gamma range: [{gamma.min()}, {gamma.max()}]")
            print(f"  Beta range: [{beta.min()}, {beta.max()}]")
            mod(X, W_1, B_1, W_2, B_2, gamma, beta, allo_out)
            print(f"  Output range: [{allo_out.min()}, {allo_out.max()}]")
        
        # Compare
        diff = np.abs(pytorch_out.astype(np.int32) - allo_out.astype(np.int32))
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="sw_emu", help="Run mode: sw_emu, hw_emu, hw, csyn")
    parser.add_argument("--reduced", action="store_true", help="Use reduced tensor sizes")
    parser.add_argument("--schedule", default=None, help="Schedule spec: dataflow or parallel")
    parser.add_argument("--parallel-factor", type=int, default=None, help="Parallel unroll factor")
    parser.add_argument("--L", type=int, default=None, help="Override L (tokens)")
    parser.add_argument("--D", type=int, default=None, help="Override D (features)")
    parser.add_argument("--project", type=str, default=None, help="HLS project name (folder)")
    args = parser.parse_args()

    # ============ CONFIGURATION ============
    TEST_TYPE = args.mode     # Options: "sw_emu", "hw_emu", "hw", "csyn"
    DTYPE = int8             # Data type for test
    REDUCED_SIZE = args.reduced
    # =======================================
    
    print("  MLP VERIFICATION TEST")
    print(f"\nTest Type: {TEST_TYPE}")
    print(f"Data Type: {DTYPE}")
    print(f"Reduced Size: {REDUCED_SIZE}")
    print()
    
    try:
        # Build schedule_spec from args
        schedule_spec = None
        if args.schedule == "dataflow":
            schedule_spec = ("dataflow", True)
        elif args.schedule == "none":
            schedule_spec = ("dataflow", False)
        elif args.schedule == "parallel" and args.parallel_factor is not None:
            schedule_spec = ("parallel", args.parallel_factor)

        # Run test, allow csyn mode for synthesis-only runs
        success = test_mlp_hls(allo_dtype=DTYPE, mode=TEST_TYPE, reduced=REDUCED_SIZE,
                               schedule_spec=schedule_spec, L_override=args.L, D_override=args.D,
                               project_name=args.project)
        
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

