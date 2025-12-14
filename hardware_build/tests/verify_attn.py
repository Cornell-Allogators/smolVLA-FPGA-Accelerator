import sys
import os
sys.path.append(os.path.abspath("/home/er495/smolVLA-Cornell/hardware_build"))
sys.path.append(os.path.abspath("/home/er495/smolVLA-Cornell/hardware_build/attention/cross_attention"))

import torch
import numpy as np
import allo
from allo.ir.types import int8, float32, int16, int32
import attention.cross_attention.self_attention as sa_mod
from attention.cross_attention.self_attention import self_attention_return

def pytorch_reference(X, W_q, W_k, W_v, W_o, scale, gamma, beta):
    # X: [L, D]
    # W_q: [H, D_h, D]
    # W_o: [H, D_h, D]
    
    L, D = X.shape
    H, D_h, _ = W_q.shape
    
    # 1. QKV Projections
    # Need to reshape W for matmul: [D, H*D_h] or calculate per head
    # Allo code does: Q[L, Dh] = X[L, D] @ W_q[h, Dh, D].T
    # So Q_h = X @ W_q[h].T
    
    Q_all = []
    K_all = []
    V_all = []
    
    for h in range(H):
        q_h = X @ W_q[h].T # [L, D] @ [D, Dh] -> [L, Dh]
        k_h = X @ W_k[h].T
        v_h = X @ W_v[h].T
        Q_all.append(q_h)
        K_all.append(k_h)
        V_all.append(v_h)
        
    Q = torch.stack(Q_all) # [H, L, Dh]
    K = torch.stack(K_all)
    V = torch.stack(V_all)
    
    # 2. Attention Scores
    # [H, L, Dh] @ [H, Dh, L] -> [H, L, L]
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 3. Softmax
    # Allo does: (scores - max) / scale -> exp -> sum -> div -> scale_back
    # Note: Allo uses int32 accumulation for QK, so we should keep high precision
    scores = scores.float()
    
    # Simulate the online max/subtraction if strictly needed, but torch.softmax is close enough
    # Allo code: exp((x - max) / scale)
    attn_weights = torch.softmax(scores / scale, dim=-1)
    
    # 4. Context
    # [H, L, L] @ [H, L, Dh] -> [H, L, Dh]
    context = torch.matmul(attn_weights, V)
    
    # 5. Output Projection (Linear)
    # Allo: out += (context[p, k] >> 15) * W_o[h, k, i]
    # Wait, the allo code does:
    # out[i_out*P + p, i_final] += (val >> 15) * weight
    # Where val is acc_out (context). 
    # Context is int32. It shifts right by 15 before multiplying W_o.
    # We need to simulate this shift to match exactly.
    
    # Simulating the context quantization/shift
    # context is float in PyTorch, but int32 in Allo logic (accumulated QK * V)
    # Let's assume input X was integer for the test to match exactly?
    # Or just check functional equivalence.
    
    final_out = torch.zeros(L, D)
    
    # context: [H, L, Dh]
    # W_o: [H, Dh, D]
    for h in range(H):
        # context[h]: [L, Dh]
        # W_o[h]: [Dh, D]
        # out += context[h] @ W_o[h]
        
        # Allo shift logic: (val >> 15)
        # We can't easily emulate >> 15 in float without casting
        # For verification, we will verify the logic flow, not bit-exactness unless we use int inputs
        
        head_out = context[h] @ W_o[h] # [L, D]
        # Allo: out += head_out
        final_out += head_out
        
    # 6. Layer Norm
    # Standard LN
    mean = final_out.mean(dim=-1, keepdim=True)
    var = final_out.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + 1e-8)
    ln_out = (final_out - mean) / std * gamma + beta
    
    return ln_out, final_out

def verify():
    L, H, D, D_h = 1024, 12, 768, 64
    P, P_s = 1, 4 # Test baseline first
    
    # Random Inputs (use small integers to avoid overflow issues in simple verification)
    np.random.seed(42)
    X_np = np.random.randint(-5, 5, (L, D)).astype(np.float32)
    W_q_np = np.random.randint(-2, 2, (H, D_h, D)).astype(np.float32)
    W_k_np = np.random.randint(-2, 2, (H, D_h, D)).astype(np.float32)
    W_v_np = np.random.randint(-2, 2, (H, D_h, D)).astype(np.float32)
    W_o_np = np.random.randint(-2, 2, (H, D_h, D)).astype(np.float32)
    gamma_np = np.ones((D,), dtype=np.float32)
    beta_np = np.zeros((D,), dtype=np.float32)
    
    scale = 8.0
    
    # Run PyTorch
    print("Running PyTorch Reference...")
    ref_ln, ref_sa = pytorch_reference(
        torch.tensor(X_np),
        torch.tensor(W_q_np),
        torch.tensor(W_k_np),
        torch.tensor(W_v_np),
        torch.tensor(W_o_np),
        scale,
        torch.tensor(gamma_np),
        torch.tensor(beta_np)
    )
    ref_sa = ref_sa.numpy()
    
    # Run Allo (Self Attention Only)
    print("Building Allo Kernel (Self-Attention Only)...")
    # We test the inner self_attention function which outputs int32
    s = allo.customize(sa_mod.self_attention, instantiate=[
        int8, L, H, D, D_h, P, P_s
    ])
    mod = s.build(target="llvm")
    
    print("Running Allo Kernel...")
    # sa output is int32
    out_allo = np.zeros((L, D), dtype=np.int32)
    
    # Inputs cast to int8
    X_i8 = X_np.astype(np.int8)
    Wq_i8 = W_q_np.astype(np.int8)
    Wk_i8 = W_k_np.astype(np.int8)
    Wv_i8 = W_v_np.astype(np.int8)
    Wo_i8 = W_o_np.astype(np.int8)
    
    # Signature: (X, Wq, Wk, Wv, Wo, scale, out)
    mod(X_i8, Wq_i8, Wk_i8, Wv_i8, Wo_i8, scale, out_allo)
    
    print("-" * 40)
    print("Comparison (Self-Attention Pre-LN):")
    print("-" * 40)
    print(f"Allo (int32) Sample:\n{out_allo[0, :8]}")
    print(f"PyTorch (float) Sample:\n{ref_sa[0, :8]}")
    
    # Scale Check: Allo uses >> 15 (div 32768) after accumulating
    # PyTorch accumulated raw floats.
    # But wait, Allo shifts context >> 15 BEFORE accumulating Wo?
    # No, logic was: out += (context >> 15) * W_o.
    # PyTorch context was QK * V.
    # So PyTorch output should be roughly (Q*K*V) * W_o.
    # Allo is ((Q*K*V) / 32768) * W_o.
    # So we expect Allo ~= PyTorch / 32768.
    
    ratio = ref_sa / (out_allo + 1e-9)
    median_ratio = np.median(ratio)
    print(f"Median Ratio (Ref / Allo): {median_ratio:.2f}")
    print(f"Expected Ratio: ~32768.0")
    
    if 30000 < abs(median_ratio) < 35000:
        print("PASS: Scaling factor matches (~32768)!")
    else:
        print("WARN: Scaling factor mismatch.")

    # Correlation Check
    flat_allo = out_allo.flatten().astype(np.float32)
    flat_ref = ref_sa.flatten()
    corr = np.corrcoef(flat_allo, flat_ref)[0, 1]
    print(f"Correlation Coefficient: {corr:.4f}")
    
    if corr > 0.9:
        print("PASS: High functional correlation.")
    else:
        print("FAIL: Low correlation.")

if __name__ == "__main__":
    verify()
