import matplotlib.pyplot as plt
import numpy as np


plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'lines.linewidth': 2
})

FREQ = 300e6
BW_PEAK = 460e9
BW_REAL = 300e9

# Peak Compute (Ops/s)
P_FP32 = 5.41e12
P_BF16 = 5.41e12 
P_INT8 = 18.6e12
P_INT4 = 37.2e12

# --- Dimensions ---

# 1. Vision Encoder (ViT)
V_LAYERS = 12
V_D = 768
V_FFN = 3072
V_PATCHES = 1024 # 512x512 image / 16x16 patch

# 2. VLM Backbone (SmolLM2-360M)
T_LAYERS = 16
T_D = 960
T_FFN = 2560
T_Q_HEADS = 15
T_KV_HEADS = 5
T_HEAD_DIM = 64
T_SEQ = 113 # 64 Visual + 48 Text + 1 State

# 3. Action Expert (Transformer)
A_LAYERS = 16
A_D = 720 # 0.75 * 960
A_FFN = A_D * 4 # Standard expansion? Or 2048? model_shape.txt says 2048 (approx 3x)
A_FFN_REAL = 2048 # From model_shape.txt (720 -> 2048)
A_Q_HEADS = 12
A_KV_HEADS = 4
A_HEAD_DIM = 80
A_SEQ = 50
A_STEPS = 10 # Diffusion steps

# Cross Attention Projection
# VLM (960) -> KV (320)
# Performed ONCE per inference.
CA_PROJ_IN = 960
CA_PROJ_OUT = 320

precisions = {
    'FP32': 4,
    'BF16': 2,
    'INT8': 1,
    'INT4': 0.5
}

def calc_macs_linear(M, K, N):
    return M * K * N

def calc_oi_linear(M, K, N, dtype_bytes):
    flops = 2 * M * K * N
    bytes_xfer = (K*N + M*K + M*N) * dtype_bytes
    return flops / bytes_xfer

def analyze_kernel(name, M, K, N, p_bytes):
    flops = 2 * M * K * N
    mem_weights = K * N * p_bytes
    mem_io = (M * K + M * N) * p_bytes
    total_bytes = mem_weights + mem_io
    oi = flops / total_bytes
    return {
        'Kernel': name,
        'OI': oi,
        'MACs': M * K * N,
        'FLOPs': flops
    }

def plot_roofline_base(ax, title):
    x = np.logspace(-2, 3, 100)
    ceilings = [
        ('FP32/BF16', P_FP32, 'k-'),
        ('INT8', P_INT8, 'b-'),
        ('INT4', P_INT4, 'g-')
    ]
    y_mem = BW_REAL * x
    ax.loglog(x, y_mem, 'r--', label='Memory Wall (300 GB/s)')
    for name, peak, style in ceilings:
        y = np.minimum(peak, y_mem)
        ax.loglog(x, y, style, label=f'{name} Peak')
    ax.set_xlabel('Operational Intensity (Ops/Byte)')
    ax.set_ylabel('Performance (Ops/s)')
    ax.set_title(title)
    ax.grid(True, which="both", ls="-", alpha=0.5)

def plot_kernels_improved(ax, metrics_list, peak_perf):
    for i, item in enumerate(metrics_list):
        oi = item['OI']
        perf = min(peak_perf, BW_REAL * oi)
        ax.plot(oi, perf, 'b^', markersize=12)
        offset = 1.4 if i % 2 == 0 else 0.6
        va = 'bottom' if offset > 1 else 'top'
        ax.text(oi, perf * offset, item['Kernel'], fontsize=9, ha='center', va=va)

# ==========================================
# ANALYSIS
# ==========================================

p_bytes = precisions['INT8']
total_macs = 0

# --- 1. Vision Encoder ---
vision_kernels = []
# Patch Embed: 1024 patches, 3 channels, 16x16 kernel, 768 out
# In effect: 1024 x (3*16*16) x 768
vision_kernels.append(analyze_kernel('PatchEmbed', V_PATCHES, 3*16*16, V_D, p_bytes))
# Layers
for _ in range(V_LAYERS):
    # Attn QKV: 1024 x 768 x (3*768) -> Treated as 3 kernels or 1
    vision_kernels.append(analyze_kernel('Attn_QKV', V_PATCHES, V_D, 3*V_D, p_bytes))
    # Attn Out: 1024 x 768 x 768
    vision_kernels.append(analyze_kernel('Attn_Out', V_PATCHES, V_D, V_D, p_bytes))
    # MLP FC1: 1024 x 768 x 3072
    vision_kernels.append(analyze_kernel('MLP_FC1', V_PATCHES, V_D, V_FFN, p_bytes))
    # MLP FC2: 1024 x 3072 x 768
    vision_kernels.append(analyze_kernel('MLP_FC2', V_PATCHES, V_FFN, V_D, p_bytes))

# Connector: Flattened tokens? Or per token?
# Report says: Proj 12288 -> 960.
# Assuming this runs ONCE per inference (Global context? Or per patch?)
# Report says "single-camera mode with sequence length L=113... 64 visual tokens".
# The connector likely takes the 1024 patches and reduces/projects them.
# Let's assume input M=1 for the connector projection (Global).
vision_kernels.append(analyze_kernel('Connector', 1, 12288, 960, p_bytes))

v_macs = sum(k['MACs'] for k in vision_kernels)
total_macs += v_macs

# --- 2. VLM Backbone (Single Pass) ---
# Sequence Length L = 113.
text_kernels = []
for _ in range(T_LAYERS):
    # Q Proj: 113 x 960 x (15*64 = 960)
    text_kernels.append(analyze_kernel('Attn_Q', T_SEQ, T_D, T_Q_HEADS * T_HEAD_DIM, p_bytes))
    # K Proj: 113 x 960 x (5*64 = 320)
    text_kernels.append(analyze_kernel('Attn_K', T_SEQ, T_D, T_KV_HEADS * T_HEAD_DIM, p_bytes))
    # V Proj: 113 x 960 x 320
    text_kernels.append(analyze_kernel('Attn_V', T_SEQ, T_D, T_KV_HEADS * T_HEAD_DIM, p_bytes))
    # Out Proj: 113 x 960 x 960
    text_kernels.append(analyze_kernel('Attn_Out', T_SEQ, T_D * T_D, 1, p_bytes)) # Simulating simplified linear
    text_kernels.append(analyze_kernel('Attn_Out_Real', T_SEQ, T_Q_HEADS * T_HEAD_DIM, T_D, p_bytes))

    # MLP: Gate (960->2560), Up (960->2560), Down (2560->960)
    text_kernels.append(analyze_kernel('MLP_Gate', T_SEQ, T_D, T_FFN, p_bytes))
    text_kernels.append(analyze_kernel('MLP_Up', T_SEQ, T_D, T_FFN, p_bytes))
    text_kernels.append(analyze_kernel('MLP_Down', T_SEQ, T_FFN, T_D, p_bytes))

t_macs = sum(k['MACs'] for k in text_kernels)
total_macs += t_macs

# --- 3. Action Expert (10 Steps) ---
action_kernels = []

# Pre-computation: VLM Context Projection (960 -> 320)
# Runs ONCE per inference. Context Length = 113.
# 113 x 960 x 320
action_kernels.append(analyze_kernel('Context_Proj', 113, CA_PROJ_IN, CA_PROJ_OUT, p_bytes))

# Step Loop
for step in range(A_STEPS):
    for _ in range(A_LAYERS):
        # -- Self Attention --
        # Q (720->960), K (720->320), V (720->320), Out (960->720)
        action_kernels.append(analyze_kernel('SA_Q', A_SEQ, A_D, A_Q_HEADS*A_HEAD_DIM, p_bytes))
        action_kernels.append(analyze_kernel('SA_K', A_SEQ, A_D, A_KV_HEADS*A_HEAD_DIM, p_bytes))
        action_kernels.append(analyze_kernel('SA_V', A_SEQ, A_D, A_KV_HEADS*A_HEAD_DIM, p_bytes))
        action_kernels.append(analyze_kernel('SA_Out', A_SEQ, A_Q_HEADS*A_HEAD_DIM, A_D, p_bytes))
        
        # -- Cross Attention --
        # Q (720->960) - Dynamic
        # K, V are PRE-COMPUTED (Static from Context Proj). No MACs per step for projection!
        # Only Attention Score (Matmul1) and Weighted Sum (Matmul2) cost.
        # But here we calculate LINEAR layer MACs.
        # Q Proj:
        action_kernels.append(analyze_kernel('CA_Q', A_SEQ, A_D, A_Q_HEADS*A_HEAD_DIM, p_bytes))
        # Out Proj:
        action_kernels.append(analyze_kernel('CA_Out', A_SEQ, A_Q_HEADS*A_HEAD_DIM, A_D, p_bytes))

        # -- MLP --
        # FFN: 720 -> 2048 -> 720 (Gate/Up/Down style? Usually 3 matrices in SwiGLU, 2 in GeLU)
        # model_shape.txt showed down/gate/up, so 3 matrices.
        action_kernels.append(analyze_kernel('Exp_MLP_Gate', A_SEQ, A_D, A_FFN_REAL, p_bytes))
        action_kernels.append(analyze_kernel('Exp_MLP_Up', A_SEQ, A_D, A_FFN_REAL, p_bytes))
        action_kernels.append(analyze_kernel('Exp_MLP_Down', A_SEQ, A_FFN_REAL, A_D, p_bytes))

a_macs = sum(k['MACs'] for k in action_kernels)
total_macs += a_macs

# --- Output ---
print(f"{'Component':<20} | {'MACs (M)':<15} | {'Ops (G)':<15} | {'% Total':<10}")
print("-" * 65)
print(f"{'Vision Encoder':<20} | {v_macs/1e6:15.2f} | {2*v_macs/1e9:15.2f} | {100*v_macs/total_macs:10.1f}%")
print(f"{'VLM Backbone':<20} | {t_macs/1e6:15.2f} | {2*t_macs/1e9:15.2f} | {100*t_macs/total_macs:10.1f}%")
print(f"{'Action Expert':<20} | {a_macs/1e6:15.2f} | {2*a_macs/1e9:15.2f} | {100*a_macs/total_macs:10.1f}%")
print("-" * 65)
print(f"{'TOTAL':<20} | {total_macs/1e6:15.2f} | {2*total_macs/1e9:15.2f} | 100.0%")

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
plot_roofline_base(ax, 'Unified Roofline Analysis')
# Plot a subset of kernels to avoid clutter
plot_kernels_improved(ax, vision_kernels[::2] + text_kernels[::3] + action_kernels[::10], P_INT8)
plt.savefig('unified_roofline.png')
print("Saved unified_roofline.png")
