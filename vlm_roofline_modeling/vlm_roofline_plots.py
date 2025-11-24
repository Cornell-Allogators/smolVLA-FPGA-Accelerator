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

# Dimensions
V_LAYERS = 12
V_D = 768
V_FFN = 3072
V_ATTN_D = 768

T_LAYERS = 16
T_D = 960
T_FFN = 2560
T_Q_D = 960
T_K_D = 320
T_V_D = 320
T_OUT_D = 960

CONN_IN = 12288
CONN_OUT = 960

B = 1
S = 50 
S_V = 256

precisions = {
    'FP32': 4,
    'BF16': 2,
    'INT8': 1,
    'INT4': 0.5
}

def calc_oi_linear(M, K, N, dtype_bytes):
    flops = 2 * M * K * N
    bytes_xfer = (K*N + M*K + M*N) * dtype_bytes
    return flops / bytes_xfer

metrics = {}
for p_name, p_bytes in precisions.items():
    metrics[p_name] = {}
    metrics[p_name]['Vis_MLP'] = calc_oi_linear(B*S_V, V_D, V_FFN, p_bytes)
    metrics[p_name]['Vis_PatchEmbed'] = calc_oi_linear(1024, 3*16*16, 768, p_bytes)
    metrics[p_name]['Txt_MLP'] = calc_oi_linear(B*S, T_D, T_FFN, p_bytes)
    metrics[p_name]['Txt_Attn_Q'] = calc_oi_linear(B*S, T_D, T_Q_D, p_bytes)
    metrics[p_name]['Txt_Attn_K'] = calc_oi_linear(B*S, T_D, T_K_D, p_bytes)
    metrics[p_name]['Connector'] = calc_oi_linear(1, 12288, 960, p_bytes)

def plot_roofline_base(ax, title):
    x = np.logspace(-1, 3, 100)
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

def plot_kernels(ax, kernel_names, metrics_dict):
    markers = {'FP32': 'o', 'BF16': 's', 'INT8': '^', 'INT4': 'D'}
    colors = {'FP32': 'black', 'BF16': 'orange', 'INT8': 'blue', 'INT4': 'green'}
    for p_name, vals in metrics_dict.items():
        if p_name in ['FP32', 'BF16']: peak = P_FP32
        elif p_name == 'INT8': peak = P_INT8
        else: peak = P_INT4
        for k_name in kernel_names:
            if k_name not in vals: continue
            oi = vals[k_name]
            perf = min(peak, BW_REAL * oi)
            ax.plot(oi, perf, marker=markers[p_name], color=colors[p_name], markersize=10)
            if p_name == 'INT4':
                ax.text(oi, perf*1.3, k_name, fontsize=8, ha='center', rotation=45)

# Vision Plot
fig, ax = plt.subplots(figsize=(10, 7))
plot_roofline_base(ax, 'Vision Encoder Roofline')
plot_kernels(ax, ['Vis_MLP', 'Vis_PatchEmbed', 'Connector'], metrics)
ax.legend()
plt.tight_layout()
plt.savefig('vision_roofline.png')
print("Saved vision_roofline.png")

# Text Plot
fig, ax = plt.subplots(figsize=(10, 7))
plot_roofline_base(ax, 'Text Encoder Roofline')
plot_kernels(ax, ['Txt_MLP', 'Txt_Attn_Q', 'Txt_Attn_K'], metrics)
ax.legend()
plt.tight_layout()
plt.savefig('text_roofline.png')
print("Saved text_roofline.png")
