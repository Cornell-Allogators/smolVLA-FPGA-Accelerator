import matplotlib.pyplot as plt
import numpy as np
import os

# Data
p_factors = [1, 2, 4, 8]
lat_false = [76.2, 38.2, 21.0, 15.3] # Millions of cycles
lat_true = [18.3, 11.3, 8.0, 6.4]    # Millions of cycles

dsp_true = [621, 988, 2160, 4888]
bram_true = [2837, 2846, 3452, 5292]

# Artifacts Directory
output_dir = "/home/er495/.gemini/antigravity/brain/6b1adbeb-bfd4-4dba-9a5b-e66a38d05802"

# 1. Latency Comparison (Bar Chart)
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(p_factors))

plt.bar(index, lat_false, bar_width, label='Dataflow=False', color='#ff9999', edgecolor='black')
plt.bar(index + bar_width, lat_true, bar_width, label='Dataflow=True', color='#66b3ff', edgecolor='black')

plt.xlabel('Parallelism Factor (P)', fontsize=12)
plt.ylabel('Latency (Million Cycles)', fontsize=12)
plt.title('Latency Reduction: Dataflow vs No-Dataflow', fontsize=14)
plt.xticks(index + bar_width / 2, [f'P={p}' for p in p_factors])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(lat_false):
    plt.text(i, v + 1, f"{v:.1f}M", ha='center', fontsize=9)
for i, v in enumerate(lat_true):
    plt.text(i + bar_width, v + 1, f"{v:.1f}M", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'latency_comparison.svg'), format='svg', bbox_inches='tight')
plt.close()

# 2. Resource Scaling (Dual Axis Line Chart)
fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()

ax1.set_xlabel('Parallelism Factor (P)', fontsize=12)
ax1.set_ylabel('DSP Usage', color='tab:red', fontsize=12)
ax1.plot(p_factors, dsp_true, color='tab:red', marker='o', linewidth=2, label='DSP')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.set_ylim(0, 6000)

ax2 = ax1.twinx()
ax2.set_ylabel('BRAM Usage (18K)', color='tab:blue', fontsize=12)
ax2.plot(p_factors, bram_true, color='tab:blue', marker='s', linewidth=2, label='BRAM')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.set_ylim(0, 6000)

plt.title('Resource Scaling vs P-Factor (Dataflow=True)', fontsize=14)
plt.xticks(p_factors)
plt.grid(True, linestyle='--', alpha=0.5)

fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'resource_scaling.svg'), format='svg', bbox_inches='tight')
plt.close()

# 3. Efficiency/Speedup (Diminishing Returns)
speedup = [lat_true[0]/l for l in lat_true] # Speedup relative to P=1 True
ideal = [p for p in p_factors] # Ideal scaling

plt.figure(figsize=(10, 6))
plt.plot(p_factors, ideal, 'k--', label='Ideal Linear Scaling')
plt.plot(p_factors, speedup, 'g-o', linewidth=2, label='Measured Speedup (vs P=1)')

plt.xlabel('Parallelism Factor (P)', fontsize=12)
plt.ylabel('Speedup Factor', fontsize=12)
plt.title('Diminishing Returns of P-Scaling', fontsize=14)
plt.xticks(p_factors)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Annotate P=8 point
plt.annotate('Softmax Stall Limit', xy=(8, speedup[-1]), xytext=(5, 2.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'efficiency_scaling.svg'), format='svg', bbox_inches='tight')
plt.close()

print("Plots generated successfully.")
