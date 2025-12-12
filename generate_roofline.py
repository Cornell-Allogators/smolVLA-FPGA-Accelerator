
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Parameters ---
# Model Metrics (Calculated Previously)
# Vision: MACs=106.30 G, Transfer=103.81 MB
# VLM: MACs=18.17 G, Transfer=160.76 MB
# Action: MACs=57.45 G, Transfer=1113.75 MB

# Convert to OPs (2x MACs) and Bytes
kernels = {
    "Vision Encoder": {"ops": 106.30 * 2 * 1e9, "bytes": 103.81 * 1e6},
    "VLM Backbone":   {"ops": 18.17 * 2 * 1e9, "bytes": 160.76 * 1e6},
    "Action Expert":  {"ops": 57.45 * 2 * 1e9, "bytes": 1113.75 * 1e6},
}

# Hardware Specs (Alveo U280)
# Peak Compute (TOPS): 9024 DSPs * 2 Ops/Cycle * 450 MHz (Allocated Freq?)
# Note: Report says 300MHz earlier, but 450MHz is achievable. 
# Let's use a conservative 300 MHz for baseline if not specified.
# 9024 * 2 * 300e6 = 5.41 TOPS.
PEAK_OPS = 5.41 * 1e12  # 5.41 TOPS
PEAK_BW = 460 * 1e9     # 460 GB/s

# -----------------

# Calculate OI and Performance
print("--- Operational Intensity (OI) ---")
plot_points = []
for name, data in kernels.items():
    oi = data["ops"] / data["bytes"]
    perf = min(PEAK_OPS, PEAK_BW * oi)
    print(f"{name}: OI = {oi:.2f} Ops/Byte, Perf Limit = {perf/1e12:.2f} TOPS")
    plot_points.append((name, oi, perf))

print("-" * 30)
ridge_point = PEAK_OPS / PEAK_BW
print(f"Ridge Point: {ridge_point:.2f} Ops/Byte")

# --- Plotting Roofline ---
fig, ax = plt.subplots(figsize=(8, 6))

# X-axis: Operational Intensity (Log Scale)
x = np.logspace(-1, 4, 100)
# Y-axis: Performance (Log Scale)
y_mem = x * PEAK_BW
y_compute = np.full_like(x, PEAK_OPS)
y = np.minimum(y_mem, y_compute)

# Plot Roofline
ax.plot(x, y / 1e12, 'k-', linewidth=2, label='U280 Roofline')
ax.set_xscale('log')
ax.set_yscale('log')

# Labels
ax.set_xlabel('Operational Intensity (Ops/Byte)')
ax.set_ylabel('Performance (TOPS)')
ax.set_title('Roofline Model: SmolVLA on Alveo U280')
ax.grid(True, which="both", ls="-", alpha=0.5)

# Plot Kernels
colors = ['red', 'blue', 'green']
for i, (name, oi, perf) in enumerate(plot_points):
    ax.scatter(oi, perf / 1e12, color=colors[i], s=100, label=name, zorder=5)
    # Annotate
    ax.annotate(f"{name}\n({oi:.0f} Ops/B)", (oi, perf/1e12), 
                xytext=(0, -20), textcoords='offset points', 
                ha='center', fontsize=9)

ax.legend()

# Save Limits
ax.set_xlim(0.1, 10000)
ax.set_ylim(0.01, 100) # Show range

# Ensure directories exist
os.makedirs("report/figures", exist_ok=True)
save_path = "report/figures/roofline_analysis.png"
plt.savefig(save_path, dpi=300)
print(f"Plot saved to {save_path}")
