import math

# Constants
BRAM_CAPACITY_KB = 2.0  # 18Kb roughly 2.25KB, use 2.0 for safety/parity
PARTITION_FACTOR = 64   # Common unroll factor for FPGA spatial arch

def calc_bram_usage(name, length, dim, num_buffers=2):
    total_bytes = length * dim
    bytes_per_partition = total_bytes / PARTITION_FACTOR
    
    # Blocks per partition
    blocks_per_partition = math.ceil(bytes_per_partition / (BRAM_CAPACITY_KB * 1024))
    
    # Total blocks
    total_blocks = blocks_per_partition * PARTITION_FACTOR * num_buffers
    
    # Efficiency
    utilized_kb = total_bytes * num_buffers / 1024
    allocated_kb = total_blocks * BRAM_CAPACITY_KB
    efficiency = utilized_kb / allocated_kb
    
    return total_blocks, allocated_kb, efficiency

print(f"{'Component':<20} | {'Shape':<15} | {'BRAMs':<5} | {'Alloc (MB)':<10} | {'Util (MB)':<10} | {'Eff'}")
print("-" * 80)

total_alloc_mb = 0

# 1. Vision Encoder Results (Intermediate Activations)
# We assume we buffer the Input (L, D) and the MLP Expansion (L, E).
# Usually we don't buffer the FULL MLP expansion on-chip if we stream?
# But if it's "layer-by-layer", we usually buffer the output of the layer.
# Let's model:
# - Residual Buffer (L, D): 2x (Ping Pong)
# - Attention Output (L, D): 2x
v_l, v_d, v_e = 1024, 768, 3072
b, alloc, eff = calc_bram_usage("Vision Res", v_l, v_d)
print(f"{'Vision Resid':<20} | {f'{v_l}x{v_d}':<15} | {b:<5} | {alloc/1024:.2f} MB     | {eff:.2f}")
total_alloc_mb += alloc/1024

# If MLP expansion is buffered (worst case):
b, alloc, eff = calc_bram_usage("Vision MLP", v_l, v_e)
print(f"{'Vision MLP':<20} | {f'{v_l}x{v_e}':<15} | {b:<5} | {alloc/1024:.2f} MB     | {eff:.2f}")
# Usually we don't store full MLP expansion. We compute it row-wise.
# So we only count the Residual/Layer IO.

# 2. VLM
t_l, t_d = 113, 960
b, alloc, eff = calc_bram_usage("VLM Resid", t_l, t_d)
print(f"{'VLM Resid':<20} | {f'{t_l}x{t_d}':<15} | {b:<5} | {alloc/1024:.2f} MB     | {eff:.2f}")
total_alloc_mb += alloc/1024

# 3. Action
a_l, a_d = 50, 720
b, alloc, eff = calc_bram_usage("Action Resid", a_l, a_d)
print(f"{'Action Resid':<20} | {f'{a_l}x{a_d}':<15} | {b:<5} | {alloc/1024:.2f} MB     | {eff:.2f}")
total_alloc_mb += alloc/1024

# 4. KV Cache (VLM)
# 16 Layers * 113 * 320 * 2 (K+V).
# This is persistent active state.
# Is it partitioned? Yes, for attention.
# But 16 layers... do we store ALL on chip?
# "Action Context Cache: 54 KB".
# That was for ONE layer? Or compressed?
# The user text said "VLM Key/Value states needed for the 10-step diffusion".
# If we store all layers: 16 * 113 * 320 * 2 = 1.1 MB.
# Partitioned by 64? (KV Head dim is 64? No, 80/64).
# Assuming Partition = 64.
kv_size = 113 * 320 * 2
# Per layer
b, alloc, eff = calc_bram_usage("VLM KV (1 Layer)", 113, 640, num_buffers=1)
# Total for 16 layers
total_kv_alloc = (alloc/1024) * 16
print(f"{'VLM KV (All)':<20} | {f'16x{113}x640':<15} | {b*16:<5} | {total_kv_alloc:.2f} MB     | {eff:.2f}")
total_alloc_mb += total_kv_alloc

print("-" * 80)
print(f"Total Model Alloc (Buffers + Cache): {total_alloc_mb:.2f} MB")
