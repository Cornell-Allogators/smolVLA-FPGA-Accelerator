
# Data Type Size (INT8)
DTYPE_SIZE = 1 # Bytes

# Vision Encoder Parameters
V_LAYERS = 12
V_L = 1024
V_D = 768
V_FFN = 3072

def calc_vision_mem():
    # Weights (Read Once per Layer? Or Once per Inference if reuse? 
    # Assumption: Layer-by-Layer execution, weights streamed from HBM per layer.
    # Weights fit in HBM, not on-chip.
    # W_Linear = (D*D)*4 (Q,K,V,Out) + (D*FFN)*2 (FC1, FC2)
    # Norms/Biases negligible for rough estimate.
    w_params_layer = (4 * V_D**2) + (2 * V_D * V_FFN)
    w_bytes_layer = w_params_layer * DTYPE_SIZE
    
    # Activations (Input + Output per Layer)
    # Read In: L*D
    # Write Out: L*D
    # (Intermediate tensors like QK^T kept on chip hopefully)
    act_bytes_layer = (V_L * V_D * 2) * DTYPE_SIZE
    
    total_bytes = V_LAYERS * (w_bytes_layer + act_bytes_layer)
    return total_bytes, w_params_layer * V_LAYERS * DTYPE_SIZE

# VLM Backbone Parameters
T_LAYERS = 16
T_L = 113
T_D = 960
T_FFN = 2560
# GQA: Q(D->D), K(D->D/3), V(D->D/3), Out(D->D)
# MLP: Gate, Up, Down (3 matrices)

def calc_vlm_mem():
    # Weights
    # Q, Out: D^2 each. K, V: D^2/3 each.
    # MLP: 3 * D * FFN.
    w_params_layer = (T_D**2 * (1 + 1 + 1/3 + 1/3)) + (3 * T_D * T_FFN)
    w_bytes_layer = w_params_layer * DTYPE_SIZE
    
    # Activations
    act_bytes_layer = (T_L * T_D * 2) * DTYPE_SIZE
    
    total_bytes = T_LAYERS * (w_bytes_layer + act_bytes_layer)
    return total_bytes

# Action Expert Parameters
A_LAYERS = 16
A_L = 50
C_L = 113
A_D = 720
A_FFN = 2048
STEPS = 10
# GQA: 12Q, 4KV (1/3)
# Head Dim = 60.

def calc_action_mem():
    # Weights (Per Layer)
    # Self-Attn: Q,K,V,Out. D=720. 
    #   Q, Out: D^2. K, V: D^2/3.
    # Cross-Attn: Q, Out (D^2). K, V (From VLM Context, projection matrices).
    #   Cross K, V Proj: VLM_D(960) -> Expert_KV(240). (Stored Layer-Wise).
    # MLP: 3 * D * FFN.
    
    # 1. Self Attn Weights
    w_sa = (A_D**2 * (1 + 1 + 1/3 + 1/3))
    
    # 2. Cross Attn Weights
    # Dynamic (Q, Out)
    w_ca_dynamic = 2 * A_D**2
    # Static (K, V Proj from VLM) - Read ONCE
    w_ca_static = 2 * 960 * 240
    
    # 3. MLP Weights (Dynamic)
    w_mlp = 3 * A_D * A_FFN
    
    w_dynamic_layer = w_sa + w_ca_dynamic + w_mlp
    w_dynamic_bytes = w_dynamic_layer * DTYPE_SIZE
    w_static_bytes = w_ca_static * DTYPE_SIZE
    
    # Activations (Per Step)
    act_bytes_step_layer = (A_L * A_D * 2) * DTYPE_SIZE
    
    # Total Access:
    # Static Weights: Read 1x
    # Dynamic Weights: Read 10x
    # Activations: Read/Write 10x
    
    total_w_access = (A_LAYERS * w_static_bytes) + (10 * A_LAYERS * w_dynamic_bytes)
    total_act_access = 10 * A_LAYERS * act_bytes_step_layer
    
    return total_w_access + total_act_access


    
    return total_w_access + total_act_access

vision_mem, vis_weights_total = calc_vision_mem()
vlm_mem = calc_vlm_mem()
action_mem = calc_action_mem()

# --- Footprint Calculations ---

# 1. Total Weight Storage (HBM Requirement)
# We need to re-sum the raw weight params, not just the transfer.
# Vision: (12 * ((4*768^2) + (2*768*3072))) * 1
vis_w_size = 12 * ((4 * 768**2) + (2 * 768 * 3072)) * DTYPE_SIZE
# VLM: 16 * ((960^2 * 8/3) + (3*960*2560))
vlm_w_size = 16 * ((960**2 * (8/3)) + (3 * 960 * 2560)) * DTYPE_SIZE
# Action: 16 * (SelfAttn + CrossAttn + MLP)
# Self: 720^2 * 8/3. Cross: 2*720^2 + 2*960*240. MLP: 3*720*2048.
act_layer_w = (720**2 * (8/3)) + (2 * 720**2) + (2 * 960 * 240) + (3 * 720 * 2048)
act_w_size = 16 * act_layer_w * DTYPE_SIZE

total_weight_footprint = vis_w_size + vlm_w_size + act_w_size

# 2. Peak Activation Footprint (On-Chip Requirement)
# Assuming double-buffering (Input Buffer + Output Buffer) for a layer.
# Vision: L=1024, D=768. 2 buffers.
vis_act_peak = 2 * 1024 * 768 * DTYPE_SIZE
# VLM: L=113, D=960.
vlm_act_peak = 2 * 113 * 960 * DTYPE_SIZE
# Action: L=50, D=720.
act_act_peak = 2 * 50 * 720 * DTYPE_SIZE

# 3. Static Context KV Cache (Action Expert)
# Stored On-Chip for efficiency?
# Size: 113 (Tokens) * 240 (KV Dim) * 2 (K+V)? No, KV Dim is usually sum of heads. 
# Code check: Experts KV Heads=4, Dim=60. Total KV Dim=240. 
# So 113 * 240 * 1 bytes? No, K and V are separate matrices usually. 
# "KV_DIM = 320" in config? 
# Wait, let's stick to calc: 4 heads * 60 dim = 240.
# So K_cache = 113 * 240. V_cache = 113 * 240.
# Total = 2 * 113 * 240 * DTYPE_SIZE.
ctx_kv_size = 2 * 113 * 240 * DTYPE_SIZE

print(f"--- Memory Footprint (Storage) ---")
print(f"Total Weights (HBM):    {total_weight_footprint/1e6:.2f} MB")
print(f"  - Vision: {vis_w_size/1e6:.2f} MB")
print(f"  - VLM:    {vlm_w_size/1e6:.2f} MB")
print(f"  - Action: {act_w_size/1e6:.2f} MB")
print(f"Peak Activation (On-Chip): {max(vis_act_peak, vlm_act_peak, act_act_peak)/1e6:.2f} MB")
print(f"  - Vision Peak: {vis_act_peak/1e6:.2f} MB")
print(f"  - VLM Peak:    {vlm_act_peak/1e6:.2f} MB")
print(f"Action Context KV Cache:   {ctx_kv_size/1e3:.2f} KB")

