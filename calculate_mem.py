
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
# 1. Total Weight Storage (HBM Requirement)
# We need to re-sum the raw weight params, not just the transfer.
# Vision: (12 * ((4*768^2) + (2*768*3072))) * 1
vis_w_size = 12 * ((4 * 768**2) + (2 * 768 * 3072)) * DTYPE_SIZE
# VLM: 16 * ((960^2 * 8/3) + (3*960*2560))
vlm_w_size = 16 * ((960**2 * (8/3)) + (3 * 960 * 2560)) * DTYPE_SIZE

# Action Expert: 16 layers interleaved.
# Based on model_shape.txt analysis:
# Layers 0, 2, ... 14 (8 Layers): "Self Attention"
#   Q: 960x720, K: 320x720, V: 320x720, O: 720x960. MLP: 3*720*2048.
# Layers 1, 3, ... 15 (8 Layers): "Cross Attention" (implied via 320-dim input K/V)
#   Q: 960x720, K: 320x320, V: 320x320, O: 720x960. MLP: 3*720*2048.

# Even Layer (SA) Params:
act_sa_layer = (960*720) + (320*720) + (320*720) + (720*960) + (3 * 720 * 2048)
# Odd Layer (CA) Params:
act_ca_layer = (960*720) + (320*320) + (320*320) + (720*960) + (3 * 720 * 2048)

act_w_size = (8 * act_sa_layer + 8 * act_ca_layer) * DTYPE_SIZE

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

# 4. Detailed Footprint (Matching model_shape.txt)

# --- Parameters from model_shape.txt ---
VOCAB_SIZE = 49280
VLM_DIM = 960
VIS_DIM = 768
CONNECTOR_DIM = 12288 # 960x12288

# A. Vision Encoder Params
# Patch Embed: 768 * 3 * 16 * 16
vis_patch_embed = 768 * 3 * 16 * 16
# Pos Embed: 1024 * 768
vis_pos_embed = 1024 * 768
# Layers (12x): Norms (4*768) + Attn (4*768^2 + 4*768 bias) + MLP (2*3072*768 + 2*3072 bias + ... bias?)
# Let's trust my vis_w_size calc for weights, but add biases/norms.
# Bias approx: 12 * (4*768 + 2*3072 + 4*768) ~ small.
vis_total_params = vis_w_size + vis_patch_embed + vis_pos_embed

# B. VLM Backbone Params
# Embed Tokens: 49280 * 960
vlm_embed = VOCAB_SIZE * VLM_DIM
# Layers (16x): Already in vlm_w_size (Weights). Add biases/norms.
# Norms: 2 per layer * 960.
vlm_extra = 16 * (2 * 960)
vlm_total_params = vlm_w_size + vlm_embed + vlm_extra

# C. Action Expert Params
# Layers (16x): Already in act_w_size.
act_total_params = act_w_size

# D. Connector & Heads
# Connector: 960 * 12288
connector_params = 960 * 12288
# LM Head: 49280 * 960
lm_head_params = VOCAB_SIZE * VLM_DIM

# --- Total Sum ---
# Note: calc_*_mem returned Op *Bytes*. We need Op *Count* first.
# My previous *_w_size vars were (Count * DTYPE).
# Let's revert to counts by dividing by DTYPE (1).
total_params_count = (vis_total_params + vlm_total_params + act_total_params) + connector_params + lm_head_params

bf16_size_mb = (total_params_count * 2) / 1e6
int8_size_mb = (total_params_count * 1) / 1e6

print(f"--- Comprehensive Memory Footprint ---")
print(f"1. Vision Encoder:   {vis_total_params/1e6:.2f} M Params")
print(f"2. VLM Backbone:     {vlm_total_params/1e6:.2f} M Params (incl. Embed)")
print(f"3. Action Expert:    {act_total_params/1e6:.2f} M Params")
print(f"4. Connector:        {connector_params/1e6:.2f} M Params")
print(f"5. LM Head:          {lm_head_params/1e6:.2f} M Params")
print(f"-" * 30)
print(f"Total Params:        {total_params_count/1e6:.2f} M")
print(f"Original Size (BF16): {bf16_size_mb:.2f} MB (~0.93 GB)")
print(f"Accelerated Size (INT8): {int8_size_mb:.2f} MB")
print(f"-" * 30)
print(f"Peak Activation (On-Chip): {max(vis_act_peak, vlm_act_peak, act_act_peak)/1e6:.2f} MB")
print(f"Action Context KV Cache:   {ctx_kv_size/1e3:.2f} KB")

