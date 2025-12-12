
# Vision Encoder Parameters
V_LAYERS = 12
V_L = 1024
V_D = 768
V_FFN = 3072
# Standard MHA: Q,K,V,Out all D->D. 
# MHA Cost: 4 * L * D^2 (Projections) + 2 * L^2 * D (Attn)
# FFN Cost: 2 * L * D * FFN + L * D * FFN = 3? No, usually 2 matrices (Up, Down). 
# ViT usually 2 layers: D->FFN, FFN->D. 2 * L * D * FFN.
# Total Linear MACs: 4*L*D^2 + 2*L*D*FFN
# Total Attn MACs: 2*L^2*D

def calc_vision_macs():
    proj_macs = 4 * V_L * (V_D**2)
    attn_macs = 2 * (V_L**2) * V_D
    mlp_macs = 2 * V_L * V_D * V_FFN
    layer_macs = proj_macs + attn_macs + mlp_macs
    total_macs = V_LAYERS * layer_macs
    return total_macs, (V_LAYERS * attn_macs) / total_macs

# VLM Backbone Parameters
T_LAYERS = 16
T_L = 113
T_D = 960
T_FFN = 2560
# GQA: Q(D->D), K(D->D/3), V(D->D/3), Out(D->D)
# Heads: 12Q, 4KV. Ration 1/3.
# Proj MACs: L*D^2 * (1 + 1/3 + 1/3 + 1) = L*D^2 * (2.666) -> 8/3 * L * D^2
# FFN: Gated often implies 3 matrices (Gate, Up, Down). Llama/Mistral use 3. SmolVLA uses 3 (Gate, Up, Down)?
# test_mlp_kernels.py showed gate_proj, up_proj, down_proj. So 3 matrices.
# FFN Cost: 3 * L * D * FFN.

def calc_vlm_macs():
    # Projections: Q, Out (Full), K, V (1/3)
    proj_macs = T_L * (T_D**2) * (1 + 1 + 1/3 + 1/3)
    attn_macs = 2 * (T_L**2) * T_D # Approx standard attn cost is L^2 * D regardless of heads?
    # Attn cost: Hq * L^2 * (D/Hq) = L^2 * D. Yes.
    
    # MLP (SwiGLU style = 3 matrices)
    mlp_macs = 3 * T_L * T_D * T_FFN
    
    layer_macs = proj_macs + attn_macs + mlp_macs
    total_macs = T_LAYERS * layer_macs
    return total_macs

# Action Expert Parameters
A_LAYERS = 16
A_L = 50 # Action tokens
C_L = 113 # Context tokens
A_D = 720
A_FFN = 2048 # From test_mlp_kernels.py (720, 2048)
STEPS = 10

# GQA Params: 12Q, 4KV. 
# Self Attn (Dynamic 10x):
#   Q,K,V,Out: Same as VLM but D=720. 
#   Attn: L_a^2 * D
# Cross Attn (Dynamic 10x):
#   Q (Action): L_a * D^2
#   K, V (Context): L_c * VLM_D * Exprt_KV_D ?? 
#   Wait, Cross Attn K/V comes from VLM. 
#   Input to Cross Attn K/V proj is VLM embeddings (D=960). 
#   Output is Action Expert KV Dim (Heads=4 * HeadDim=60 = 240).
#   So K_proj: 960 -> 240. V_proj: 960 -> 240.
#   Optimization: Computed ONCE per inference (Static).
#   Q_proj: 720 -> 720 (Dynamic 10x).
#   Out_proj: 720 -> 720 (Dynamic 10x).
#   Attn: L_a * L_c * D (Dynamic 10x).

def calc_action_macs():
    # 1. Self Attention (Per Step)
    sa_proj = A_L * (A_D**2) * (1 + 1 + 1/3 + 1/3)
    sa_attn = 2 * (A_L**2) * A_D
    sa_total_per_step = sa_proj + sa_attn
    
    # 2. Cross Attention
    # Static Part (Once per inference)
    # K, V Proj: L_c * 960 * 240. (Two matrices)
    ca_static = 2 * C_L * 960 * 240
    
    # Dynamic Part (Per Step)
    # Q Proj: 720->720
    # Out Proj: 720->720
    # Attn: L_a * L_c * D (Score) + L_a * L_c * D (Agg) = 2 * L_a * L_c * D? 
    #   Verify dims: Q(La, D), K(Lc, D). QK^T(La, Lc). * V(Lc, D) -> (La, D).
    #   Yes, 2 * La * Lc * D. (Assuming D is effective D, typically sum of heads).
    ca_dynamic_per_step = (A_L * A_D**2) + (A_L * A_D**2) + (2 * A_L * C_L * A_D)
    
    # 3. MLP (Per Step)
    # 3 matrices (Gate, Up, Down)
    mlp_per_step = 3 * A_L * A_D * A_FFN
    
    # Total Action Expert
    total_per_step = (sa_total_per_step + ca_dynamic_per_step + mlp_per_step)
    total_macs = (A_LAYERS * total_per_step * STEPS) + (A_LAYERS * ca_static)
    
    return total_macs

vision, vis_attn_share = calc_vision_macs()
vlm = calc_vlm_macs()
action = calc_action_macs()

total = vision + vlm + action

print(f"Vision Encoder: {vision/1e9:.2f} G-MACs")
print(f"VLM Backbone:   {vlm/1e9:.2f} G-MACs")
print(f"Action Expert:  {action/1e9:.2f} G-MACs")
print(f"Total:          {total/1e9:.2f} G-MACs")

print("-" * 20)
print(f"Vision %: {vision/total*100:.1f}%")
print(f"VLM %:    {vlm/total*100:.1f}%")
print(f"Action %: {action/total*100:.1f}%")

print("-" * 20)
# Ops = 2 * MACs
print(f"Total FLOPs: {total*2/1e9:.2f} G-FLOPs")
