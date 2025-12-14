
# Architecture Parameters

# 1. Vision Encoder (SigLIP-like / ViT-Base)
V_LAYERS = 12
V_D = 768
V_FFN = 3072
# Parameters:
# Patch Embed: (16*16*3) * 768 + 768 bias
# Layers: 12 * [ (4*D^2 + 4*D) {Attn} + (2*D*FFN + D + FFN) {MLP} + 4*D {Norms} ]
# No classification head (it's an encoder).
# Connector: 12288 -> 960 (Weight + Bias)

def calc_vision_params():
    patch_embed = (16*16*3) * V_D + V_D
    layer = 4*(V_D**2 + V_D) + (2*V_D*V_FFN + V_D + V_FFN) + 4*V_D
    # Connector (12288 -> 960)
    connector = 12288 * 960 + 960
    return patch_embed + V_LAYERS * layer + connector

# 2. VLM Backbone (SmolLM2-360M)
T_LAYERS = 16
T_D = 960
T_FFN = 2560
T_VOCAB = 49280
# Layers: 16 * [ (4*D^2) {Attn - approx GQA as dense for params usually?} 
# Wait, GQA means K/V are smaller.
# Q: D*D. K: D*(D/3). V: D*(D/3). Out: D*D.
# Simplify: Attn Params = D*D + 2*D*(D/3) + D*D = 2*D^2 + 2/3*D^2 = 2.66*D^2.
# Let's use exact heads: 15 Q, 5 KV.
# Q: 960*960. K: 960*320. V: 960*320. Out: 960*960.
# MLP: Up(D->FFN), Gate(D->FFN), Down(FFN->D).
# Embeddings: Vocab * D.
# LM Head: D * Vocab (tied? usually yes, but let's count once).

def calc_vlm_params():
    # Attn
    q = 960 * 960
    k = 960 * 320
    v = 960 * 320
    o = 960 * 960
    attn = q + k + v + o + 4*960 # biases
    
    # MLP (SwiGLU -> 3 matrices)
    gate = 960 * 2560
    up = 960 * 2560
    down = 2560 * 960
    mlp = gate + up + down + 3*960 # biases?
    
    # Norms
    norm = 2 * 960 # RMSNorm (1 per layer + post)? Usually 2 per layer (pre-attn, pre-mlp).
    
    layer = attn + mlp + norm
    
    embed = T_VOCAB * T_D
    # LM Head (often tied with embed, but if not...)
    # SmolLM usually ties weights. Let's assume tied (0 extra params).
    
    return embed + T_LAYERS * layer

# 3. Action Expert
A_LAYERS = 16
A_D = 720
A_FFN = 2048
# Heads: 12 Q, 4 KV. Head Dim 80.
# Q: 720 -> 12*80=960.  Matrix [960, 720]
# K: 720 -> 4*80=320.   Matrix [320, 720]
# V: 720 -> 320.        Matrix [320, 720]
# Out: 960 -> 720.      Matrix [720, 960]
#
# Cross Attention:
# Q: 720 -> 960.
# K, V: 960(VLM) -> 320. Matrix [320, 960].
# Out: 960 -> 720.

def calc_action_params():
    # Self Attn
    sa_q = 960 * 720
    sa_k = 320 * 720
    sa_v = 320 * 720
    sa_o = 720 * 960
    sa = sa_q + sa_k + sa_v + sa_o
    
    # Cross Attn
    ca_q = 960 * 720
    # K/V Projections (VLM -> KV). 
    # Computed ONCE globally or per layer?
    # model_shape.txt showed `layers.1.self_attn.k_proj` as `[320, 320]`.
    # AND `[320, 960]` weights existed.
    # If `[320, 320]` exists, it means valid params.
    # Check if `[320, 960]` are shared or per layer.
    # Let's assuming per layer for worst case safely, or check if shared.
    # The user said "efficient KV reuse... computed only once per inference".
    # But ARE THE WEIGHTS shared?
    # Usually CrossAttn weights are per layer.
    # If the *projection result* is reused, the weights are still there.
    # So we count them:
    ca_k = 320 * 320 # Based on `layers.odd.k_proj` [320, 320]
    ca_v = 320 * 320 # Based on similar logic?
    # But wait, we saw `[320, 960]` in the grep.
    # If `[320, 960]` exists, those are parameters.
    # If they are inside the VLM, we counted them.
    # If they are for CA, we count them.
    # Let's assume CA uses `[320, 320]` as seen in `model_shape` for `k_proj`.
    # And maybe an adapter `[320, 960]` exists?
    # Let's assume standard Cross Attn: Query(720->960), Key(Context->320), Value(Context->320), Out(960->720).
    # If Context is projected to 320 first (Adapter), then Key is 320->320.
    # Let's count SA + CA parameters.
    # CA Q: 960*720. CA Out: 720*960.
    # CA K/V: 320*320 + 320*320 (from model_shape layer 1).
    ca = ca_q + 720*960 + 320*320 + 320*320
    
    # MLP
    gate = 720 * 2048
    up = 720 * 2048
    down = 2048 * 720
    mlp = gate + up + down
    
    # Layer: SA + CA + MLP ?
    # Alternating? 
    # model_shape.txt showed `layers.0` (SA), `layers.1` (CA/SA?).
    # Report says "Alternates between Self-Attention and Cross-Attention".
    # So 8 SA layers, 8 CA layers?
    # Or 16 layers each having both?
    # "Alternates... where the latter attends to the VLM".
    # Usually: Layer N: SA -> CA -> MLP. (Standard Decoder).
    # But report says "Alternates".
    # If 16 layers total. 8 SA, 8 CA?
    # SmolVLA paper says "16 layers... alternates".
    # Let's assume 16 blocks, each has SA and CA?
    # Or 8 blocks (SA+CA)?
    # User's text: "16-layer architecture that alternates... odd-numbered Cross-Attention".
    # This implies Layers 0, 2, 4... are SA. Layers 1, 3, 5... are CA.
    # So 8 SA layers, 8 CA layers.
    
    # Parameters for 8 SA layers:
    params_sa = 8 * (sa + mlp + 4*720) # Norms
    # Parameters for 8 CA layers:
    # Do CA layers have SA too?
    # "Alternates between SA and CA". implies pure alternation.
    # If Layer 1 is CA, does it have SA?
    # `model_shape.txt` for `layers.1` showed `self_attn`.
    # And `k_proj` was `[320, 320]`.
    # Is it possible `layers.1` IS the Cross Attention layer, but named `self_attn` in code?
    # Yes.
    # So 8 layers of Type SA, 8 layers of Type CA.
    params_ca = 8 * (ca + mlp + 4*720)
    
    # Embeddings?
    # Action In/Out Proj:
    # In: 720 * 6 (Action dim?) or similar. Negligible.
    # Time MLP: Small.
    
    return params_sa + params_ca

v = calc_vision_params()
t = calc_vlm_params()
a = calc_action_params()

total = v + t + a

print(f"Vision: {v/1e6:.2f} M")
print(f"VLM: {t/1e6:.2f} M")
print(f"Action: {a/1e6:.2f} M")
print(f"Total Params: {total/1e6:.2f} M")
print(f"BF16 Size: {total*2/1024/1024:.2f} MB")
print(f"INT8 Size: {total*1/1024/1024:.2f} MB")
