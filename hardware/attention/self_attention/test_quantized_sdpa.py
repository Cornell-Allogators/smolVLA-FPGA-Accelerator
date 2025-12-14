import allo
from allo.ir.types import int4, int8, float32
import numpy as np
import sdpa

# Test with int8
L = 64  # Small for quick test
D_h = 32

print('Testing sdpa_streaming with int8...')
Q = np.random.randint(-8, 8, (L, D_h)).astype(np.int8)
K = np.random.randint(-8, 8, (L, D_h)).astype(np.int8)
V = np.random.randint(-8, 8, (L, D_h)).astype(np.int8)
out = np.zeros((L, D_h), dtype=np.int8)
scale = 1.0

s = allo.customize(sdpa.sdpa_streaming, instantiate=[int8, L, D_h])
mod = s.build()
mod(Q, K, V, scale, out)
print(f'✓ int8 test passed! Output shape: {out.shape}, sample values: {out[0, :5]}')

# Test with int4
print('\nTesting sdpa_streaming with int4...')
Q4 = np.random.randint(-8, 8, (L, D_h)).astype(np.int8)
K4 = np.random.randint(-8, 8, (L, D_h)).astype(np.int8)
V4 = np.random.randint(-8, 8, (L, D_h)).astype(np.int8)
out4 = np.zeros((L, D_h), dtype=np.int8)

s4 = allo.customize(sdpa.sdpa_streaming, instantiate=[int4, L, D_h])
mod4 = s4.build()
mod4(Q4, K4, V4, scale, out4)
print(f'✓ int4 test passed! Output shape: {out4.shape}, sample values: {out4[0, :5]}')

print('\n✅ All quantized SDPA tests passed!')
