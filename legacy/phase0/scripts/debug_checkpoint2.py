
import torch, sys
sys.path.insert(0, ".")
from thought_vectors import ThoughtEncoder

enc = ThoughtEncoder(16384, 256, 4, 4, 0.1, 512, 256)
sd = enc.state_dict()
print(f"Encoder state dict keys (first 8):")
for k in list(sd.keys())[:8]:
    print(f"  {k}")

from thought_vectors import ThoughtDecoder
dec = ThoughtDecoder(16384, 256, 4, 4, 0.1, 512)
sd2 = dec.state_dict()
print(f"\nDecoder state dict keys (first 8):")
for k in list(sd2.keys())[:8]:
    print(f"  {k}")
