"""P9.4 segmented-execution demonstration: split at token 65, carry state, compare."""
import torch
from problem_9 import chunk_kimi_delta_attention

torch.manual_seed(2026)
B, H, T, Dk, Dv = 1, 1, 129, 128, 64
q = torch.randn(B, H, T, Dk, device="cuda") * 0.2
k = torch.randn_like(q)
k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
v = torch.randn(B, H, T, Dv, device="cuda") * 0.2
gate = -torch.rand(B, H, T, Dk, device="cuda") * 0.1   # unequal channel gates
beta = torch.rand(B, H, T, device="cuda")
s0 = torch.randn(B, H, Dk, Dv, device="cuda") * 0.05    # nonzero initial memory

out_full, state_full = chunk_kimi_delta_attention(q, k, v, gate, beta,
                                                  initial_state=s0, output_final_state=True)

split = 65  # interior token, not a chunk boundary
out_a, state_a = chunk_kimi_delta_attention(q[:, :, :split], k[:, :, :split], v[:, :, :split],
                                            gate[:, :, :split], beta[:, :, :split],
                                            initial_state=s0, output_final_state=True)
out_b, state_b = chunk_kimi_delta_attention(q[:, :, split:], k[:, :, split:], v[:, :, split:],
                                            gate[:, :, split:], beta[:, :, split:],
                                            initial_state=state_a, output_final_state=True)

out_cat = torch.cat((out_a, out_b), dim=2)
print(f"out max_abs = {(out_cat - out_full).abs().max().item():.3e}")
print(f"final state max_abs = {(state_b - state_full).abs().max().item():.3e}")
