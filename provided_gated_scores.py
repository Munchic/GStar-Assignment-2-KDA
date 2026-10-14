"""Temporary direct FP32 PyTorch baseline for P4; replace with P8 gated_scores.

Uses bounded chunk-local tensors, masks before exp, and contains no secondary
tiling/anchor solution. This is the sole supplied attention arithmetic baseline.
"""

import torch
from common import CHUNK_SIZE


@torch.no_grad()
def direct_gated_scores(left, key, gate_cumsum, scale=1.0, strict=False):
    b, h, t, _ = key.shape
    if t % CHUNK_SIZE:
        raise ValueError("The baseline requires padded inputs")
    out = torch.zeros(b, h, t, CHUNK_SIZE, device=key.device, dtype=torch.float32)
    rows = torch.arange(CHUNK_SIZE, device=key.device)
    causal = rows[:, None] > rows[None, :] if strict else rows[:, None] >= rows[None, :]
    for start in range(0, t, CHUNK_SIZE):
        l, k, g = (x[:, :, start:start + CHUNK_SIZE].float()
                   for x in (left, key, gate_cumsum))
        diff = g.unsqueeze(-2) - g.unsqueeze(-3)
        decay = torch.where(causal[..., None], diff, -torch.inf).exp()
        scores = (l.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(-1)
        out[:, :, start:start + CHUNK_SIZE] = scores * scale
    return out
