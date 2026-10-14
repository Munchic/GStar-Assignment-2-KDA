"""Checker-only launch harnesses; contains no attention or inverse solution."""
import torch
import triton
import triton.language as tl
from problem_6 import triangular_inverse


@triton.jit
def _inverse_test(Lower, Out, CHUNK: tl.constexpr):
    tile = tl.program_id(0)
    rows = tl.arange(0, CHUNK)
    offsets = tile * CHUNK * CHUNK + rows[:, None] * CHUNK + rows[None, :]
    lower = tl.load(Lower + offsets)
    inverse = triangular_inverse(lower, CHUNK)
    tl.store(Out + offsets, inverse)


def inverse_forward(lower):
    lower = lower.contiguous()
    out = torch.empty_like(lower)
    _inverse_test[(lower.shape[0],)](lower, out, CHUNK=lower.shape[-1])
    return out
