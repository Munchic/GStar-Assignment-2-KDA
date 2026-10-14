"""Supplied tensor plumbing. No attention arithmetic is implemented here.

P1 needs only PyTorch; Triton is optional until a GPU problem is imported.
All kernels assume contiguous [B,H,T,D] tensors with 64/128 key/value dims.
"""

import math
import torch
import torch.nn.functional as F

CHUNK_SIZE = 64
SUB_CHUNK = 16
BLOCK_K = BLOCK_V = 64
INVERSE_BLOCK = 16

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
else:
    INVERSE_BLOCK = tl.constexpr(16)

    @triton.jit
    def tile_ptrs(base, row_base, rows, cols, row_stride):
        return base + (row_base + rows[:, None]) * row_stride + cols[None, :]


def prepare(query, key, value, scale=None, initial_state=None, gate=None,
            beta=None, gate_kind=None, cuda=False):
    """Validate and normalize layout, without modifying caller-owned tensors.

    Q/K/V must share a floating dtype/device. Gates and beta may use another
    floating dtype (FP32 recommended). No normalization is performed here.
    P1 permits arbitrary positive Dk/Dv; GPU kernels support 64 or 128.
    """
    if any(x.ndim != 4 for x in (query, key, value)):
        raise ValueError("Q/K/V must be [B,H,T,D]")
    b, h, t, dk = query.shape
    dv = value.shape[-1]
    if min(b, h, t, dk, dv) <= 0:
        raise ValueError("All tensor dimensions must be positive")
    if key.shape != query.shape or value.shape[:3] != query.shape[:3]:
        raise ValueError("Q/K must match; V must share B/H/T")
    if not query.is_floating_point() or any(
            x.dtype != query.dtype or x.device != query.device for x in (key, value)):
        raise ValueError("Q/K/V must share a floating dtype and device")
    if cuda:
        if not query.is_cuda:
            raise ValueError("Problems 2-9 require CUDA; use Problem 1 on CPU")
        if dk not in (64, 128) or dv not in (64, 128):
            raise ValueError("GPU assignment supports Dk and Dv independently in {64,128}")
        if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("GPU assignment supports FP16, BF16 and FP32")
    for name, x, shape in (
        ("gate", gate, (b, h, t) if gate_kind == "scalar" else query.shape),
        ("beta", beta, (b, h, t)),
    ):
        if x is not None and (tuple(x.shape) != tuple(shape)
                              or x.device != query.device or not x.is_floating_point()):
            raise ValueError(f"{name} must have shape {tuple(shape)} and match the input device")
    if gate_kind is not None and gate is None:
        raise ValueError("This operator requires a log gate")
    if gate is not None and (not torch.isfinite(gate).all() or (gate > 0).any()):
        raise ValueError("Log gates must be finite and <= 0")
    if beta is not None and (not torch.isfinite(beta).all() or (beta < 0).any() or (beta > 1).any()):
        raise ValueError("beta must be finite and in [0,1]")
    scale = dk ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite (zero is supported)")
    if initial_state is None:
        initial_state = torch.zeros(b, h, dk, dv, device=query.device, dtype=torch.float32)
    elif (tuple(initial_state.shape) != (b, h, dk, dv)
          or initial_state.device != query.device or not initial_state.is_floating_point()):
        raise ValueError("initial_state must be floating [B,H,Dk,Dv] on the input device")
    # Copy state: students may update a local view in P1 without changing input.
    state_dtype = torch.float32 if cuda or query.dtype != torch.float64 else torch.float64
    return (query.contiguous(), key.contiguous(), value.contiguous(), scale,
            initial_state.to(state_dtype).contiguous().clone(),
            None if gate is None else gate.contiguous(),
            None if beta is None else beta.contiguous())


def pad_sequence(x, block=CHUNK_SIZE):
    """Pad token axis 2 with zeros, including [B,H,T] gates/beta."""
    padding = -x.shape[2] % block
    return F.pad(x, [0, 0] * (x.ndim - 3) + [0, padding]).contiguous()


def allocate_states(key, value, initial_state):
    b, h, t, dk = key.shape
    states = torch.empty(b, h, t // CHUNK_SIZE + 1, dk, value.shape[-1],
                         device=key.device, dtype=torch.float32)
    states[:, :, 0] = initial_state
    return states


def finish(out, final, output_final_state):
    return (out, final) if output_final_state else out
