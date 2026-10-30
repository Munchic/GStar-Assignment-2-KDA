"""Problem 5: Recurrent DeltaNet (forward only).

Predict values with the previous state, form a residual, weight by beta,
write the correction, then read the updated state. Keys are already normalized.
See the matching handout section for derivation, points, and checks.
Delete each tl.static_assert only when you implement its TODO.
"""

import torch
import triton
import triton.language as tl

from common import (
    CHUNK_SIZE,
    BLOCK_K,
    BLOCK_V,
    SUB_CHUNK,
    INVERSE_BLOCK,
    tile_ptrs,
    prepare,
    pad_sequence,
    allocate_states,
    finish,
)


@triton.jit
def _recurrent_forward(
    Q, K, V, Beta, Out, InitialState, FinalState,
    scale: tl.constexpr, seq_len,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, BLOCK_V: tl.constexpr,
):
    value_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)

    q_ptrs = Q + batch_head * seq_len * HEAD_DIM + key_offsets
    k_ptrs = K + batch_head * seq_len * HEAD_DIM + key_offsets
    v_ptrs = V + batch_head * seq_len * VALUE_DIM + value_offsets
    beta_ptrs = Beta + batch_head * seq_len
    out_ptrs = Out + batch_head * seq_len * VALUE_DIM + value_offsets

    state = tl.load(tile_ptrs(InitialState, batch_head * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM))

    for t in range(seq_len):
        q = tl.load(q_ptrs + t * HEAD_DIM).to(tl.float32) * scale
        k = tl.load(k_ptrs + t * HEAD_DIM).to(tl.float32)
        v = tl.load(v_ptrs + t * VALUE_DIM).to(tl.float32)
        beta = tl.load(beta_ptrs + t).to(tl.float32)

        # --- STUDENT IMPLEMENTATION REQUIRED: P5.RECURRENT.RESIDUAL ---
        # Set error [BLOCK_V] using the previous state.
        v_pred = tl.sum(state * k[:, None], axis=0)
        # --- END OF STUDENT IMPLEMENTATION ---

        # --- STUDENT IMPLEMENTATION REQUIRED: P5.RECURRENT.WRITE ---
        # Apply the beta-weighted rank-one correction.
        u = beta * (v - v_pred)
        # --- END OF STUDENT IMPLEMENTATION ---

        # --- STUDENT IMPLEMENTATION REQUIRED: P5.RECURRENT.READ ---
        # Set out using the updated state.
        state += k[:, None] * u[None, :]
        out = tl.sum(q[:, None] * state, axis=0) 
        # --- END OF STUDENT IMPLEMENTATION ---

        # problem 5.3
        # the normalized keys help deal with state accumulation in linear attention
        # higher beta (closer to 1) means we keep more of the new value v_t (replace)
        # lower beta (closer to 0) means we keep more of our old memory (retain)

        tl.store(out_ptrs + t * VALUE_DIM, out.to(Out.dtype.element_ty))

    tl.store(tile_ptrs(FinalState, batch_head * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


@torch.no_grad()
def recurrent_deltanet(
    query,
    key,
    value,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
):
    query, key, value, scale, initial_state, gate, beta = prepare(
        query, key, value, scale, initial_state, None, beta,
        gate_kind=None, cuda=True)

    batch_size, num_heads, seq_len, head_dim = query.shape
    value_dim = value.shape[-1]

    out = torch.empty_like(value)
    final_state = torch.empty_like(initial_state)

    grid = (value_dim // BLOCK_V, batch_size * num_heads)
    _recurrent_forward[grid](
        query, key, value, beta, out, initial_state, final_state, scale, seq_len,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, BLOCK_V=BLOCK_V)

    return finish(out, final_state, output_final_state)
