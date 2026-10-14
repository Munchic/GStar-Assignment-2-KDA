"""Problem 2: Recurrent linear attention (forward only).

Keep an FP32 [Dk, BLOCK_V] state. Write k outer v before reading with q.
Implement update and reduction; the supplied loop handles decoding.
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
    Q, K, V, Out, InitialState, FinalState,
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
    out_ptrs = Out + batch_head * seq_len * VALUE_DIM + value_offsets

    state = tl.load(tile_ptrs(InitialState, batch_head * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM))

    for t in range(seq_len):
        q = tl.load(q_ptrs + t * HEAD_DIM).to(tl.float32) * scale
        k = tl.load(k_ptrs + t * HEAD_DIM).to(tl.float32)
        v = tl.load(v_ptrs + t * VALUE_DIM).to(tl.float32)

        # --- STUDENT IMPLEMENTATION REQUIRED: P2.RECURRENT.UPDATE ---
        # Update the [Dk, BLOCK_V] state with the current association.
        # TODO: replace this assertion with your implementation.
        tl.static_assert(False, "P2.RECURRENT.UPDATE: implement this subpart")
        # --- END OF STUDENT IMPLEMENTATION ---

        # --- STUDENT IMPLEMENTATION REQUIRED: P2.RECURRENT.READ ---
        # Set out [BLOCK_V] by reducing the updated state over key channels.
        # TODO: replace this assertion with your implementation.
        tl.static_assert(False, "P2.RECURRENT.READ: implement this subpart")
        # --- END OF STUDENT IMPLEMENTATION ---

        tl.store(out_ptrs + t * VALUE_DIM, out.to(Out.dtype.element_ty))

    tl.store(tile_ptrs(FinalState, batch_head * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


@torch.no_grad()
def recurrent_linear_attention(
    query,
    key,
    value,
    scale=None,
    initial_state=None,
    output_final_state=False,
):
    query, key, value, scale, initial_state, gate, beta = prepare(
        query, key, value, scale, initial_state, None, None,
        gate_kind=None, cuda=True)

    batch_size, num_heads, seq_len, head_dim = query.shape
    value_dim = value.shape[-1]

    out = torch.empty_like(value)
    final_state = torch.empty_like(initial_state)

    grid = (value_dim // BLOCK_V, batch_size * num_heads)
    _recurrent_forward[grid](
        query, key, value, out, initial_state, final_state, scale, seq_len,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, BLOCK_V=BLOCK_V)

    return finish(out, final_state, output_final_state)
