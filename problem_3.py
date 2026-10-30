"""Problem 3: Chunked linear attention (forward only).

First scan chunk boundaries, then process outputs in parallel.
States[:, :, i] is the ENTRY state for chunk i. Local scores include the diagonal.
Accumulate across every key tile; Dk and Dv are independent.
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
def _chunk_state_forward(
    K, V, States,
    seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    key_block = tl.program_id(0)
    value_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    chunk_offsets = tl.arange(0, CHUNK)

    # States[b, h, i] is the state entering chunk i; States[b, h, 0] holds the initial state on entry
    state_row_base = batch_head * (num_chunks + 1) * HEAD_DIM
    state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))

    for chunk in range(num_chunks):
        rows = chunk * CHUNK + chunk_offsets
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))

        # --- STUDENT IMPLEMENTATION REQUIRED: P3.STATE.UPDATE ---
        # Accumulate K.T @ V into the FP32 chunk boundary state.
        state += tl.dot(k.trans(), v)
        # --- END OF STUDENT IMPLEMENTATION ---

        tl.store(tile_ptrs(States, state_row_base + (chunk + 1) * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


@triton.jit
def _chunk_output_forward(
    Q, K, V, States, Out,
    scale: tl.constexpr, seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    value_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    chunk_offsets = tl.arange(0, CHUNK)
    rows = chunk * CHUNK + chunk_offsets
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    state_row_base = (batch_head * (num_chunks + 1) + chunk) * HEAD_DIM

    v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
    inter = tl.zeros([CHUNK, BLOCK_V], tl.float32)
    scores = tl.zeros([CHUNK, CHUNK], tl.float32)

    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)) * scale
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))

        # --- STUDENT IMPLEMENTATION REQUIRED: P3.OUTPUT.PRODUCTS ---
        # Accumulate inter [C,BLOCK_V] and scores [C,C] across key tiles.
        inter += tl.dot(q, state)
        scores += tl.dot(q, k.trans())
        # --- END OF STUDENT IMPLEMENTATION ---

    # --- STUDENT IMPLEMENTATION REQUIRED: P3.OUTPUT.MASK ---
    # Apply an inclusive lower-triangular mask; masked scores are ZERO.
    local = tl.arange(0, CHUNK)
    mask = local[:, None] >= local[None, :]
    scores = tl.where(mask, scores, 0.0)

    # --- END OF STUDENT IMPLEMENTATION ---

    # --- STUDENT IMPLEMENTATION REQUIRED: P3.OUTPUT.COMBINE ---
    # Combine inter-chunk reads and intra-chunk weighted values.
    out = inter + tl.dot(scores, v)
    # --- END OF STUDENT IMPLEMENTATION ---

    # problem 3.4
    # current state still depends on previous state even if we chunk (but it's just faster)
    # between chunks, the output calculations are all independent so each chunk can run its own independent program

    tl.store(tile_ptrs(Out, batch_head * seq_len, rows, value_offsets, VALUE_DIM), out.to(Out.dtype.element_ty))


def _chunk_states(key, value, initial_state):
    batch_size, num_heads, seq_len, head_dim = key.shape
    value_dim = value.shape[-1]
    num_chunks = seq_len // CHUNK_SIZE
    states = torch.empty(batch_size, num_heads, num_chunks + 1, head_dim, value_dim, device=key.device, dtype=torch.float32)
    states[:, :, 0] = initial_state

    grid = (head_dim // BLOCK_K, value_dim // BLOCK_V, batch_size * num_heads)
    _chunk_state_forward[grid](
        key, value, states, seq_len, num_chunks,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
    )

    return states


@torch.no_grad()
def chunk_linear_attention(
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

    seq_len = query.shape[2]
    query, key, value = (pad_sequence(x, CHUNK_SIZE) for x in (query, key, value))

    batch_size, num_heads, padded_len, head_dim = query.shape
    value_dim = value.shape[-1]
    num_chunks = padded_len // CHUNK_SIZE
    kernel_args = dict(
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        CHUNK=CHUNK_SIZE,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
    )

    states = _chunk_states(key, value, initial_state)

    out = torch.empty_like(value)

    grid = (num_chunks, value_dim // BLOCK_V, batch_size * num_heads)
    _chunk_output_forward[grid](
        query, key, value, states, out, scale, padded_len, num_chunks, **kernel_args)

    return finish(out[:, :, :seq_len].contiguous(), states[:, :, -1].contiguous(), output_final_state)
