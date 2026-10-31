"""Problem 6: Chunked DeltaNet and triangular inverse (forward only).

Build L = strictly_lower(diag(beta) K K.T), then A = (I + L)^-1.
Compute W/U, pseudo-values, and chunk boundaries. Reuse P3 for the output.
The inverse is a shared STUDENT primitive reused by P7 and P9.
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
def triangular_inverse(lower, CHUNK: tl.constexpr):
    # (I + lower)^-1 for a strictly lower triangular tile: forward substitution inside INVERSE_BLOCK x INVERSE_BLOCK
    # diagonal blocks (all blocks at once), then merged pairwise with [[A, 0], [C, B]]^-1 = [[A^-1, 0], [-B^-1 C A^-1, B^-1]]
    offsets = tl.arange(0, CHUNK)
    rows, cols = offsets[:, None], offsets[None, :]
    same_block = rows // INVERSE_BLOCK == cols // INVERSE_BLOCK
    block_lower = tl.where(same_block, lower, 0.0)
    inverse = tl.where(rows == cols, 1.0, 0.0)

    for r in range(1, INVERSE_BLOCK):
        # --- STUDENT IMPLEMENTATION REQUIRED: P6.INVERSE.SUBSTITUTE ---
        # Forward-substitute row r in every diagonal block. Update inverse [C,C].
        row_mask = offsets % INVERSE_BLOCK == r
        col_mask = offsets % INVERSE_BLOCK < r
        mask = row_mask[:, None] & col_mask[None, :] & same_block
        inverse = tl.where(mask, -tl.dot(block_lower, inverse), inverse)
        # --- END OF STUDENT IMPLEMENTATION ---

    for level in tl.static_range(6):
        if INVERSE_BLOCK * 2**level < CHUNK:
            merged = rows // (INVERSE_BLOCK * 2 ** (level + 1)) == cols // (INVERSE_BLOCK * 2 ** (level + 1))
            lower_left = (rows // (INVERSE_BLOCK * 2**level) == cols // (INVERSE_BLOCK * 2**level) + 1) & merged

            # --- STUDENT IMPLEMENTATION REQUIRED: P6.INVERSE.MERGE ---
            # Apply the lower-left block inverse identity. Update inverse only in lower_left.
            inverse = tl.where(lower_left, -tl.dot(tl.dot(inverse, lower), inverse), inverse)
            # --- END OF STUDENT IMPLEMENTATION ---

    return inverse


@triton.jit
def _chunk_wy_representation(
    K, V, Beta, W, U, A,
    seq_len,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    batch_head = tl.program_id(1)
    chunk_offsets = tl.arange(0, CHUNK)
    rows = chunk * CHUNK + chunk_offsets
    beta = tl.load(Beta + batch_head * seq_len + rows).to(tl.float32)

    # L = tril(diag(beta) K K^T, -1)
    lower = tl.zeros([CHUNK, CHUNK], tl.float32)

    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))

        # --- STUDENT IMPLEMENTATION REQUIRED: P6.WY.GRAM ---
        # Accumulate beta-weighted key Gram matrix across key tiles.
        lower += beta[:, None] * tl.dot(k, k.trans())
        # --- END OF STUDENT IMPLEMENTATION ---

    # --- STUDENT IMPLEMENTATION REQUIRED: P6.WY.STRICT ---
    # Finish L with STRICT causality.
    mask = chunk_offsets[:, None] > chunk_offsets[None, :]
    lower = tl.where(mask, lower, 0.0)
    # --- END OF STUDENT IMPLEMENTATION ---

    # A = (I + L)^{-1} for the strictly lower triangular L
    inverse = triangular_inverse(lower, CHUNK)

    tl.store(tile_ptrs(A, batch_head * seq_len, rows, chunk_offsets, CHUNK), inverse)

    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))

        # --- STUDENT IMPLEMENTATION REQUIRED: P6.WY.W ---
        # Set w [C,BLOCK_K] using A and beta-weighted keys.
        w = tl.dot(inverse, beta[:, None] * k)
        # --- END OF STUDENT IMPLEMENTATION ---

        tl.store(tile_ptrs(W, batch_head * seq_len, rows, key_offsets, HEAD_DIM), w.to(W.dtype.element_ty))

    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))

        # --- STUDENT IMPLEMENTATION REQUIRED: P6.WY.U ---
        # Set u [C,BLOCK_V] using A and beta-weighted values.
        u = tl.dot(inverse, beta[:, None] * v)
        # --- END OF STUDENT IMPLEMENTATION ---

        tl.store(tile_ptrs(U, batch_head * seq_len, rows, value_offsets, VALUE_DIM), u.to(U.dtype.element_ty))


@triton.jit
def _chunk_state_forward(
    K, W, U, States, PseudoValues,
    seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_V: tl.constexpr,
):
    value_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    chunk_offsets = tl.arange(0, CHUNK)

    # States[b, h, i] is the state entering chunk i; States[b, h, 0] holds the initial state on entry
    state_row_base = batch_head * (num_chunks + 1) * HEAD_DIM
    state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))

    for chunk in tl.range(num_chunks, num_stages=1):
        rows = chunk * CHUNK + chunk_offsets
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        w = tl.load(tile_ptrs(W, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        u = tl.load(tile_ptrs(U, batch_head * seq_len, rows, value_offsets, VALUE_DIM))

        # --- STUDENT IMPLEMENTATION REQUIRED: P6.STATE.PSEUDO ---
        # Set pseudo_values [C,BLOCK_V] from u, w and the entry state.
        pseudo_values = u - tl.dot(w, state)
        # --- END OF STUDENT IMPLEMENTATION ---

        tl.store(tile_ptrs(PseudoValues, batch_head * seq_len, rows, value_offsets, VALUE_DIM), pseudo_values.to(PseudoValues.dtype.element_ty))

        # --- STUDENT IMPLEMENTATION REQUIRED: P6.STATE.UPDATE ---
        # Advance the boundary using K.T @ pseudo_values.
        state += tl.dot(k.trans(), pseudo_values)
        # --- END OF STUDENT IMPLEMENTATION ---

        tl.store(tile_ptrs(States, state_row_base + (chunk + 1) * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


from problem_3 import _chunk_output_forward  # Reuse your P3 kernel with pseudo-values.


def _wy_representation(key, value, beta):
    batch_size, num_heads, seq_len, head_dim = key.shape
    value_dim = value.shape[-1]
    w, u = torch.empty_like(key), torch.empty_like(value)
    inverse = torch.empty(batch_size, num_heads, seq_len, CHUNK_SIZE, device=key.device, dtype=torch.float32)

    grid = (seq_len // CHUNK_SIZE, batch_size * num_heads)
    _chunk_wy_representation[grid](
        key, value, beta, w, u, inverse, seq_len,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
    )

    return w, u, inverse


def _chunk_states(key, w, u, initial_state):
    batch_size, num_heads, seq_len, head_dim = key.shape
    value_dim = u.shape[-1]
    num_chunks = seq_len // CHUNK_SIZE
    states = torch.empty(batch_size, num_heads, num_chunks + 1, head_dim, value_dim, device=key.device, dtype=torch.float32)
    states[:, :, 0] = initial_state
    pseudo_values = torch.empty_like(u)

    grid = (value_dim // BLOCK_V, batch_size * num_heads)
    _chunk_state_forward[grid](
        key, w, u, states, pseudo_values, seq_len, num_chunks,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_V=BLOCK_V,
    )

    return states, pseudo_values


@torch.no_grad()
def chunk_deltanet(
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

    seq_len = query.shape[2]
    query, key, value = (pad_sequence(x, CHUNK_SIZE) for x in (query, key, value))
    beta = pad_sequence(beta, CHUNK_SIZE)  # Zero beta = no padded writes.

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

    w, u, inverse = _wy_representation(key, value, beta)

    states, pseudo_values = _chunk_states(key, w, u, initial_state)

    out = torch.empty_like(value)

    grid = (num_chunks, value_dim // BLOCK_V, batch_size * num_heads)
    _chunk_output_forward[grid](
        query, key, pseudo_values, states, out, scale, padded_len, num_chunks, **kernel_args)

    return finish(out[:, :, :seq_len].contiguous(), states[:, :, -1].contiguous(), output_final_state)
