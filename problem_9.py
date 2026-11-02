"""Problem 9: Chunked Kimi Delta Attention (forward only).

Compose P4 gate scan, P6 inverse, and P8 gated score builder.
Reduce W @ state over ALL key tiles before updating ANY state tile.
Output scale applies to query reads/scores only, never to the WY system.
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
def _chunk_wy_representation(
    K, V, GateCumsum, Beta, KernelMatrix, W, U, A,
    seq_len,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    batch_head = tl.program_id(1)
    chunk_offsets = tl.arange(0, CHUNK)
    rows = chunk * CHUNK + chunk_offsets
    beta = tl.load(Beta + batch_head * seq_len + rows).to(tl.float32)

    # A = (I + L)^{-1} for the strictly lower triangular L = diag(beta) P

    # --- STUDENT IMPLEMENTATION REQUIRED: P9.WY.LOWER ---
    # Load the supplied STRICT key-key matrix and beta-weight rows to set lower.
    G_kk = tl.load(tile_ptrs(KernelMatrix, batch_head * seq_len, rows, chunk_offsets, CHUNK)).to(tl.float32)
    lower = beta[:, None] * G_kk
    # --- END OF STUDENT IMPLEMENTATION ---

    inverse = triangular_inverse(lower, CHUNK)

    tl.store(tile_ptrs(A, batch_head * seq_len, rows, chunk_offsets, CHUNK), inverse)

    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        gate_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows, key_offsets, HEAD_DIM))

        # --- STUDENT IMPLEMENTATION REQUIRED: P9.WY.W ---
        # Set w using A, beta and channel-wise entry-decayed keys.
        k_start = k * tl.exp(gate_cumsum)
        w = tl.dot(inverse, beta[:, None] * k_start)
        # --- END OF STUDENT IMPLEMENTATION ---

        tl.store(tile_ptrs(W, batch_head * seq_len, rows, key_offsets, HEAD_DIM), w.to(W.dtype.element_ty))

    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))

        # --- STUDENT IMPLEMENTATION REQUIRED: P9.WY.U ---
        # Set u using A and beta-weighted values.
        u = tl.dot(inverse, beta[:, None] * v)
        # --- END OF STUDENT IMPLEMENTATION ---

        tl.store(tile_ptrs(U, batch_head * seq_len, rows, value_offsets, VALUE_DIM), u.to(U.dtype.element_ty))


@triton.jit
def _chunk_state_forward(
    K, W, U, GateCumsum, States, PseudoValues,
    seq_len, num_chunks,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    value_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    chunk_offsets = tl.arange(0, CHUNK)

    # States[b, h, i] is the state entering chunk i; States[b, h, 0] holds the initial state on entry.
    # The state is read back from States one key block at a time so every tile is (CHUNK x BLOCK) whatever the head dim.
    for chunk in range(num_chunks):
        rows = chunk * CHUNK + chunk_offsets
        state_row_base = (batch_head * (num_chunks + 1) + chunk) * HEAD_DIM
        pseudo_values = tl.load(tile_ptrs(U, batch_head * seq_len, rows, value_offsets, VALUE_DIM)).to(tl.float32)

        for key_block in range(HEAD_DIM // BLOCK_K):
            key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
            w = tl.load(tile_ptrs(W, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
            state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))

            # --- STUDENT IMPLEMENTATION REQUIRED: P9.STATE.REDUCE ---
            # Subtract this key tile contribution from pseudo_values. Finish ALL key tiles before the update loop.
            pseudo_values -= tl.dot(w, state)
            # --- END OF STUDENT IMPLEMENTATION ---

        tl.store(tile_ptrs(PseudoValues, batch_head * seq_len, rows, value_offsets, VALUE_DIM), pseudo_values.to(PseudoValues.dtype.element_ty))

        for key_block in range(HEAD_DIM // BLOCK_K):
            key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
            k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
            gate_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
            chunk_decay = tl.load(GateCumsum + (batch_head * seq_len + chunk * CHUNK + CHUNK - 1) * HEAD_DIM + key_offsets)
            state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))

            # --- STUDENT IMPLEMENTATION REQUIRED: P9.STATE.UPDATE ---
            # Decay this entry-state tile and add exit-decayed corrected writes.
            k_end = k * tl.exp(chunk_decay - gate_cumsum)
            state = tl.exp(chunk_decay)[:, None] * state + tl.dot(k_end.trans(), pseudo_values)
            # --- END OF STUDENT IMPLEMENTATION ---

            tl.store(tile_ptrs(States, state_row_base + HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


@triton.jit
def _chunk_output_forward(
    Q, GateCumsum, PseudoValues, States, Scores, Out,
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

    pseudo_values = tl.load(tile_ptrs(PseudoValues, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
    scores = tl.load(tile_ptrs(Scores, batch_head * seq_len, rows, chunk_offsets, CHUNK))

    # --- STUDENT IMPLEMENTATION REQUIRED: P9.OUTPUT.LOCAL ---
    # Set out from the already-scaled inclusive scores and pseudo-values.
    out = tl.dot(scores, pseudo_values)
    # --- END OF STUDENT IMPLEMENTATION ---

    for key_block in tl.range(HEAD_DIM // BLOCK_K, num_stages=1):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)) * scale
        gate_cumsum = tl.load(tile_ptrs(GateCumsum, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))

        # --- STUDENT IMPLEMENTATION REQUIRED: P9.OUTPUT.ENTRY ---
        # Add the channel-decayed entry-state read. q already includes scale.
        q_start = q * tl.exp(gate_cumsum)
        out += tl.dot(q_start, state)
        # --- END OF STUDENT IMPLEMENTATION ---

    # problem 9.4
    # in terms of dependencies, the WY can be calculated without knowing entry state
    # while the outputs and next state must wait for the entry state (bcs of query and pseudo values needing entry state)
    # for segmented execution for an interior token, we only need the previous segment's final state
    # which we can use as the current segment's initial state to do decays/outputs and check against the chunking way
    # running problem_9.4.py, i can verify the chunk execution correctedness:
    # out max_abs = 1.821e-05
    # final state max_abs = 8.265e-05
    
    tl.store(tile_ptrs(Out, batch_head * seq_len, rows, value_offsets, VALUE_DIM), out.to(Out.dtype.element_ty))


from problem_6 import triangular_inverse


from problem_4 import chunk_cumsum


from problem_8 import gated_scores


def _wy_representation(key, value, gate_cumsum, beta, kernel_matrix):
    batch_size, num_heads, seq_len, head_dim = key.shape
    value_dim = value.shape[-1]
    w, u = torch.empty_like(key), torch.empty_like(value)
    inverse = torch.empty_like(kernel_matrix)

    grid = (seq_len // CHUNK_SIZE, batch_size * num_heads)
    _chunk_wy_representation[grid](
        key, value, gate_cumsum, beta, kernel_matrix, w, u, inverse, seq_len,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
    )

    return w, u, inverse


def _chunk_states(key, w, u, gate_cumsum, initial_state):
    batch_size, num_heads, seq_len, head_dim = key.shape
    value_dim = u.shape[-1]
    num_chunks = seq_len // CHUNK_SIZE
    states = torch.empty(batch_size, num_heads, num_chunks + 1, head_dim, value_dim, device=key.device, dtype=torch.float32)
    states[:, :, 0] = initial_state
    pseudo_values = torch.empty_like(u)

    grid = (value_dim // BLOCK_V, batch_size * num_heads)
    _chunk_state_forward[grid](
        key, w, u, gate_cumsum, states, pseudo_values, seq_len, num_chunks,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, num_stages=1,
    )

    return states, pseudo_values


@torch.no_grad()
def chunk_kimi_delta_attention(
    query,
    key,
    value,
    gate,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
):
    query, key, value, scale, initial_state, gate, beta = prepare(
        query, key, value, scale, initial_state, gate, beta,
        gate_kind='channel', cuda=True)

    seq_len = query.shape[2]
    query, key, value = (pad_sequence(x, CHUNK_SIZE) for x in (query, key, value))
    gate = pad_sequence(gate, CHUNK_SIZE)  # Zero log gate = identity decay.
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

    # Local gate prefixes and strict key-key scores for the WY system.
    gate_cumsum = chunk_cumsum(gate)

    kernel_matrix = gated_scores(key, key, gate_cumsum, scale=1.0, strict=True)

    # WY factors, followed by the state-dependent pseudo-values.
    w, u, inverse = _wy_representation(key, value, gate_cumsum, beta, kernel_matrix)

    states, pseudo_values = _chunk_states(key, w, u, gate_cumsum, initial_state)

    # Inclusive query-key scores already include the output scale.
    scores = gated_scores(query, key, gate_cumsum, scale=scale, strict=False)

    out = torch.empty_like(value)

    grid = (num_chunks, value_dim // BLOCK_V, batch_size * num_heads)
    _chunk_output_forward[grid](
        query, gate_cumsum, pseudo_values, states, scores, out, scale, padded_len, num_chunks, **kernel_args)

    return finish(out[:, :, :seq_len].contiguous(), states[:, :, -1].contiguous(), output_final_state)
