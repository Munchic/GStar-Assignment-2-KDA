"""Problem 7: Gated DeltaNet (forward only).

Scalar log gates decay the state before prediction.
Apply scalar relative decay to the strict WY matrix and inclusive output scores.
Reuse the P6 triangular inverse; never scale the WY matrix by query scale.
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
    Q, K, V, Gate, Beta, Out, InitialState, FinalState,
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
    gate_ptrs = Gate + batch_head * seq_len
    beta_ptrs = Beta + batch_head * seq_len
    out_ptrs = Out + batch_head * seq_len * VALUE_DIM + value_offsets

    state = tl.load(tile_ptrs(InitialState, batch_head * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM))

    for t in range(seq_len):
        q = tl.load(q_ptrs + t * HEAD_DIM).to(tl.float32) * scale
        k = tl.load(k_ptrs + t * HEAD_DIM).to(tl.float32)
        v = tl.load(v_ptrs + t * VALUE_DIM).to(tl.float32)
        gate = tl.load(gate_ptrs + t).to(tl.float32)
        beta = tl.load(beta_ptrs + t).to(tl.float32)

        # --- STUDENT IMPLEMENTATION REQUIRED: P7.RECURRENT.DECAY ---
        # Apply scalar forgetting BEFORE prediction.
        state = tl.exp(gate) * state
        # --- END OF STUDENT IMPLEMENTATION ---

        # --- STUDENT IMPLEMENTATION REQUIRED: P7.RECURRENT.RESIDUAL ---
        # Set error [BLOCK_V] using the decayed previous state.
        u = beta * (v - tl.sum(state * k[:, None], axis=0))
        # --- END OF STUDENT IMPLEMENTATION ---

        # --- STUDENT IMPLEMENTATION REQUIRED: P7.RECURRENT.WRITE ---
        # Apply the beta-weighted rank-one correction.
        state += k[:, None] * u[None, :]
        # --- END OF STUDENT IMPLEMENTATION ---

        # --- STUDENT IMPLEMENTATION REQUIRED: P7.RECURRENT.READ ---
        # Set out using the updated state.
        out = tl.sum(q[:, None] * state, axis=0)
        # --- END OF STUDENT IMPLEMENTATION ---

        tl.store(out_ptrs + t * VALUE_DIM, out.to(Out.dtype.element_ty))

    tl.store(tile_ptrs(FinalState, batch_head * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


@triton.jit
def _chunk_wy_representation(
    K, V, Gate, Beta, W, U, A,
    seq_len,
    HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    chunk = tl.program_id(0)
    batch_head = tl.program_id(1)
    chunk_offsets = tl.arange(0, CHUNK)
    rows = chunk * CHUNK + chunk_offsets
    beta = tl.load(Beta + batch_head * seq_len + rows).to(tl.float32)

    # --- STUDENT IMPLEMENTATION REQUIRED: P7.WY.GATES ---
    # Load scalar log gates; set inclusive gate_cumsum [C] and strictly causal decay [C,C]. Mask differences before exp.
    gate = tl.load(Gate + batch_head * seq_len + rows).to(tl.float32)
    gate_cumsum = tl.cumsum(gate, axis=0)
    mask = chunk_offsets[:, None] > chunk_offsets[None, :] 
    gate_diff = gate_cumsum[:, None] - gate_cumsum[None, :]
    decay = tl.exp(tl.where(mask, gate_diff, float("-inf")))
    # --- END OF STUDENT IMPLEMENTATION ---

    # L = tril(diag(beta) (K K^T ⊙ decay), -1)
    lower = tl.zeros([CHUNK, CHUNK], tl.float32)

    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))

        # --- STUDENT IMPLEMENTATION REQUIRED: P7.WY.GRAM ---
        # Accumulate beta-weighted key Gram matrix across key tiles.
        lower += beta[:, None] * tl.dot(k, k.trans())
        # --- END OF STUDENT IMPLEMENTATION ---

    # --- STUDENT IMPLEMENTATION REQUIRED: P7.WY.STRICT ---
    # Finish L with STRICT causality and relative scalar decay.
    lower = tl.where(mask, lower * decay, 0.0)
    # --- END OF STUDENT IMPLEMENTATION ---

    # A = (I + L)^{-1} for the strictly lower triangular L
    inverse = triangular_inverse(lower, CHUNK)

    tl.store(tile_ptrs(A, batch_head * seq_len, rows, chunk_offsets, CHUNK), inverse)

    for key_block in range(HEAD_DIM // BLOCK_K):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))

        # --- STUDENT IMPLEMENTATION REQUIRED: P7.WY.W ---
        # Set w [C,BLOCK_K] using A and beta-weighted entry-decayed keys.
        k_start = k * tl.exp(gate_cumsum)[:, None]
        w = tl.dot(inverse, beta[:, None] * k_start)
        # --- END OF STUDENT IMPLEMENTATION ---

        tl.store(tile_ptrs(W, batch_head * seq_len, rows, key_offsets, HEAD_DIM), w.to(W.dtype.element_ty))

    for value_block in range(VALUE_DIM // BLOCK_V):
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        v = tl.load(tile_ptrs(V, batch_head * seq_len, rows, value_offsets, VALUE_DIM))

        # --- STUDENT IMPLEMENTATION REQUIRED: P7.WY.U ---
        # Set u [C,BLOCK_V] using A and beta-weighted values.
        u = tl.dot(inverse, beta[:, None] * v)
        # --- END OF STUDENT IMPLEMENTATION ---

        tl.store(tile_ptrs(U, batch_head * seq_len, rows, value_offsets, VALUE_DIM), u.to(U.dtype.element_ty))


@triton.jit
def _chunk_state_forward(
    K, W, U, Gate, States, PseudoValues,
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
        gate = tl.load(Gate + batch_head * seq_len + rows).to(tl.float32)

        # --- STUDENT IMPLEMENTATION REQUIRED: P7.STATE.PREFIX ---
        # Set inclusive gate_cumsum [C] and scalar chunk_decay.
        gate_cumsum = tl.cumsum(gate, axis=0)
        chunk_decay = tl.exp(tl.sum(gate))
        # --- END OF STUDENT IMPLEMENTATION ---

        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        w = tl.load(tile_ptrs(W, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        u = tl.load(tile_ptrs(U, batch_head * seq_len, rows, value_offsets, VALUE_DIM))

        # --- STUDENT IMPLEMENTATION REQUIRED: P7.STATE.PSEUDO ---
        # Set pseudo_values [C,BLOCK_V] from u, w and the entry state.
        pseudo_values = u - tl.dot(w, state)
        # --- END OF STUDENT IMPLEMENTATION ---

        tl.store(tile_ptrs(PseudoValues, batch_head * seq_len, rows, value_offsets, VALUE_DIM), pseudo_values.to(PseudoValues.dtype.element_ty))

        # --- STUDENT IMPLEMENTATION REQUIRED: P7.STATE.UPDATE ---
        # Decay entry state and add exit-decayed corrected writes.
        k_end = tl.exp((tl.sum(gate) - gate_cumsum)[:, None]) * k
        state = chunk_decay * state + tl.dot(k_end.trans(), pseudo_values)
        # --- END OF STUDENT IMPLEMENTATION ---

        tl.store(tile_ptrs(States, state_row_base + (chunk + 1) * HEAD_DIM, key_offsets, value_offsets, VALUE_DIM), state)


@triton.jit
def _chunk_output_forward(
    Q, K, Gate, PseudoValues, States, Out,
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

    # --- STUDENT IMPLEMENTATION REQUIRED: P7.OUTPUT.GATES ---
    # Set chunk-local prefixes and INCLUSIVE relative decay; mask before exponentiation.
    gate = tl.load(Gate + batch_head * seq_len + rows).to(tl.float32)
    gate_cumsum = tl.cumsum(gate, axis=0)
    mask = chunk_offsets[:, None] >= chunk_offsets[None, :]
    diff = gate_cumsum[:, None] - gate_cumsum[None, :]
    decay = tl.exp(tl.where(mask, diff, float("-inf")))

    # --- END OF STUDENT IMPLEMENTATION ---

    pseudo_values = tl.load(tile_ptrs(PseudoValues, batch_head * seq_len, rows, value_offsets, VALUE_DIM))
    inter = tl.zeros([CHUNK, BLOCK_V], tl.float32)
    scores = tl.zeros([CHUNK, CHUNK], tl.float32)

    for key_block in tl.range(HEAD_DIM // BLOCK_K, num_stages=1):
        key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        q = tl.load(tile_ptrs(Q, batch_head * seq_len, rows, key_offsets, HEAD_DIM)) * scale
        k = tl.load(tile_ptrs(K, batch_head * seq_len, rows, key_offsets, HEAD_DIM))
        state = tl.load(tile_ptrs(States, state_row_base, key_offsets, value_offsets, VALUE_DIM))

        # --- STUDENT IMPLEMENTATION REQUIRED: P7.OUTPUT.PRODUCTS ---
        # Accumulate entry-decayed inter reads and un-gated local scores.
        q_start = q * tl.exp(gate_cumsum)[:, None]
        inter += tl.dot(q_start, state)
        scores += tl.dot(q, k.trans())
        # --- END OF STUDENT IMPLEMENTATION ---

    # --- STUDENT IMPLEMENTATION REQUIRED: P7.OUTPUT.COMBINE ---
    # Apply relative decay to scores; combine inter and corrected local writes.
    scores = scores * decay
    out = inter + tl.dot(scores, pseudo_values)
    # --- END OF STUDENT IMPLEMENTATION ---

    # problem 7.4
    # we decay first before writing the gate forgets past memory
    # if we writen then decay, the new write will be multiplied by e^{g_t}
    # so we get a different recurrence 
    # i guess mathematically it doesn't matter but for the sake of semantics
    # past memories forget, what we observe now is compelete (undecayed)

    tl.store(tile_ptrs(Out, batch_head * seq_len, rows, value_offsets, VALUE_DIM), out.to(Out.dtype.element_ty))


from problem_6 import triangular_inverse


@torch.no_grad()
def recurrent_gated_deltanet(
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
        gate_kind='scalar', cuda=True)

    batch_size, num_heads, seq_len, head_dim = query.shape
    value_dim = value.shape[-1]

    out = torch.empty_like(value)
    final_state = torch.empty_like(initial_state)

    grid = (value_dim // BLOCK_V, batch_size * num_heads)
    _recurrent_forward[grid](
        query, key, value, gate, beta, out, initial_state, final_state, scale, seq_len,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, BLOCK_V=BLOCK_V)

    return finish(out, final_state, output_final_state)


def _wy_representation(key, value, gate, beta):
    batch_size, num_heads, seq_len, head_dim = key.shape
    value_dim = value.shape[-1]
    w, u = torch.empty_like(key), torch.empty_like(value)
    inverse = torch.empty(batch_size, num_heads, seq_len, CHUNK_SIZE, device=key.device, dtype=torch.float32)

    grid = (seq_len // CHUNK_SIZE, batch_size * num_heads)
    _chunk_wy_representation[grid](
        key, value, gate, beta, w, u, inverse, seq_len,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
    )

    return w, u, inverse


def _chunk_states(key, w, u, gate, initial_state):
    batch_size, num_heads, seq_len, head_dim = key.shape
    value_dim = u.shape[-1]
    num_chunks = seq_len // CHUNK_SIZE
    states = torch.empty(batch_size, num_heads, num_chunks + 1, head_dim, value_dim, device=key.device, dtype=torch.float32)
    states[:, :, 0] = initial_state
    pseudo_values = torch.empty_like(u)

    grid = (value_dim // BLOCK_V, batch_size * num_heads)
    _chunk_state_forward[grid](
        key, w, u, gate, states, pseudo_values, seq_len, num_chunks,
        HEAD_DIM=head_dim, VALUE_DIM=value_dim, CHUNK=CHUNK_SIZE, BLOCK_V=BLOCK_V,
    )

    return states, pseudo_values


@torch.no_grad()
def chunk_gated_deltanet(
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
        gate_kind='scalar', cuda=True)

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

    w, u, inverse = _wy_representation(key, value, gate, beta)

    states, pseudo_values = _chunk_states(key, w, u, gate, initial_state)

    out = torch.empty_like(value)

    grid = (num_chunks, value_dim // BLOCK_V, batch_size * num_heads)
    _chunk_output_forward[grid](
        query, key, gate, pseudo_values, states, out, scale, padded_len, num_chunks, **kernel_args)

    return finish(out[:, :, :seq_len].contiguous(), states[:, :, -1].contiguous(), output_final_state)
