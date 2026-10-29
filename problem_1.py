"""Problem 1: Understand matrix memory in PyTorch (forward only).

Q/K: [B,H,T,Dk], V: [B,H,T,Dv], state: [B,H,Dk,Dv].
This is causal unnormalized linear attention: there is no softmax or denominator.
Read AFTER writing the current token, so the local causal mask is inclusive.
Replace each NotImplementedError with your code. Keep FP32 state/accumulation
(FP64 when inputs are FP64); cast only the returned output to the input dtype.
See Problem 1 in the handout for the derivation and worked example.
"""

import torch
from common import prepare, finish


@torch.no_grad()
def linear_reference_student(query, key, value, scale=None, initial_state=None,
                             output_final_state=False):
    query, key, value, scale, state, _, _ = prepare(
        query, key, value, scale, initial_state)

    out = torch.empty_like(value)

    for t in range(query.shape[2]):
        q = query[:, :, t].to(state.dtype) * scale
        k = key[:, :, t].to(state.dtype)
        v = value[:, :, t].to(state.dtype)

        # --- STUDENT IMPLEMENTATION REQUIRED: P1.RECURRENT.UPDATE ---
        # Add the current key/value outer product to state [B,H,Dk,Dv].
        state += k.unsqueeze(-1) @ v.unsqueeze(-2)
        # --- END OF STUDENT IMPLEMENTATION ---

        # --- STUDENT IMPLEMENTATION REQUIRED: P1.RECURRENT.READ ---
        # Set o [B,H,Dv] by reading the UPDATED state with q.
        o = (q.unsqueeze(-2) @ state).squeeze(-2)
        # --- END OF STUDENT IMPLEMENTATION ---

        out[:, :, t] = o.to(value.dtype)

    return finish(out, state, output_final_state)


@torch.no_grad()
def linear_chunk_student(query, key, value, scale=None, initial_state=None,
                         output_final_state=False, chunk_size=64):
    query, key, value, scale, state, _, _ = prepare(
        query, key, value, scale, initial_state)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    out = torch.empty_like(value)

    for start in range(0, query.shape[2], chunk_size):
        end = min(start + chunk_size, query.shape[2])
        q = query[:, :, start:end].to(state.dtype) * scale  # [B, H, C, Dk]
        k = key[:, :, start:end].to(state.dtype)  # [B, H, C, Dk]
        v = value[:, :, start:end].to(state.dtype)  # [B, H, C, Dv]
        rows = torch.arange(end - start, device=query.device)

        # --- STUDENT IMPLEMENTATION REQUIRED: P1.CHUNK.OUTPUT ---
        # Set o [B,H,C,Dv] from the ENTRY state and local scores [B,H,C,C].
        # Use an inclusive causal mask with zero scores above the diagonal.
        # Do not call linear_reference_student or loop over individual tokens.
        o = q @ state + torch.tril(q @ k.transpose(-2, -1)) @ v
        # --- END OF STUDENT IMPLEMENTATION ---

        out[:, :, start:end] = o.to(value.dtype)

        # --- STUDENT IMPLEMENTATION REQUIRED: P1.CHUNK.STATE ---
        # Advance state using this chunk's keys/values, AFTER computing o.
        state = state + k.transpose(-2, -1) @ v
        # --- END OF STUDENT IMPLEMENTATION ---

    return finish(out, state, output_final_state)
