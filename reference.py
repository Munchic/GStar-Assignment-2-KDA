"""Public independent FP64 recurrence oracles; never imported by student code.

These intentionally slow definitions specify behavior, not Triton/chunk solutions.
No dependency on src/, FLA, or student subparts. Returns FP64 output and state.
"""

import torch


@torch.no_grad()
def recurrence(query, key, value, gate=None, beta=None, scale=None,
               initial_state=None):
    b, h, t, dk = query.shape
    dv = value.shape[-1]
    q, k, v = (x.to(torch.float64) for x in (query, key, value))
    scale = dk ** -0.5 if scale is None else scale
    state = (torch.zeros(b, h, dk, dv, device=q.device, dtype=torch.float64)
             if initial_state is None else initial_state.to(torch.float64).clone())
    out = torch.empty(b, h, t, dv, device=q.device, dtype=torch.float64)
    for i in range(t):
        if gate is not None:
            decay = gate[:, :, i].to(torch.float64).exp()
            if gate.ndim == 3:
                decay = decay.unsqueeze(-1)
            state = state * decay.unsqueeze(-1)
        write = v[:, :, i]
        if beta is not None:
            prediction = torch.einsum('bhk,bhkv->bhv', k[:, :, i], state)
            write = (write - prediction) * beta[:, :, i].to(torch.float64).unsqueeze(-1)
        state = state + k[:, :, i].unsqueeze(-1) * write.unsqueeze(-2)
        out[:, :, i] = torch.einsum('bhk,bhkv->bhv', q[:, :, i] * scale, state)
    return out, state


@torch.no_grad()
def direct_scores(left, key, prefixes, scale=1.0, strict=False, chunk_size=64):
    """FP64 local oracle; masks BEFORE exponentiation. No anchor factorization."""
    b, h, t, _ = key.shape
    if t % chunk_size:
        raise ValueError("Score oracle requires padded input")
    out = torch.zeros(b, h, t, chunk_size, device=key.device, dtype=torch.float64)
    rows = torch.arange(chunk_size, device=key.device)
    causal = rows[:, None] > rows[None, :] if strict else rows[:, None] >= rows[None, :]
    for start in range(0, t, chunk_size):
        l, k, g = (x[:, :, start:start + chunk_size].double() for x in (left, key, prefixes))
        difference = g.unsqueeze(-2) - g.unsqueeze(-3)
        decay = torch.where(causal[..., None], difference, -torch.inf).exp()
        out[:, :, start:start + chunk_size] = (
            l.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(-1) * scale
    return out
