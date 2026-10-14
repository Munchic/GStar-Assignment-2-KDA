"""Public forward correctness/checkpoint checker, independent of src/ and FLA.

Numerical thresholds are provisional until validated on course CUDA hardware.
This checker reports failures, not official point scores. Run --check-templates
without PyTorch/Triton to audit syntax and the student dependency graph.
"""

import argparse
import ast
import importlib
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
STAGES = {
    1: ("linear", ("linear_reference_student", "linear_chunk_student")),
    2: ("linear", ("recurrent_linear_attention",)),
    3: ("linear", ("chunk_linear_attention",)),
    4: ("gla", ("recurrent_gated_linear_attention", "chunk_gated_linear_attention")),
    5: ("delta", ("recurrent_deltanet",)),
    6: ("delta", ("chunk_deltanet",)),
    7: ("gated_delta", ("recurrent_gated_deltanet", "chunk_gated_deltanet")),
    8: ("kda", ("recurrent_kimi_delta_attention",)),
    9: ("kda", ("chunk_kimi_delta_attention",)),
}


def audit_templates():
    """Static release audit; safe to run before installing course dependencies."""
    failures = []
    for p in range(1, 10):
        path = ROOT / f"problem_{p}.py"
        try:
            source = path.read_text()
            tree = ast.parse(source, filename=str(path))
            compile(tree, str(path), "exec")
            functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            for name in STAGES[p][1]:
                assert name in functions, f"Missing entry point {name}"
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imports = [node.module or ""]
                else:
                    imports = []
                assert not any(x.startswith(("src", "fla", "reference")) for x in imports), "Solution/oracle import"
                for name in imports:
                    if name.startswith("problem_"):
                        assert int(name.split("_")[1]) < p, "Dependency must be an earlier problem"
                if isinstance(node, ast.FunctionDef):
                    assert "backward" not in node.name, "Backward function in forward-only pack"
            count = source.count("# --- STUDENT IMPLEMENTATION REQUIRED:")
            print(f"P{p}: syntax/API/dependencies OK; {count} student regions")
        except Exception as exc:
            failures.append(f"P{p}: {exc}")
    for error in failures:
        print("FAIL", error)
    return not failures


class Runner:
    def __init__(self, verbose=False):
        self.passed = self.failed = 0
        self.verbose = verbose

    def test(self, label, fn):
        try:
            fn()
        except Exception as exc:
            self.failed += 1
            print(f"FAIL {label}: {type(exc).__name__}: {exc}")
            if self.verbose:
                traceback.print_exc()
            return False
        self.passed += 1
        print(f"PASS {label}")
        return True


def compare(actual, expected, label, dtype):
    import torch
    assert actual.shape == expected.shape, f"{label}: wrong shape {actual.shape}"
    assert actual.device == expected.device, f"{label}: wrong device"
    assert torch.isfinite(actual).all(), f"{label}: nonfinite result"
    a, e = actual.double(), expected.double()
    difference = a - e
    max_abs = difference.abs().max().item()
    rms = difference.square().mean().sqrt().item()
    ref_rms = e.square().mean().sqrt().item()
    relative = rms / max(ref_rms, 1e-12)
    # Provisional dtype-specific limits, explicitly not calibrated course grades.
    atol, rtol = {torch.float64: (1e-9, 1e-9), torch.float32: (5e-3, 1e-2),
                  torch.float16: (1e-2, 2e-2), torch.bfloat16: (3e-2, 5e-2)}[dtype]
    print(f"    {label}: max_abs={max_abs:.3e}, relative_RMS={relative:.3e}")
    assert max_abs <= atol + rtol * e.abs().max().item(), f"{label}: max absolute error too large"
    assert rms <= atol + rtol * ref_rms, f"{label}: RMS error too large"


def make_case(kind, shape, device, dtype, mode="random"):
    import torch
    b, h, t, dk, dv = shape
    q = torch.randn(b, h, t, dk, device=device, dtype=dtype) * 0.2
    k = torch.randn_like(q).float()
    k = (k / k.norm(dim=-1, keepdim=True).clamp_min(1e-8)).to(dtype)
    v = torch.randn(b, h, t, dv, device=device, dtype=dtype) * 0.2
    state = torch.randn(b, h, dk, dv, device=device, dtype=torch.float32) * 0.05
    kwargs = {"scale": 0.37, "initial_state": state, "output_final_state": True}
    if kind in ("gla", "gated_delta", "kda"):
        shape_g = (b, h, t) if kind == "gated_delta" else (b, h, t, dk)
        kwargs["gate"] = -torch.rand(shape_g, device=device) * (5.0 if mode == "strong_decay" else 0.1)
    if kind in ("delta", "gated_delta", "kda"):
        kwargs["beta"] = torch.rand(b, h, t, device=device)
    if mode == "zero_gates" and "gate" in kwargs:
        kwargs["gate"].zero_()
    if mode == "zero_beta" and "beta" in kwargs:
        kwargs["beta"].zero_()
    if mode == "zero_keys":
        k.zero_()
    if mode == "repeat_keys":
        k.zero_()
        k[..., 0] = 1
        if "beta" in kwargs:
            kwargs["beta"].fill_(1)
    if mode == "scalar_channels" and "gate" in kwargs and kwargs["gate"].ndim == 4:
        kwargs["gate"] = kwargs["gate"][..., :1].expand_as(q).contiguous()
    if mode == "zero_scale":
        kwargs["scale"] = 0.0
    if mode == "zero_state":
        state.zero_()
    if mode == "default_scale":
        kwargs.pop("scale")
    if mode == "default_state":
        kwargs.pop("initial_state")
    if mode == "noncontiguous":
        def strided(x):
            return x.transpose(-1, -2).contiguous().transpose(-1, -2)
        q, k, v = (strided(x) for x in (q, k, v))
        for name in ("initial_state", "gate", "beta"):
            if name in kwargs:
                kwargs[name] = strided(kwargs[name])
    return q, k, v, kwargs


def expected_case(q, k, v, kwargs):
    from reference import recurrence
    return recurrence(q, k, v, **{x: y for x, y in kwargs.items() if x != "output_final_state"})


def verify_call(fn, q, k, v, kwargs, oracle=None):
    import torch
    # Check all input buffers, including gate/beta/state, for mutation.
    tensors = [q, k, v] + [x for x in kwargs.values() if isinstance(x, torch.Tensor)]
    snapshots = [x.clone() for x in tensors]
    expected = expected_case(q, k, v, kwargs) if oracle is None else oracle
    result = fn(q, k, v, **kwargs)
    assert isinstance(result, tuple) and len(result) == 2, "Expected (output, final_state)"
    out, final = result
    assert out.dtype == v.dtype, "Output must preserve V dtype"
    assert final.dtype == (torch.float64 if q.dtype == torch.float64 else torch.float32), "Final state must use accumulation dtype"
    for before, after in zip(snapshots, tensors):
        assert torch.equal(before, after), "Input tensor was modified"
    compare(out, expected[0], "output", q.dtype)
    compare(final, expected[1], "final state", q.dtype)
    return out, final


def verify_segmented(fn, q, k, v, kwargs, split):
    import torch
    full = verify_call(fn, q, k, v, kwargs)
    first_kw, second_kw = kwargs.copy(), kwargs.copy()
    for name in ("gate", "beta"):
        if name in kwargs:
            first_kw[name] = kwargs[name][:, :, :split].contiguous()
            second_kw[name] = kwargs[name][:, :, split:].contiguous()
    first = verify_call(fn, q[:, :, :split], k[:, :, :split], v[:, :, :split], first_kw)
    second_kw["initial_state"] = first[1]
    second = verify_call(fn, q[:, :, split:], k[:, :, split:], v[:, :, split:], second_kw)
    compare(torch.cat((first[0], second[0]), dim=2), full[0], "segmented output", q.dtype)
    compare(second[1], full[1], "segmented final state", q.dtype)


def prefixes(gate):
    import torch
    if gate.ndim == 3:
        return gate.reshape(*gate.shape[:2], -1, 64).cumsum(-1).reshape_as(gate)
    b, h, t, d = gate.shape
    return gate.reshape(b, h, t // 64, 64, d).cumsum(-2).reshape_as(gate)


def correct_wy(key, value, beta, prefix=None):
    """Checker-only FP64 chunk algebra for isolating later student kernels."""
    import torch
    from reference import direct_scores
    b, h, t, dk = key.shape
    dv = value.shape[-1]
    k = key.double().reshape(b, h, t // 64, 64, dk)
    v = value.double().reshape(b, h, t // 64, 64, dv)
    bet = beta.double().reshape(b, h, t // 64, 64, 1)
    if prefix is None:
        gram = k @ k.transpose(-1, -2)
        entry_keys = k
    elif prefix.ndim == 3:
        g = prefix.double().reshape(b, h, t // 64, 64)
        rows = torch.arange(64, device=key.device)
        difference = g[..., :, None] - g[..., None, :]
        decay = torch.where(rows[:, None] > rows[None, :], difference, -torch.inf).exp()
        gram = (k @ k.transpose(-1, -2)) * decay
        entry_keys = k * g.exp().unsqueeze(-1)
    else:
        gram = direct_scores(key, key, prefix, strict=True).reshape(b, h, t // 64, 64, 64)
        entry_keys = k * prefix.double().reshape_as(k).exp()
    lower = torch.tril(bet * gram, diagonal=-1)
    eye = torch.eye(64, device=key.device, dtype=torch.float64).expand_as(lower)
    inverse = torch.linalg.solve_triangular(eye + lower, eye, upper=False, unitriangular=True)
    w, u = inverse @ (bet * entry_keys), inverse @ (bet * v)
    return (w.reshape_as(key).to(key.dtype), u.reshape_as(value).to(value.dtype),
            inverse.reshape(b, h, t, 64).float())


def boundary_oracle(q, k, v, kwargs):
    import torch
    from reference import recurrence
    boundaries = [kwargs["initial_state"].double()]
    outputs = []
    for start in range(0, q.shape[2], 64):
        kw = {name: kwargs[name][:, :, start:start+64] for name in ("gate", "beta") if name in kwargs}
        out, state = recurrence(q[:, :, start:start+64], k[:, :, start:start+64], v[:, :, start:start+64],
                                scale=kwargs["scale"], initial_state=boundaries[-1], **kw)
        outputs.append(out)
        boundaries.append(state)
    return torch.stack(boundaries, dim=2).float(), torch.cat(outputs, dim=2)


def checkpoints(p, module, runner, device, dtype):
    """Inject independent intermediates so an earlier TODO does not block a later test."""
    import torch
    kind = STAGES[p][0]
    q, k, v, kw = make_case(kind, (1, 2, 128, 128, 128), device, dtype)
    g = kw.get("gate")
    pre = prefixes(g) if g is not None else None
    if p == 4:
        runner.test("P4 gate prefix (isolated)", lambda: compare(module.chunk_cumsum(g), pre, "prefix", torch.float32))
    if p == 6:
        def inverse_test():
            from grader_kernels import inverse_forward
            lower = torch.tril(torch.randn(2, 64, 64, device=device) * 0.02, diagonal=-1)
            eye = torch.eye(64, device=device).expand_as(lower)
            expect = torch.linalg.solve_triangular(eye + lower, eye, upper=False, unitriangular=True)
            actual = inverse_forward(lower)
            compare(actual, expect, "inverse", torch.float32)
            compare((eye + lower) @ actual, eye, "inverse residual", torch.float32)
        runner.test("P6 triangular inverse (isolated)", inverse_test)
    if p == 8:
        from reference import direct_scores
        for strict in (False, True):
            # Large negative prefixes catch exp(-prefix) factorization overflow.
            stressed = prefixes(-torch.rand_like(k.float()) * 5)
            runner.test(f"P8 gated scores strict={strict} (isolated)", lambda strict=strict: compare(
                module.gated_scores(q, k, stressed, scale=0.37, strict=strict),
                direct_scores(q, k, stressed, scale=0.37, strict=strict), "scores", dtype))
        # P4 -> P8 integration uses P4 student kernels, not an instructor kernel.
        def gla_integration():
            p4 = importlib.import_module("problem_4")
            case = make_case("gla", (1, 1, 65, 64, 128), device, dtype)
            verify_call(lambda *args, **kwargs: p4.chunk_gated_linear_attention(
                *args, **kwargs, score_builder=module.gated_scores), *case)
        runner.test("P8 score builder integrated into P4", gla_integration)
    if p in (6, 7, 9):
        w, u, inverse = correct_wy(k, v, kw["beta"], pre)
        def wy_test():
            if p == 6:
                actual = module._wy_representation(k, v, kw["beta"])
            elif p == 7:
                actual = module._wy_representation(k, v, g, kw["beta"])
            else:
                from reference import direct_scores
                correct_matrix = direct_scores(k, k, pre, strict=True).float()
                actual = module._wy_representation(k, v, pre, kw["beta"], correct_matrix)
            for label, result, target in zip(("W", "U", "A"), actual, (w, u, inverse)):
                compare(result, target, label, dtype)
        runner.test(f"P{p} WY checkpoints", wy_test)
    if p not in (3, 4, 6, 7, 9):
        return
    states, expected_output = boundary_oracle(q, k, v, kw)
    pseudo = None
    if p in (6, 7, 9):
        chunks = []
        for i in range(2):
            chunks.append((u[:, :, i*64:(i+1)*64].double()
                           - w[:, :, i*64:(i+1)*64].double() @ states[:, :, i].double()).to(dtype))
        pseudo = torch.cat(chunks, dim=2)
    def state_test():
        if p == 3:
            actual = module._chunk_states(k, v, kw["initial_state"])
        elif p == 4:
            actual = module._chunk_states(k, v, pre, kw["initial_state"])
        elif p == 6:
            actual, actual_pseudo = module._chunk_states(k, w, u, kw["initial_state"])
        else:
            actual, actual_pseudo = module._chunk_states(k, w, u, g if p == 7 else pre, kw["initial_state"])
        compare(actual, states, "chunk boundaries", dtype)
        if pseudo is not None:
            compare(actual_pseudo, pseudo, "pseudo-values", dtype)
    runner.test(f"P{p} state kernel with supplied correct inputs", state_test)
    def output_test():
        from reference import direct_scores
        if p in (3, 6):
            args = (q, k, v if p == 3 else pseudo, states)
        elif p == 7:
            args = (q, k, g, pseudo, states)
        else:
            scores = direct_scores(q, k, pre, scale=kw["scale"]).float()
            args = ((q, v, pre, states, scores) if p == 4 else (q, pre, pseudo, states, scores))
        out = torch.empty_like(v)
        module._chunk_output_forward[(2, 2, 2)](*args, out, kw["scale"], 128, 2,
            HEAD_DIM=128, VALUE_DIM=128, CHUNK=64, BLOCK_K=64, BLOCK_V=64)
        compare(out, expected_output, "isolated output", dtype)
    runner.test(f"P{p} output kernel with supplied correct inputs", output_test)


def benchmark(fn, case):
    import torch
    q, k, v, kw = case
    for _ in range(5):
        fn(q, k, v, **kw)
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(20):
        fn(q, k, v, **kw)
    end.record()
    torch.cuda.synchronize()
    print(f"    forward time (including validation/allocations): {begin.elapsed_time(end) / 20:.3f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for p in range(1, 10):
        parser.add_argument(f"--p{p}", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--check-templates", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args()
    if args.check_templates:
        return 0 if audit_templates() else 1
    try:
        import torch
    except ImportError:
        print("PyTorch is required. Install dependencies from requirements.txt.")
        return 1
    selected = [p for p in STAGES if getattr(args, f"p{p}")]
    if args.all or not selected:
        selected = list(STAGES)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable. P1 can run with --p1 --device cpu; P2-P9 require CUDA.")
        return 1
    if args.benchmark and args.device != "cuda":
        print("Benchmarking requires CUDA.")
        return 1
    torch.manual_seed(2026)
    dtype = getattr(torch, args.dtype)
    runner = Runner(args.verbose)
    print("Public checker: provisional tolerances; official grading requires course GPU calibration.")
    print(f"PyTorch={torch.__version__}, device={args.device}, dtype={args.dtype}")
    if args.device == "cuda":
        print(f"GPU={torch.cuda.get_device_name()}")
    for p in selected:
        if p != 1 and args.device == "cpu":
            runner.test(f"P{p} requires CUDA", lambda: (_ for _ in ()).throw(ValueError("CUDA required")))
            continue
        loaded = []
        if not runner.test(f"P{p} import", lambda: loaded.append(importlib.import_module(f"problem_{p}"))):
            continue
        module = loaded[0]
        if p != 1:
            runner.test(f"P{p} checkpoint setup", lambda: checkpoints(p, module, runner, args.device, dtype))
        kind, names = STAGES[p]
        for name in names:
            fn = getattr(module, name)
            failures_before = runner.failed
            for t in (1, 63, 64, 65, 127, 128, 129, 200):
                shape = (1, 1, t, 64, 64)
                runner.test(f"P{p} {name} T={t}", lambda shape=shape: verify_call(
                    fn, *make_case(kind, shape, args.device, dtype)))
            for dk, dv in ((128, 64), (64, 128), (128, 128)):
                shape = (2, 2, 65, dk, dv)
                runner.test(f"P{p} {name} Dk/Dv={dk}/{dv}", lambda shape=shape: verify_call(
                    fn, *make_case(kind, shape, args.device, dtype)))
            for mode in ("zero_gates", "zero_beta", "zero_keys", "repeat_keys", "strong_decay", "scalar_channels", "zero_scale", "zero_state", "default_scale", "default_state", "noncontiguous"):
                runner.test(f"P{p} {name} {mode}", lambda mode=mode: verify_call(
                    fn, *make_case(kind, (1, 1, 65, 64, 64), args.device, dtype, mode)))
            for split in (1, 63, 65):
                runner.test(f"P{p} {name} state passing split={split}", lambda split=split: verify_segmented(
                    fn, *make_case(kind, (1, 1, 129, 64, 64), args.device, dtype), split))
            # Final-state flag and P1 arbitrary dimensions/chunk sizes have their own cases.
            def output_only():
                q, k, v, kw = make_case(kind, (1, 1, 65, 64, 64), args.device, dtype)
                expected = expected_case(q, k, v, kw)[0]
                kw["output_final_state"] = False
                result = fn(q, k, v, **kw)
                assert isinstance(result, torch.Tensor), "Output-only call must return a Tensor"
                compare(result, expected, "output-only", dtype)
            runner.test(f"P{p} {name} output-only", output_only)
            if p == 1:
                for c in (1, 7, 32):
                    case = make_case(kind, (2, 2, 17, 5, 3), args.device, torch.float64 if args.device == "cpu" else dtype)
                    if name == "linear_chunk_student":
                        case[3]["chunk_size"] = c
                    # Chunk size is a student option, not an oracle argument.
                    expected = expected_case(*case[:3], {k: v for k, v in case[3].items() if k != "chunk_size"})
                    runner.test(f"P1 {name} arbitrary dims/chunk={c}", lambda case=case, expected=expected:
                                verify_call(fn, *case, oracle=expected))
            if args.benchmark and runner.failed == failures_before:
                runner.test(f"P{p} {name} benchmark", lambda: benchmark(
                    fn, make_case(kind, (1, 4, 1024, 64, 64), args.device, dtype)))
    print(f"{runner.passed} passed; {runner.failed} failed")
    return 1 if runner.failed else 0


if __name__ == "__main__":
    sys.exit(main())
