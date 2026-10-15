# GStar Assignment 2: Build Kimi Delta Attention

How can a model remember a long context with a fixed-size state? In this
assignment, you will build the forward kernels step by step: start with linear
attention, add forgetting and prediction-error correction, then combine your
work into chunked **Kimi Delta Attention (KDA)**.

Like Assignment 1, the starter code supplies the GPU plumbing. You implement
the masked arithmetic in PyTorch and Triton. Only the forward pass is required.

## Start here

1. Read the [overview](docs/00_overview.pdf) or open the
   [introduction slides](docs/slides/KDA-KimiK3.pdf).
2. Follow Problems 1–9 in order, reading each PDF before editing its code.
3. Implement the `STUDENT IMPLEMENTATION REQUIRED` regions and run the checker.

## Learning path

| Problem | Implement | Reading |
|---|---|---|
| [P1](problem_1.py) | Sequential and chunked linear attention in PyTorch | [PDF](docs/problem_1.pdf) |
| [P2](problem_2.py) | Recurrent linear attention in Triton | [PDF](docs/problem_2.pdf) |
| [P3](problem_3.py) | Chunked linear attention | [PDF](docs/problem_3.pdf) |
| [P4](problem_4.py) | Forgetting: gated linear attention | [PDF](docs/problem_4.pdf) |
| [P5](problem_5.py) | Prediction-error correction: DeltaNet | [PDF](docs/problem_5.pdf) |
| [P6](problem_6.py) | Chunked DeltaNet and the triangular inverse | [PDF](docs/problem_6.pdf) |
| [P7](problem_7.py) | Combine forgetting and correction: Gated DeltaNet | [PDF](docs/problem_7.pdf) |
| [P8](problem_8.py) | Recurrent KDA and stable scores | [PDF](docs/problem_8.pdf) |
| [P9](problem_9.py) | Assemble the complete chunked KDA forward pass | [PDF](docs/problem_9.pdf) |

Later problems reuse your earlier kernels. In particular, P9 combines the P4
prefix scan, P6 triangular inverse, and P8 score builder. P4 uses a supplied
score baseline until you replace it with your P8 implementation.

## Setup

Use **Python 3.10+**. P1 runs on CPU, including macOS. P2–P9 require **Linux,
Triton, and an NVIDIA CUDA GPU**.

```bash
cd GStar-Assignment-2-KDA
python -m pip install -r requirements.txt
python autograder.py --check-templates
```

## Implement and check

Replace each `NotImplementedError` or `tl.static_assert(False, ...)` in the
student regions. Keep the function signatures and region markers intact.
The unfinished starter code is expected to fail numerical checks.

```bash
# Start with P1 on CPU.
python autograder.py --p1 --device cpu

# On a CUDA GPU, check each problem as you finish it.
python autograder.py --p2   # use --p3 through --p9 for later problems

# Check the complete assignment, then benchmark KDA after correctness passes.
python autograder.py --all --dtype bfloat16
python autograder.py --p9 --benchmark
```

The checker compares outputs and final states against independent references.
Its numerical tolerances are provisional; final grading criteria come from the
course staff. Use the required Triton kernels and reuse earlier student work;
do not call the checker references from your implementations.

For shapes, precision, and kernel conventions, consult the
[implementation reference](docs/11_reference.pdf). The formulas omit query
scaling; use `scale=1.0` to reproduce the written examples.

## Submit

Submit `problem_1.py` through `problem_9.py` with the explanation comments
requested in each problem. See the
[verification and submission guide](docs/10_verification.pdf) for details.