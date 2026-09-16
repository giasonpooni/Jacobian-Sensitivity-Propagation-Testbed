# Sealed gate

NumPy JSPT is the development oracle. The Rust crate is the compiled gate.
CUDA and C++ are backends behind that gate. They are not a second constitution.

## Law that must survive the gate

- MAX_CONDITION_NUMBER = 1e12 (k2(T) via SVD)
- ROUND_TRIP_TOLERANCE = 1e-8 in max(|mean|, sigma) and sigma_i sigma_j units
- solve, do not invert; no clip / nearest-PSD
- exact-zero rows stay exact on the round trip
- Skeel is diagnostic; it does not move the cap
- k2(J) only for square full-rank maps; otherwise rank + singular values

No backend may expose those numbers as kwargs.

## Binding slice

`rust/sensitivity-gate` is the CPU crate. The `python` feature exposes
Chart and push_covariance via PyO3. Hatch JSPT stays the oracle.
`tests/contracts/test_gate_parity.py` compares them and skips if the
cdylib is not installed.

## Language clients (inaugural, not bolted on)

Python is the development oracle. It is not the only process that may call the law.

| Client | Binding | Status |
| --- | --- | --- |
| Python | PyO3 cdylib (`python` feature) | opened |
| Julia | ccall to a stable C ABI on the crate | planned |
| C++ | same C ABI | planned with CUDA backend |

Julia does not sit above Python and does not own k2 or fidelity.
A future Sensitivity.jl may only wrap gate symbols. No LinearAlgebra.cond
as a second cap. No Zygote path that bypasses refuse-repair.

Consequence now: design symbols as a C ABI (row-major float64, integer
GateError, no kwargs). Do not make PyO3 the only door. Do not add a Julia
package until that ABI exists and the two-tank contract passes through it.

Observation records and instrument manifests stay language-neutral.

## Refusal

Do not add CuPy, JAX-as-runtime, or a local T @ P @ T.T in sibling repos.
