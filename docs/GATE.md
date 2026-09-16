# Sealed gate (planned)

NumPy JSPT is the oracle. A later Rust crate is the only compiled gate.
CUDA and C++ are backends behind that gate. They are not a second constitution.

## Law that must survive the gate

- MAX_CONDITION_NUMBER = 1e12 (κ2(T) via SVD)
- ROUND_TRIP_TOLERANCE = 1e-8 in max(|μ|,σ) and σiσj units
- solve, do not invert; no clip / nearest-PSD
- exact-zero rows stay exact on the round trip
- Skeel is diagnostic; it does not move the cap
- κ2(J) only for square full-rank maps; otherwise rank + singular values

No backend may expose those numbers as kwargs.

## Stack (not implemented)

Python oracle (this repository) → pyo3 → Rust gate → CPU first, C++/CUDA later.

CPU must pass tests/contracts/test_chart_law.py through the gate.

## Refusal

Do not add CuPy, JAX-as-runtime, or a local T @ P @ T.T in sibling repos.
