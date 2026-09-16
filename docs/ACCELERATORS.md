# Accelerators

JSPT owns charts and `J Sigma J^T` on arrays. It does not own a GPU
runtime or a proving guest.

CUDA, SP1, and Rust ingest — if ever invoked from a consumer — are
CSE `rust-effort-gate`, default refuse. This package does not import
`gat` and does not open those gates.
