# sensitivity-gate

CPU crate is opened under `sensitivity-gate/`. No PyO3 yet.

```
cd rust/sensitivity-gate
cargo test
```

The crate owns constitution, Chart, push_covariance, structure, and typed GateError.
Small dense f64 kernel (LU + Jacobi). Bindings wait until this contract stays green.
