# Kernel and drift ledger

JSPT owns A2-A5 as code. Domain repos wrap types; they do not fork the law.
Consumption is one way. This package does not import `set_lcm` or `gat`.
Peers pin a git SHA, not floating `main`.

## Ownership

| Axiom | Shared object | Owner | Domain wrapper |
| --- | --- | --- | --- |
| A2 first-order map | `jacobian_at`, `jvp`, `first_order_covariance` | JSPT | FSRT predict uses F, Q; GAT propagate uses J |
| A2 exact mean | `propagate_belief` / `evaluate` | JSPT | GAT means are re-evaluations |
| A3 chart law | `x'=Tx+c`, `J'=S J T^{-1}`, `P'=T P T^T`, `F'=T F T^{-1}` | JSPT | FSRT GaussianState + fidelity checks |
| A4 additive error | `invariant_error`, `retract` | JSPT | FSRT masks, Joseph, declaration |
| A5 refuse repair | PSD-in-correlation, no clip, cond cap | JSPT constants | FSRT singular-S refusal; GAT fail-closed geometry |
| Site / IFC / verdict | -- | not shared | FSRT declaration, GAT disposition |

If a function needs a tank id or an IFC id, it does not go here.
If it only needs ndarray and a law, it does not stay forked in FSRT or GAT.

## Frozen names

`AffineCoordinates`, `transform_jacobian`, `first_order_covariance`,
`propagate_belief`, `local_structure`, `invariant_error`, `retract`,
`push_covariance`, `predict`, `update`

Add freely behind them. Do not rename to sound like GAT or FSRT.

## Constitution

- `MAX_CONDITION_NUMBER = 1e12`
- `ROUND_TRIP_TOLERANCE = 1e-8` in `max(|mean|, sigma)` and sigma-product units
- solve, do not invert
- symmetrize after `T P T^T`; never clip or nearest-PSD
- exact-zero variance keeps zero cross-covariance through a chart round-trip
- a chart that turns an exact-zero direction into a cross-term is refused

## Ledger

| law | module | consumers |
| --- | --- | --- |
| `J' = S J T^{-1}` | `coordinates.py` | FSRT adapter, GAT later, JSPT tests |
| `P' = T P T^T` with fidelity | `coordinates.push_covariance` | `affine.transform_state`, FSRT charts |
| `Sigma_y = J Sigma J^T` | `covariance.py` | FSRT predict, GAT belief |
| `f(mu) != J mu` | `structure.py` | GAT honesty, beam example |
| `ker J` | `structure.py` | none yet (observer later) |

When FSRT or GAT adds an ndarray helper that exists after deleting domain names,
the review question is why it is not an import.
