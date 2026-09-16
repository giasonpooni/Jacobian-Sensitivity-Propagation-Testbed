# Kernel and drift ledger

JSPT owns A2–A5 as code. Domain repos wrap types; they do not fork the law.

## Frozen names

`AffineCoordinates`, `transform_jacobian`, `push_covariance`,
`first_order_covariance`, `propagate_belief`, `local_structure`,
`invariant_error`, `retract`, `predict`, `update`

## Constitution

- `MAX_CONDITION_NUMBER = 1e12`
- `ROUND_TRIP_TOLERANCE = 1e-8` in `max(|mean|, σ)` and σ-product units
- solve, do not invert
- symmetrize after `T P Tᵀ`; never clip or nearest-PSD
- exact-zero variance keeps zero cross-covariance through a chart round-trip

## Ledger

| law | module | consumers |
| --- | --- | --- |
| `J' = S J T⁻¹` | `coordinates.py` | JSPT tests; FSRT adapter next |
| `P' = T P Tᵀ` with fidelity | `coordinates.push_covariance` | `affine.transform_state` |
| `Σ_y = J Σ Jᵀ` | `covariance.py` | `propagate_belief`, affine predict |
| `f(μ) ≠ Jμ` | `structure.py` | beam example |
| `ker J` | `structure.py` | none yet |

Consumption is one way. This package does not import `set_lcm` or `gat`.
Peers pin a git SHA, not floating `main`.
