# HANDOFF

Public siblings: Fluid-State-Reconstruction-Testbed, Construction-State-Estimator-for-BIM,
Parameterized-Lyapunov-Stability-Runtime, Geodesic-Flow-and-Jacobi-Field-Testbed.
This repository is the local first-order sensitivity slice. It does not
import those packages and does not absorb their domains.

## Delivered

- `sensitivity` package: declared maps, Jacobians, composition, linear charts,
  first-order covariance, remainder sweeps, declared metrics.
- Chart support follows the FSRT numeric limit: condition number <= 1e12,
  solves instead of explicit inverses, refused ill-conditioned T or S.
- Covariance is checked in correlation coordinates. Exact zero-variance
  rows must carry zero cross-covariance. No nearest-PSD repair.
- Portable examples: two-tank storage/balance/measurement, Euler-Bernoulli
  midspan deflection. These are stand-ins, not FSRT or GAT.

## Not delivered

- Global sensitivity, switching modes, trajectory / Jacobi / tangent-linear
  ODE-PDE maps, Lyapunov certificates, PLC/SCADA, BIM kernels.

## Run

```
uv run --python 3.13 python examples/quickstart.py
uv run --python 3.13 python examples/fluid_composition.py
uv run --python 3.13 python examples/construction_composition.py
uv run --python 3.13 --dev pytest
```
