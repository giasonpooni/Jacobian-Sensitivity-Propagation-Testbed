# Development workflow

Maintain the Jacobian and Sensitivity Propagation Testbed as one project on `main`.

- Work directly on `main` and push completed, validated changes to `origin/main`.
- Do not create development branches, separate project copies, or pull requests unless
  the user explicitly requests them.
- Fetch before pushing, preserve concurrent work, and never force-push `main`.
- Run the default test suite and the quickstart example before pushing library changes.
- Keep the README focused on delivered functionality. Mark research extensions as planned
  until implemented and validated.
- First-release scope is local, first-order sensitivity through explicit differentiable
  model compositions. Do not imply global sensitivity, discontinuous mode changes,
  trajectory sensitivities, or a production runtime until those exist and are tested.
- Sensitivity scores that depend on matrix norms or singular values must declare a
  scaling or metric. Do not compare raw Jacobian entries across unit systems.
