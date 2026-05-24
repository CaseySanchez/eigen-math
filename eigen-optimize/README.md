# eigen-optimize

## `levenberg_marquardt`

Nonlinear least-squares optimization with the Levenberg–Marquardt algorithm. The Jacobian is computed numerically using forward finite differences, so only a residuals function is required. An optional `PostStepFunctor` is called after each accepted step to enforce constraints (e.g. quaternion normalization).
