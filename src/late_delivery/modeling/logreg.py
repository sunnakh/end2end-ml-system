"""L2-regularised logistic regression fitted with Newton's method, in plain NumPy.

Objective:  mean log loss  +  (l2 / 2) * ||w||^2   (the intercept b is not penalised)
Newton step: solve H d = g, where H = X'SX / n + l2*I (S = diag(p(1-p))). With ~50
columns each step costs one small linear solve, and convergence takes a handful of steps.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])  # never overflows: z < 0 here
    out[~pos] = ez / (1.0 + ez)
    return out


def _log_loss(z: np.ndarray, y: np.ndarray) -> float:
    # log(1 + e^z) - y*z, computed stably as logaddexp(0, z) - y*z
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def loss_and_grad(
    w: np.ndarray, b: float, x: np.ndarray, y: np.ndarray, l2: float
) -> tuple[float, np.ndarray, float]:
    z = x @ w + b
    residual = (_sigmoid(z) - y) / len(y)
    loss = _log_loss(z, y) + 0.5 * l2 * float(w @ w)
    return loss, x.T @ residual + l2 * w, float(residual.sum())


def fit_logreg(
    x: np.ndarray, y: np.ndarray, *, l2: float, max_iter: int = 50, tol: float = 1e-10
) -> tuple[np.ndarray, float]:
    """Returns (w, b). Newton steps with backtracking, stopping when the step is tiny."""
    n, d = x.shape
    xb = np.hstack([x, np.ones((n, 1))])  # last column = intercept
    theta = np.zeros(d + 1)
    penalty = np.r_[np.full(d, l2), 0.0]

    def objective(t: np.ndarray) -> float:
        return _log_loss(xb @ t, y) + 0.5 * float((penalty * t) @ t)

    current = objective(theta)
    for _ in range(max_iter):
        p = _sigmoid(xb @ theta)
        grad = xb.T @ (p - y) / n + penalty * theta
        hess = (xb * (p * (1 - p))[:, None]).T @ xb / n + np.diag(penalty)
        hess[np.diag_indices_from(hess)] += 1e-12  # keeps the solve defined if l2 = 0
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while True:  # backtracking: accept the step once the objective does not rise
            candidate = theta - t * step
            value = objective(candidate)
            if value <= current + 1e-15 or t < 1e-8:
                break
            t /= 2
        theta, current = candidate, value
        if np.max(np.abs(t * step)) < tol:
            break
    return theta[:-1].copy(), float(theta[-1])


def predict_proba(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return _sigmoid(x @ w + b)
