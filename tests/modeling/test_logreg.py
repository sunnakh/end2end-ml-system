import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from late_delivery.modeling.logreg import fit_logreg, loss_and_grad, predict_proba


def _data(n: int = 2000, d: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    true_w = rng.normal(size=d)
    y = (rng.random(n) < 1 / (1 + np.exp(-(x @ true_w - 1.0)))).astype(np.float64)
    return x, y


def test_gradient_matches_finite_differences() -> None:
    x, y = _data(300, 5)
    rng = np.random.default_rng(1)
    w, b, l2 = rng.normal(size=5), 0.3, 0.05
    _, gw, gb = loss_and_grad(w, b, x, y, l2)
    eps = 1e-6
    num_w = np.array([
        (loss_and_grad(w + eps * e, b, x, y, l2)[0] - loss_and_grad(w - eps * e, b, x, y, l2)[0])
        / (2 * eps)
        for e in np.eye(5)
    ])  # fmt: skip
    num_b = (loss_and_grad(w, b + eps, x, y, l2)[0] - loss_and_grad(w, b - eps, x, y, l2)[0]) / (
        2 * eps
    )
    assert np.linalg.norm(gw - num_w) / np.linalg.norm(num_w) < 1e-6
    assert gb == pytest.approx(num_b, rel=1e-6)


def test_matches_sklearn() -> None:
    x, y = _data()
    l2 = 0.01
    w, b = fit_logreg(x, y, l2=l2)
    ref = LogisticRegression(C=1 / (l2 * len(y)), tol=1e-12, max_iter=10_000).fit(x, y)
    assert np.abs(w - ref.coef_[0]).max() < 1e-5
    assert abs(b - ref.intercept_[0]) < 1e-5
    assert np.abs(predict_proba(x, w, b) - ref.predict_proba(x)[:, 1]).max() < 1e-6


def test_intercept_not_penalised() -> None:
    # No features carry signal; with heavy L2 the intercept alone fits the base rate.
    x = np.zeros((1000, 3))
    y = np.r_[np.ones(100), np.zeros(900)]
    w, b = fit_logreg(x, y, l2=10.0)
    assert predict_proba(x, w, b)[0] == pytest.approx(0.1, abs=1e-8)


def test_converges_on_separable_data_with_l2() -> None:
    x = np.r_[np.linspace(-3, -1, 50), np.linspace(1, 3, 50)][:, None]
    y = np.r_[np.zeros(50), np.ones(50)]
    w, b = fit_logreg(x, y, l2=1e-3)
    assert np.isfinite(w).all() and np.isfinite(b)
    assert (predict_proba(x, w, b)[y == 1] > 0.5).all()


def test_extreme_scores_do_not_overflow() -> None:
    p = predict_proba(np.array([[1000.0], [-1000.0]]), np.array([1.0]), 0.0)
    assert p.tolist() == [1.0, 0.0]
