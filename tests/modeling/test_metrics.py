import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from late_delivery.modeling.metrics import average_precision, brier, ece, reliability_bins, roc_auc


def _cases() -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(7)
    out = []
    for i in range(20):
        n = int(rng.integers(20, 400))
        y = (rng.random(n) < rng.uniform(0.05, 0.5)).astype(int)
        y[0], y[1] = 0, 1  # both classes present
        s = rng.random(n)
        if i % 2:  # heavy ties
            s = np.round(s, 1)
        out.append((y, s))
    return out


@pytest.mark.parametrize("case", range(20))
def test_roc_auc_matches_sklearn(case: int) -> None:
    y, s = _cases()[case]
    assert abs(roc_auc(y, s) - roc_auc_score(y, s)) < 1e-12


@pytest.mark.parametrize("case", range(20))
def test_average_precision_matches_sklearn(case: int) -> None:
    y, s = _cases()[case]
    assert abs(average_precision(y, s) - average_precision_score(y, s)) < 1e-12


@pytest.mark.parametrize("case", range(20))
def test_brier_matches_sklearn(case: int) -> None:
    y, s = _cases()[case]
    assert abs(brier(y, s) - brier_score_loss(y, s)) < 1e-12


def test_single_class_gives_nan() -> None:
    y, s = np.zeros(5, dtype=int), np.linspace(0, 1, 5)
    assert np.isnan(roc_auc(y, s))
    assert np.isnan(average_precision(y, s))


def test_reliability_bins_equal_count() -> None:
    p = np.array([0.1] * 50 + [0.9] * 50)
    y = np.array([0] * 45 + [1] * 5 + [1] * 45 + [0] * 5)
    bins = reliability_bins(y, p, n_bins=2)
    assert bins["n"].to_list() == [50, 50]
    assert bins["mean_pred"].to_list() == pytest.approx([0.1, 0.9])
    assert bins["frac_late"].to_list() == pytest.approx([0.1, 0.9])
    assert ece(y, p, n_bins=2) == pytest.approx(0.0)


def test_ece_of_overconfident_scores() -> None:
    p = np.full(100, 0.5)
    y = np.array([1] * 10 + [0] * 90)
    assert ece(y, p, n_bins=1) == pytest.approx(0.4)


@pytest.mark.parametrize("case", range(6))
def test_pr_points_match_sklearn(case: int) -> None:
    from sklearn.metrics import precision_recall_curve

    from late_delivery.modeling.metrics import pr_points

    y, s = _cases()[case]
    pts = pr_points(y, s)
    p, r, _ = precision_recall_curve(y, s)
    # sklearn lists points from high threshold to low with a final (1, 0) anchor reversed
    assert np.allclose(sorted(zip(pts["recall"], pts["precision"], strict=True)),
                       sorted(zip(r[:-1], p[:-1], strict=True)))  # fmt: skip


def test_calibration_of_empty_input_is_nan_without_warning() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(ece(np.array([]), np.array([])))
        assert reliability_bins(np.array([]), np.array([])).height == 0
