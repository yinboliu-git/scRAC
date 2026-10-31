"""Regression checks for the explicit mathematical definitions in the manuscript."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scrac.losses import flow_matching_objective
from scrac.metrics import (
    METRIC_SCHEMA, central_moment_discrepancy, evaluate, summarize_five_folds,
)


def manual_cmd(prediction: np.ndarray, target: np.ndarray, *, columns: bool) -> float:
    """Independent slow reference implementation, one variable at a time."""
    if columns:
        prediction, target = prediction.T, target.T
    discrepancies = []
    for a, b in zip(prediction, target):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        d = abs(a.mean() - b.mean())
        for m in range(2, 6):
            d += abs(np.mean((a - a.mean()) ** m) - np.mean((b - b.mean()) ** m))
        discrepancies.append(d)
    return float(np.mean(discrepancies))


def test_cmd5_exact_manuscript_definition() -> None:
    prediction = np.array([[0.5, 3., 4.], [2., 1., 6.], [3.5, 7., 2.], [4., 2., 5.]])
    target = np.array([[0., 4., 4.], [2., 2., 5.], [4., 6., 1.], [3., 0., 6.]])
    assert central_moment_discrepancy(prediction, target, axis="protein", chunk_size=2) == pytest.approx(
        manual_cmd(prediction, target, columns=True), abs=1e-11
    )
    assert central_moment_discrepancy(prediction, target, axis="cell", chunk_size=2) == pytest.approx(
        manual_cmd(prediction, target, columns=False), abs=1e-11
    )


def test_cmd5_identity_mean_shift_and_robust_chunks() -> None:
    rng = np.random.default_rng(23)
    target = rng.normal(size=(19, 7))
    assert central_moment_discrepancy(target, target, axis="protein") == pytest.approx(0.)
    assert central_moment_discrepancy(target, target, axis="cell") == pytest.approx(0.)
    assert central_moment_discrepancy(target + 2, target, axis="protein", chunk_size=1) == pytest.approx(2.)
    assert central_moment_discrepancy(target + 2, target, axis="cell", chunk_size=3) == pytest.approx(2.)
    assert central_moment_discrepancy(target * 1.3, target, axis="protein", chunk_size=1) == pytest.approx(
        central_moment_discrepancy(target * 1.3, target, axis="protein", chunk_size=7)
    )


def test_cmd_is_not_correlation_matrix_cosine_distance() -> None:
    y = np.arange(12, dtype=float).reshape(4, 3)
    p = y + 1
    # A constant shift does not change correlation matrices. It DOES change CMD.
    assert central_moment_discrepancy(p, y, axis="protein") == pytest.approx(1.)
    assert central_moment_discrepancy(p, y, axis="cell") == pytest.approx(1.)


def test_evaluate_has_paper_spearman_and_errors() -> None:
    target = np.array([[1., 8., 3.], [4., 2., 6.], [5., 7., 1.], [3., 5., 2.]])
    prediction = np.array([[3., 8., 2.], [5., 4., 7.], [6., 6., 0.], [1., 2., 4.]])
    output = evaluate(prediction, target)
    assert output["metric_schema"] == METRIC_SCHEMA
    assert output["cmd_order"] == 5
    assert output["cmd_protein"] == pytest.approx(manual_cmd(prediction, target, columns=True))
    assert output["cmd_cell"] == pytest.approx(manual_cmd(prediction, target, columns=False))
    expected_scc_p = np.mean([spearmanr(prediction[:, i], target[:, i]).statistic for i in range(3)])
    expected_scc_c = np.mean([spearmanr(prediction[i], target[i]).statistic for i in range(4)])
    assert output["scc_protein_mean"] == pytest.approx(expected_scc_p)
    assert output["scc_cell_mean"] == pytest.approx(expected_scc_c)
    expected_mse = np.mean((prediction - target)**2)
    assert output["mse"] == pytest.approx(expected_mse)
    assert output["rmse"] == pytest.approx(np.sqrt(expected_mse))


def test_constant_rows_and_columns_have_defined_zero_correlations() -> None:
    output = evaluate(np.ones((4, 3)), np.ones((4, 3)))
    for key in ("pcc_protein_mean", "pcc_cell_mean", "scc_protein_mean", "scc_cell_mean"):
        assert output[key] == 0.


def test_five_fold_uses_sample_sd_and_is_not_pooled() -> None:
    rows = []
    for i in range(5):
        row = {"metric_schema": METRIC_SCHEMA}
        from scrac.metrics import SCALAR_METRIC_KEYS
        for key in SCALAR_METRIC_KEYS:
            row[key] = float(i)
        rows.append(row)
    stats = summarize_five_folds(rows)
    assert stats["metrics"]["cmd_protein"]["mean"] == pytest.approx(2.)
    assert stats["metrics"]["cmd_protein"]["sample_sd"] == pytest.approx(np.sqrt(2.5))
    with pytest.raises(ValueError, match="exactly five"):
        summarize_five_folds(rows[:4])


def test_flow_objectives_use_paper_squared_l2_norm() -> None:
    class ZeroVelocity(torch.nn.Module):
        def forward(self, x, t, latent, similarity):
            return torch.zeros_like(x)
    source = torch.ones(2, 3)
    target = torch.zeros(2, 3)
    result = flow_matching_objective(
        ZeroVelocity(), source, target,
        torch.zeros(2, 128), torch.zeros(2, 1),
        mask_probability=0., endpoint_weight=1.,
    )
    assert result.endpoint.item() == pytest.approx(3.)
    assert result.flow_matching.item() == pytest.approx(3.)
    assert result.total.item() == pytest.approx(6.)
