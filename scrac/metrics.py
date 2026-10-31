"""Evaluation metrics following the manuscript's Appendix (correlations, errors, CMD_5).

CMD is a discrepancy between the first five *marginal central moments*, NOT
an inter-cell/inter-protein correlation-matrix distance. All calculations of
central moments use float64, the population empirical moment (denominator n),
and no feature-range rescaling: this matches the equation in Appendix CMD.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


METRIC_SCHEMA = "paper_cmd5_v1"
SCALAR_METRIC_KEYS = (
    "pcc_protein_mean", "pcc_cell_mean",
    "scc_protein_mean", "scc_cell_mean",
    "cmd_protein", "cmd_cell", "mse", "rmse", "mmd",
)


def _validate_pairs(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(prediction)
    y = np.asarray(target)
    if p.ndim != 2 or y.ndim != 2 or p.shape != y.shape:
        raise ValueError("Prediction and target must be aligned 2D (cells x proteins) matrices.")
    if p.size == 0 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Prediction and target must be nonempty and finite.")
    return p, y


def _correlation_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r for each row; zero for any constant row (explicit convention)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    numerator = np.sum(a_centered * b_centered, axis=1)
    denominator = np.sqrt(
        np.sum(a_centered**2, axis=1) * np.sum(b_centered**2, axis=1)
    )
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def _correlation_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _correlation_rows(a.T, b.T)


def protein_pearson_mean(prediction: np.ndarray, target: np.ndarray) -> float:
    p, y = _validate_pairs(prediction, target)
    return float(_correlation_columns(p, y).mean())


def _spearman_by_axis(prediction: np.ndarray, target: np.ndarray, *, axis: int) -> np.ndarray:
    """Spearman r = Pearson correlation of average-tie ranks, per feature/cell."""
    p_rank = rankdata(prediction, method="average", axis=axis)
    y_rank = rankdata(target, method="average", axis=axis)
    if axis == 0:  # columns are proteins
        return _correlation_columns(p_rank, y_rank)
    return _correlation_rows(p_rank, y_rank)


def central_moment_discrepancy(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    axis: str = "protein",
    order: int = 5,
    chunk_size: int = 4096,
) -> float:
    r"""Appendix CMD_M: average over variables of |mu_p-mu_y| + sum_m |c_m(p)-c_m(y)|.

    c_m(U) = (1/n) sum_i (u_i - mean(U))**m, for m=2..M.
    axis='protein' treats each column as one variable over cells (CMD-P).
    axis='cell' treats each row as one variable over proteins (CMD-C).
    No sample SD correction, clipping, standardization or unreported scaling.
    Streaming row chunks bound memory consumption for large CITE-seq datasets.
    """
    p, y = _validate_pairs(prediction, target)
    if axis not in ("protein", "cell"):
        raise ValueError("axis must be 'protein' or 'cell'.")
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be an integer >= 1.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    n_cells, n_proteins = p.shape
    if axis == "protein":
        mu_p = p.mean(axis=0, dtype=np.float64)
        mu_y = y.mean(axis=0, dtype=np.float64)
        discrepancy = np.abs(mu_p - mu_y)
        # Separate sums for each central moment. The means are calculated over
        # all held-out cells, not separately for every processing chunk.
        sums_p = np.zeros((max(order - 1, 0), n_proteins), dtype=np.float64)
        sums_y = np.zeros_like(sums_p)
        for start in range(0, n_cells, chunk_size):
            a = np.asarray(p[start:start + chunk_size], dtype=np.float64) - mu_p
            b = np.asarray(y[start:start + chunk_size], dtype=np.float64) - mu_y
            a_power, b_power = a * a, b * b
            for idx in range(order - 1):
                sums_p[idx] += a_power.sum(axis=0)
                sums_y[idx] += b_power.sum(axis=0)
                a_power *= a
                b_power *= b
        for idx in range(order - 1):
            discrepancy += np.abs(sums_p[idx] / n_cells - sums_y[idx] / n_cells)
        return float(discrepancy.mean())

    # Each row is a separate scalar variable: independent mean and moments
    # across its protein channels, followed by averaging over all cells.
    discrepancy_sum = 0.0
    for start in range(0, n_cells, chunk_size):
        a = np.asarray(p[start:start + chunk_size], dtype=np.float64)
        b = np.asarray(y[start:start + chunk_size], dtype=np.float64)
        mu_a = a.mean(axis=1, keepdims=True)
        mu_b = b.mean(axis=1, keepdims=True)
        row_differences = np.abs(mu_a[:, 0] - mu_b[:, 0])
        a = a - mu_a
        b = b - mu_b
        a_power, b_power = a * a, b * b
        for _ in range(2, order + 1):
            row_differences += np.abs(a_power.mean(axis=1) - b_power.mean(axis=1))
            a_power *= a
            b_power *= b
        discrepancy_sum += float(row_differences.sum())
    return discrepancy_sum / n_cells


def rbf_mmd(
    prediction: np.ndarray,
    target: np.ndarray,
    max_samples: int = 2000,
    seed: int = 42,
) -> float:
    """Optional diagnostic; not substituted for CMD in paper comparisons."""
    rng = np.random.default_rng(seed)
    if prediction.shape[0] > max_samples:
        prediction = prediction[rng.choice(prediction.shape[0], max_samples, replace=False)]
    if target.shape[0] > max_samples:
        target = target[rng.choice(target.shape[0], max_samples, replace=False)]
    x = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        squared = (
            np.sum(a * a, axis=1, keepdims=True)
            + np.sum(b * b, axis=1, keepdims=True).T
            - 2.0 * (a @ b.T)
        )
        squared = np.maximum(squared, 0.0) / a.shape[1]
        return np.exp(-squared)

    xx = kernel(x, x)
    yy = kernel(y, y)
    xy = kernel(x, y)
    return float(xx.mean() + yy.mean() - 2.0 * xy.mean())


def evaluate(prediction: np.ndarray, target: np.ndarray, seed: int = 42) -> dict[str, object]:
    """Evaluate one held-out fold, never pre-average predictions from five folds."""
    p, y = _validate_pairs(prediction, target)
    pcc_p = _correlation_columns(p, y)
    pcc_c = _correlation_rows(p, y)
    scc_p = _spearman_by_axis(p, y, axis=0)
    scc_c = _spearman_by_axis(p, y, axis=1)
    mse = float(np.mean((p.astype(np.float64) - y.astype(np.float64)) ** 2))
    return {
        "metric_schema": METRIC_SCHEMA,
        "cmd_order": 5,
        "pcc_protein": pcc_p.astype(np.float32),
        "pcc_cell": pcc_c.astype(np.float32),
        "scc_protein": scc_p.astype(np.float32),
        "scc_cell": scc_c.astype(np.float32),
        "pcc_protein_mean": float(pcc_p.mean()),
        "pcc_cell_mean": float(pcc_c.mean()),
        "scc_protein_mean": float(scc_p.mean()),
        "scc_cell_mean": float(scc_c.mean()),
        "cmd_protein": central_moment_discrepancy(p, y, axis="protein", order=5),
        "cmd_cell": central_moment_discrepancy(p, y, axis="cell", order=5),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mmd": rbf_mmd(p, y, seed=seed),
    }


def summarize_five_folds(fold_metrics: list[dict[str, object]]) -> dict[str, object]:
    """Paper main tables: mean and SAMPLE SD of per-fold metrics (ddof=1)."""
    if len(fold_metrics) != 5:
        raise ValueError("The paper five-fold summary requires exactly five held-out folds.")
    for metrics in fold_metrics:
        if metrics.get("metric_schema") != METRIC_SCHEMA:
            raise ValueError("Mixed metric versions must not be pooled.")
    summary = {
        "metric_schema": METRIC_SCHEMA,
        "folds": len(fold_metrics),
        "aggregation": "mean_and_sample_sd_of_five_heldout_fold_metrics",
        "metrics": {},
    }
    for key in SCALAR_METRIC_KEYS:
        values = np.asarray([m[key] for m in fold_metrics], dtype=np.float64)
        summary["metrics"][key] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "per_fold": values.tolist(),
        }
    return summary
