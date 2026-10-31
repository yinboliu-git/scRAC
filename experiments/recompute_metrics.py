"""Re-evaluate existing five-fold prediction files using the manuscript CMD_5 formula.

Usage:
    python experiments/recompute_metrics.py --run-dir outputs/cbmc

Creates a *new* paper_cmd5_recomputed/ directory. Original metrics and model
artifacts are not overwritten. Requires outputs.npz for every fold; therefore
no GPU/retraining is necessary for the metric correction itself.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scrac.metrics import METRIC_SCHEMA, evaluate, summarize_five_folds
from scrac.training import save_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Default: <run-dir>/paper_cmd5_recomputed")
    args = parser.parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir / "paper_cmd5_recomputed"

    fold_metrics: list[dict[str, object]] = []
    prior_metrics: list[dict[str, object]] = []
    all_test_indices: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    sources: list[dict[str, object]] = []
    for fold_id in range(5):
        path = run_dir / f"fold_{fold_id}" / "outputs.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing fold data (no published value will be guessed): {path}")
        with np.load(path, allow_pickle=False) as data:
            required = ("prediction", "target", "retrieval_source", "test_indices")
            if any(k not in data for k in required):
                raise ValueError(f"{path} is missing required arrays: {required}")
            prediction = data["prediction"].copy()
            target = data["target"].copy()
            source = data["retrieval_source"].copy()
            indices = data["test_indices"].copy()
        if prediction.shape != target.shape or source.shape != target.shape:
            raise ValueError(f"Mismatched protein panels or cell ordering in {path}")
        if len(indices) != len(target) or indices.ndim != 1:
            raise ValueError(f"Invalid held-out indices in {path}")
        current = evaluate(prediction, target)
        prior = evaluate(source, target)
        fold_metrics.append(current)
        prior_metrics.append(prior)
        all_predictions.append(prediction)
        all_targets.append(target)
        all_test_indices.append(indices)
        sources.append({"fold": fold_id, "file": str(path), "sha256": sha256(path)})
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / f"fold_{fold_id}_prediction_metrics.json", current)
        save_json(output_dir / f"fold_{fold_id}_retrieval_metrics.json", prior)

    indices = np.concatenate(all_test_indices)
    if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise ValueError("Held-out indices overlap or include invalid negatives")
    if not np.array_equal(np.sort(indices), np.arange(len(indices))):
        raise ValueError("Held-out indices do not cover each cell exactly once")
    order = np.argsort(indices)
    pooled_predictions = np.concatenate(all_predictions, axis=0)[order]
    pooled_targets = np.concatenate(all_targets, axis=0)[order]
    five_fold = summarize_five_folds(fold_metrics)
    retrieval_five_fold = summarize_five_folds(prior_metrics)
    save_json(output_dir / "five_fold_metrics.json", five_fold)
    save_json(output_dir / "five_fold_retrieval_metrics.json", retrieval_five_fold)
    save_json(output_dir / "pooled_metrics_diagnostic.json", evaluate(pooled_predictions, pooled_targets))
    save_json(output_dir / "provenance.json", {
        "metric_schema": METRIC_SCHEMA,
        "description": "Re-evaluation of existing fold predictions; no model fitting and no overwrite",
        "source_files": sources,
    })
    print(f"Saved corrected paper metrics in {output_dir}")
    print("Main-table PCC-P mean +/- sample SD:", five_fold["metrics"]["pcc_protein_mean"])
    print("Main-table CMD-P mean +/- sample SD:", five_fold["metrics"]["cmd_protein"])


if __name__ == "__main__":
    main()
