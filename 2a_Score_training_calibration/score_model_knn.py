from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors


def fit_score_knn(X, scores, *, n_jobs=-1):
    targets = np.asarray(scores, dtype=float)

    if targets.ndim != 2 or targets.shape[0] != X.shape[0]:
        raise ValueError("Scores must have one row per training prompt.")

    if not np.isfinite(targets).all():
        raise ValueError("Scores must be finite.")

    if np.any((targets < 0) | (targets > 1)):
        raise ValueError("Scores must lie between 0 and 1.")

    search = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_jobs=n_jobs,
    )
    search.fit(X)

    return {
        "search": search,
        "targets": targets,
    }


def predict_score_grid(model, X, *, k_values, batch_size=128):
    k_values = sorted(set(k_values))
    targets = model["targets"]

    if not k_values or any(
        not isinstance(k, (int, np.integer))
        or k < 1
        or k > len(targets)
        for k in k_values
    ):
        raise ValueError(
            "Each k must be an integer from 1 to the training size."
        )

    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    predictions = {
        k: np.empty((X.shape[0], targets.shape[1]), dtype=float)
        for k in k_values
    }

    for start in range(0, X.shape[0], batch_size):
        stop = min(start + batch_size, X.shape[0])

        neighbors = model["search"].kneighbors(
            X[start:stop],
            n_neighbors=max(k_values),
            return_distance=False,
        )

        cumulative_scores = targets[neighbors].cumsum(axis=1)

        for k in k_values:
            predictions[k][start:stop] = (
                cumulative_scores[:, k - 1, :] / k
            )

    return predictions


def cross_validate_score_knn(
    texts,
    scores,
    groups,
    cv_splits,
    vectorizer,
    *,
    k_values,
    batch_size=128,
    n_jobs=-1,
):
    texts = np.asarray(texts)
    targets = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    k_values = sorted(set(k_values))
    cv_splits = list(cv_splits)

    if not (len(texts) == len(targets) == len(groups)):
        raise ValueError(
            "Texts, scores, and groups must have matching rows."
        )

    coverage = np.zeros(len(texts), dtype=int)

    for fit_positions, heldout_positions in cv_splits:
        if np.intersect1d(
            groups[fit_positions],
            groups[heldout_positions],
        ).size:
            raise ValueError(
                "A prompt group appears on both sides of a fold."
            )

        np.add.at(coverage, heldout_positions, 1)

    if not np.all(coverage == 1):
        raise ValueError("Each prompt must be held out exactly once.")

    squared_error_sums = np.zeros(
        (len(k_values), targets.shape[1])
    )
    baseline_error_sum = 0.0
    fold_rows = []

    for fold, (fit_positions, heldout_positions) in enumerate(
        cv_splits,
        start=1,
    ):
        start = perf_counter()

        print(
            f"Fold {fold}/{len(cv_splits)}: fitting TF-IDF",
            flush=True,
        )

        fold_vectorizer = clone(vectorizer)

        X_fit = fold_vectorizer.fit_transform(
            texts[fit_positions]
        )
        X_heldout = fold_vectorizer.transform(
            texts[heldout_positions]
        )

        model = fit_score_knn(
            X_fit,
            targets[fit_positions],
            n_jobs=n_jobs,
        )

        print(
            f"Fold {fold}: predicting {len(heldout_positions)} prompts",
            flush=True,
        )

        predictions = predict_score_grid(
            model,
            X_heldout,
            k_values=k_values,
            batch_size=batch_size,
        )

        observed = targets[heldout_positions]
        baseline = targets[fit_positions].mean(axis=0)

        baseline_error_sum += (
            (observed - baseline) ** 2
        ).sum()

        for position, k in enumerate(k_values):
            squared_error_sums[position] += (
                (predictions[k] - observed) ** 2
            ).sum(axis=0)

        seconds = perf_counter() - start

        fold_rows.append({
            "fold": fold,
            "fit_prompts": len(fit_positions),
            "heldout_prompts": len(heldout_positions),
            "features": X_fit.shape[1],
            "zero_fit_rows": int(
                (X_fit.getnnz(axis=1) == 0).sum()
            ),
            "zero_heldout_rows": int(
                (X_heldout.getnnz(axis=1) == 0).sum()
            ),
            "seconds": seconds,
        })

        print(
            f"Fold {fold} finished in {seconds:.1f} seconds",
            flush=True,
        )

        del model, X_fit, X_heldout, predictions, fold_vectorizer

    mse_by_model = pd.DataFrame(
        squared_error_sums / len(texts),
        index=pd.Index(k_values, name="k"),
        columns=scores.columns,
    )

    baseline_mse = baseline_error_sum / targets.size

    summary = pd.DataFrame({
        "k": k_values,
        "CV_MSE": mse_by_model.mean(axis=1).to_numpy(),
        "baseline_CV_MSE": baseline_mse,
    })

    summary["MSE_reduction_pct"] = (
        100 * (1 - summary["CV_MSE"] / baseline_mse)
        if baseline_mse > 0 else np.nan
    )

    return {
        "summary": summary.sort_values(
            ["CV_MSE", "k"],
            ascending=[True, False],
        ).reset_index(drop=True),
        "mse_by_model": mse_by_model,
        "folds": pd.DataFrame(fold_rows).set_index("fold"),
    }