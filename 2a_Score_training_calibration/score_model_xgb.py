# Imports for score training, persistence, and prediction.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

def train_score_model(
    X,
    y,
    *,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    max_bin=64,
    n_jobs=4,
    random_state=42,
):
    """Train one LLM's score classifier."""

    model = XGBClassifier(
        objective="multi:softprob",
        tree_method="hist",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_bin=max_bin,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    model.fit(X, y)
    return model



def load_score_models(folder):
    """Load saved classifiers and their metadata."""
    folder = Path(folder)

    with (folder / "metadata.json").open() as file:
        metadata = json.load(file)

    models = {}

    for name, filename in metadata["model_files"].items():
        model = XGBClassifier(n_jobs=4)

        # Compatibility workaround for older XGBoost wrappers.
        if not hasattr(model, "_estimator_type"):
            model._estimator_type = "classifier"

        model.load_model(folder / filename)
        models[name] = model

    return models, metadata

def predict_scores(X, models, metadata):
    """Return uncalibrated expected scores, one column per LLM."""
    if X.shape[1] != metadata["n_features"]:
        raise ValueError("Feature count does not match the saved models.")

    score_values = np.array(metadata["score_values"])
    predictions = {}

    for name, model in models.items():
        probabilities = model.predict_proba(X)
        class_scores = score_values[model.classes_.astype(int)]

        predictions[name] = (
            probabilities * class_scores
        ).sum(axis=1)

    return pd.DataFrame(predictions)