# Imports for cost training, persistence, and prediction.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def train_cost_model(X, y, **overrides):
    """Train one LLM's cost regressor in the supplied target units."""

    parameters = {
        "objective": "reg:squarederror",
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_bin": 64,
        "tree_method": "hist",
        "device": "cpu",
        "n_jobs": 4,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    parameters.update(overrides)

    model = XGBRegressor(**parameters)
    model.fit(X, y)

    return model


# Load saved cost models and their training normalization.
def load_cost_models(folder):
    """Load saved cost regressors and their normalization metadata."""
    folder = Path(folder)

    with (folder / "metadata.json").open() as file:
        metadata = json.load(file)

    models = {}

    for name, filename in metadata["model_files"].items():
        model = XGBRegressor(n_jobs=4)

        if not hasattr(model, "_estimator_type"):
            model._estimator_type = "regressor"

        model.load_model(folder / filename)
        models[name] = model

    return models, metadata


# Convert normalized predictions into dollars per prompt.
def predict_costs(X, models, metadata, *, index=None):
    """Return predicted dollars per prompt, with one column per LLM."""
    if X.shape[1] != metadata["n_features"]:
        raise ValueError("Feature count does not match the saved cost models.")

    if metadata["target_transform"] != "divide_by_training_mean":
        raise ValueError("Unsupported cost normalization.")

    predictions = {}

    for name, model in models.items():
        normalized = model.predict(X)

        if not np.isfinite(normalized).all():
            raise ValueError(f"Nonfinite cost predictions for {name}.")

        predictions[name] = (
            np.maximum(normalized, metadata["normalized_prediction_floor"])
            * metadata["training_mean_costs"][name]
        )

    return pd.DataFrame(predictions, index=index)
