# RouterBench — Regression Model Training Handoff

## 1. Overview

This document describes 44 trained reference cost-prediction
models developed for the Erdős Institute LLM-routing project.

The objective is to predict LLM inference costs from question
features, providing teammates with pretrained models for
subsequent calibration, evaluation, and routing experiments.

Four model families were trained:

| Regression | Features | Models |
|---|---|---:|
| Gamma | Character TF-IDF + length | 11 |
| Log-normal | Character TF-IDF + length | 11 |
| Gamma | E5 embeddings + length | 11 |
| Log-normal | E5 embeddings + length | 11 |
| **Total** | | **44** |

All four pipelines are complete, and their saved model bundles
have been reloaded and verified against the original parameters
and predictions.

## 2. Data Usage — Important

All final training notebooks use only the original RouterBench
training partition containing 21,900 observations.

The training partition was divided into:

- Internal sub-training: 17,520 observations.
- Internal sub-validation: 4,380 observations.

A group-aware split was used, with random_state=42 and
question_group as the grouping variable.

The internal sub-training observations were used to fit coefficients.
The internal sub-validation observations were used exclusively for
early stopping and checkpoint selection.

**The original calibration, validation, and test datasets were
not loaded or used by these final training notebooks.**

Those original datasets are reserved for subsequent team work.

The internal sub-validation set is NOT the original RouterBench
validation set.

The saved split is located at:

    saved_models/training_splits/
        routerbench_grouped_subtrain_subval_seed42.npz

It contains:

    subtrain_idx
    subval_idx

Note: The saved character TF-IDF preprocessing and length scaler
were fitted before this internal split. Therefore, internal
sub-validation estimates should not be treated as fully
preprocessing-independent holdout estimates.

## 3. Feature Representations

### 3.1 Character TF-IDF + Length

The TF-IDF representation consists of:

- 50,000 character TF-IDF features.
- One scaled character-length feature.

Total: 50,001 features.

The original training feature matrix has shape:

    (21900, 50001)

Its files are located under:

    1_data_preparation/data/processed/char_tfidf_50k/

Relevant files:

    X_train.npz
    train_costs.pkl
    train_questions.pkl
    preprocessing.joblib

The preprocessing.joblib dictionary includes:

    tfidf_vectorizer
    length_scaler
    feature_order

For new questions, reuse the saved vectorizer and scaler.
Do not refit them.

### 3.2 E5 Embeddings + Length

The E5 representation consists of:

- 384 E5 embedding features.
- One scaled character-length feature.

Total: 385 features.

The training embeddings are stored at:

    1_data_preparation/data/processed/e5_small/E5_train.npy

The resulting feature matrix has shape:

    (21900, 385)

The recorded embedding model is:

    intfloat/multilingual-e5-small

The same scaled character-length feature used by the TF-IDF
pipeline is appended as the final column.

Important: For new questions, the original E5 encoding pipeline
must be reproduced, including text prefixes, tokenization,
normalization, and preprocessing settings.

The exact embedding-generation procedure should be verified
against the original data-preparation notebook before inference.
Embedding dimensions alone do not guarantee compatibility.

The row order of E5_train.npy must match the corresponding
original training questions.

## 4. Training Methodology

Both regression families used:

| Parameter | Value |
|---|---|
| Optimizer | L-BFGS-B |
| L2 regularization lambda | 0.1 |
| Maximum optimizer iterations | 200 |
| Early-stopping patience | 5 |
| Minimum improvement | 1e-6 |
| Optimizer ftol | 1e-12 |
| Optimizer gtol | 1e-8 |
| Checkpoint | Minimum internal sub-validation loss |

The intercept was not penalized.

For each model, the retained parameter vector contains the
intercept first, followed by the feature coefficients.

Gamma + TF-IDF: 50,002 parameters.
Log-normal + TF-IDF: 50,002 parameters.
Gamma + E5: 386 parameters.
Log-normal + E5: 386 parameters.

The lambda above is the *regression regularization coefficient*,
not a downstream routing cost-quality tradeoff parameter.

### 4.1 Gamma Regression

Gamma regression uses a log link:

    eta = intercept + X @ coefficients
    predicted_mean_cost = exp(eta)

The regularized training objective is:

    0.5 * mean_gamma_deviance
    + 0.5 * lambda * sum(coefficients**2)

Mean Gamma deviance is:

    2 * mean(y / mu - 1 - log(y / mu))

Early stopping monitors unpenalized Gamma deviance on the
internal sub-validation partition.

These are model-based conditional mean-cost predictions.
No additional downstream calibration was performed.

### 4.2 Log-Normal Regression

The log-normal regression predicts log-cost:

    eta = intercept + X @ coefficients

The regularized training objective is:

    0.5 * mean((log(y) - eta)**2)
    + 0.5 * lambda * sum(coefficients**2)

Early stopping monitors unpenalized log-MSE on the internal
sub-validation partition.

Exponentiating eta produces a positive median/geometric-location
cost estimate under the usual log-normal assumptions:

    exp(eta)

This is NOT automatically the conditional expected cost.

A retransformation or calibration correction may be needed before
using the log-normal predictions for expected-cost routing.

The original calibration data has been left untouched for teammates.

## 5. Saved Model Artifacts

All paths below are relative to the Erdős Router project root.

### 5.1 Gamma — TF-IDF + Length

    saved_models/gamma_final/
        gamma_tfidf_length_early_stopped.joblib
        gamma_tfidf_length_training_history.csv
        gamma_tfidf_length_training_summary.csv

### 5.2 Log-Normal — TF-IDF + Length

    saved_models/lognormal_final/
        lognormal_tfidf_length_early_stopped.joblib
        lognormal_tfidf_length_training_history.csv
        lognormal_tfidf_length_training_summary.csv

### 5.3 Gamma — E5 + Length

    saved_models/gamma_final/
        gamma_e5_length_early_stopped.joblib
        gamma_e5_length_training_history.csv
        gamma_e5_length_training_summary.csv

### 5.4 Log-Normal — E5 + Length

    saved_models/lognormal_final/
        lognormal_e5_length_early_stopped.joblib
        lognormal_e5_length_training_history.csv
        lognormal_e5_length_training_summary.csv

The .joblib files contain all 11 parameter vectors for their
respective families, along with training metadata.

The CSV files preserve iteration histories and training summaries.

## 6. Load All Four Model Bundles

Run the following from a trusted local environment with the
required Python dependencies installed.

```python
from pathlib import Path
import joblib

ROOT = Path.home() / "Desktop" / "Erdos-Router"

paths = {
    "gamma_tfidf":
        ROOT / "saved_models/gamma_final/"
        "gamma_tfidf_length_early_stopped.joblib",

    "lognormal_tfidf":
        ROOT / "saved_models/lognormal_final/"
        "lognormal_tfidf_length_early_stopped.joblib",

    "gamma_e5":
        ROOT / "saved_models/gamma_final/"
        "gamma_e5_length_early_stopped.joblib",

    "lognormal_e5":
        ROOT / "saved_models/lognormal_final/"
        "lognormal_e5_length_early_stopped.joblib",
}

bundles = {
    name: joblib.load(path)
    for name, path in paths.items()
}

for name, bundle in bundles.items():
    print(name, len(bundle["models"]))
```

Expected: four families, each containing 11 models.

Security note: Only load joblib files from trusted sources.
joblib loading can execute Python code.

## 7. Generate Predictions from Prepared Features

The following helper accepts a saved bundle and an already
prepared feature matrix.

```python
import numpy as np

def predict_from_bundle(bundle, X):
    expected_features = bundle["feature_count"]

    if X.shape[1] != expected_features:
        raise ValueError("Incorrect feature dimension.")

    predictions = {}

    for model_name in bundle["model_names"]:
        params = bundle["models"][model_name]

        eta = params[0] + X @ params[1:]
        eta = np.asarray(eta).ravel()

        if bundle["model_type"] == "gamma_log_link_l2":
            predictions[model_name] = np.exp(eta)

        elif bundle["model_type"] == "lognormal_log_cost_l2":
            # Preserve the raw log-cost predictions.
            predictions[model_name] = eta

        else:
            raise ValueError("Unknown model type.")

    return predictions
```

For Gamma, outputs estimate mean inference cost.

For log-normal, outputs are on the log-cost scale.
Do not confuse log-cost predictions with expected costs.

The feature matrix must match the corresponding representation:

    TF-IDF + length: 50001 columns
    E5 + length:      385 columns

## 8. Prepare TF-IDF Features for New Questions

Example:

```python
import numpy as np
import pandas as pd
from scipy import sparse

bundle = bundles["gamma_tfidf"]
prep = bundle["preprocessing"]

questions = [
    "Explain the central limit theorem.",
    "Write a Python function for binary search."
]

tfidf = prep["tfidf_vectorizer"].transform(questions)

lengths = pd.DataFrame({
    "char_length": [len(q) for q in questions]
})

scaled_length = prep["length_scaler"].transform(lengths)

X_new_tfidf = sparse.hstack(
    [tfidf, scaled_length],
    format="csr"
)

predictions = predict_from_bundle(
    bundle, X_new_tfidf
)
```

Use the corresponding log-normal TF-IDF bundle to obtain
log-cost predictions for the same feature matrix.

## 9. Prepare E5 Features for New Questions

The original E5 data-preparation pipeline must be reused.

Conceptual steps:

1. Load the original E5 encoder.
2. Apply its exact training-time text formatting.
3. Encode the questions using the same normalization settings.
4. Obtain 384-dimensional embeddings.
5. Transform character lengths with the saved scaler.
6. Concatenate embeddings and scaled lengths in that order.

The Gamma and log-normal E5 bundles contain a
length_preprocessing scaler.

Once compatible E5 embeddings are available:

```python
import numpy as np
import pandas as pd

bundle = bundles["gamma_e5"]

# e5_embeddings must have been generated using the
# original E5 preprocessing and encoding procedure.
# Shape: (number_of_questions, 384)

questions = [
    "Explain the central limit theorem.",
    "Write a Python function for binary search."
]

lengths = pd.DataFrame({
    "char_length": [len(q) for q in questions]
})

scaled_length = bundle[
    "length_preprocessing"
].transform(lengths)

X_new_e5 = np.hstack([
    e5_embeddings,
    scaled_length
])

predictions = predict_from_bundle(
    bundle, X_new_e5
)
```

This is an inference template, not a complete embedding-generation
implementation. Verify the original E5 encoder settings first.

## 10. Reproducibility and Verification

The training notebooks were developed independently for the
four regression/feature combinations.

Each notebook:

- Loads the original training partition.
- Reuses the saved group-aware internal split.
- Defines the training objective and analytical gradient.
- Verifies the gradient numerically.
- Runs L-BFGS-B optimization with early stopping.
- Retains the best internal sub-validation checkpoint.
- Exports all 11 trained models and training histories.
- Reloads and verifies saved model parameters and predictions.

The saved E5 Gamma and E5 log-normal bundles were checked for
identical parameters and predictions following serialization.

The TF-IDF Gamma and log-normal bundles were likewise verified.

These checks establish serialization consistency, not independent
generalization performance.

## 11. Work Reserved for Teammates

The following tasks were intentionally excluded:

- Calibration using the original calibration partition.
- Model selection using the original validation partition.
- Final evaluation using the original test partition.
- Selection of downstream routing tradeoff parameters.
- Final cost-quality routing evaluation.
- Comparison against other routing methods.

Any calibration or selection should use the appropriate reserved
data under the team's agreed experimental protocol.

## 12. Handoff Checklist

- [x] Gamma + TF-IDF models trained and saved.
- [x] Log-normal + TF-IDF models trained and saved.
- [x] Gamma + E5 models trained and saved.
- [x] Log-normal + E5 models trained and saved.
- [x] All four saved bundles reloaded and verified.
- [x] Original calibration, validation, and test partitions unused.
- [ ] Verify exact E5 encoder reproduction from raw text.
- [ ] Test model loading in a fresh teammate environment.
- [ ] Add the final artifacts and documentation to the team repo.

## 13. Important Notes

The four regression families are reference predictors, not
automatically a complete, calibrated routing system.

Gamma and log-normal estimates are not directly equivalent:
one models mean cost, while the other predicts log-cost.

Regularization lambda was fixed at 0.1 for these reference
experiments; it was not selected using the original
validation or calibration partitions.

All training summaries report results from the internal
training-only split.
