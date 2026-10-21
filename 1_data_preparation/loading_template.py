from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "1_data_preparation").is_dir():
    ROOT = ROOT.parent

processed_dir = ROOT / "1_data_preparation/data/processed"
assert processed_dir.is_dir(), f"Data folder not found: {processed_dir}"

processed_dir = ROOT / "1_data_preparation/data/processed"
tables_dir = processed_dir / "char_tfidf_50k"
embedding_dir = processed_dir / "multilingual_e5_small_chunk480"

preprocessing = joblib.load(tables_dir / "preprocessing.joblib")
model_names = preprocessing["model_names"]

def load_split(split):
    X = np.load(
        embedding_dir / f"X_{split}_e5_full_features.npy"
    )

    questions = pd.read_pickle(tables_dir / f"{split}_questions.pkl")
    scores = pd.read_pickle(
    tables_dir / f"{split}_scores.pkl").astype(np.float64)
    costs = pd.read_pickle(tables_dir / f"{split}_costs.pkl")

    assert X.shape == (len(questions), 388)
    assert scores.shape == costs.shape == (
        len(questions), len(model_names)
    )
    assert questions.index.equals(scores.index)
    assert questions.index.equals(costs.index)
    assert scores.columns.tolist() == costs.columns.tolist() == model_names
    assert questions["split"].eq(split).all()

    assert np.isfinite(X).all()
    assert np.isfinite(scores.to_numpy()).all()
    assert np.isfinite(costs.to_numpy()).all()
    assert scores.ge(0).all().all() and scores.le(1).all().all()
    assert costs.gt(0).all().all()

    print(split, "Features:", X.shape, "Scores:", scores.shape)
    return X, questions, scores, costs

if __name__ == "__main__":
    X_train, train_questions, train_scores, train_costs = load_split("train")
    
    X_calibration, calibration_questions, calibration_scores, calibration_costs = (
        load_split("calibration")
    )
    
    X_validation, validation_questions, validation_scores, validation_costs = (
        load_split("validation")
    )