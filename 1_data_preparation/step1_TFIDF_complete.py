# One complete notebook cell: RouterBench preparation and the saved TF-IDF baseline.
# Run in erdos_ds_environment, or another environment with the dependencies below.
# The raw download is reused if present; TF-IDF and scaling are always fitted anew.

import ast
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# 1. Locate or download the raw RouterBench dataset
PROJECT_DIR = Path("/Users/shaoguanhua/Desktop/fall-2026-right-for-less/1_data_preparation")
RAW_PATH = PROJECT_DIR / "data/raw/routerbench_0shot.pkl"
OUTPUT_DIR = PROJECT_DIR / "data/processed/char_tfidf_50k"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLITS = ("train", "calibration", "validation", "test")

if not RAW_PATH.exists():
    from huggingface_hub import hf_hub_download

    RAW_PATH = Path(hf_hub_download(
        repo_id="withmartian/routerbench",
        filename="routerbench_0shot.pkl",
        repo_type="dataset",
        local_dir=str(RAW_PATH.parent),
    ))

df = pd.read_pickle(RAW_PATH)


# 2. Extract aligned question, fractional-score, and cost tables
metadata_columns = [
    "sample_id", "prompt", "eval_name", "oracle_model_to_route_to",
]
model_names = [
    column for column in df.columns
    if column not in metadata_columns and "|" not in column
]
assert len(model_names) == 11

scores = df[model_names].copy()
costs = df[[name + "|total_cost" for name in model_names]].copy()
costs.columns = model_names
questions = df[["sample_id", "prompt", "eval_name"]].copy()

assert questions["sample_id"].notna().all()
assert not questions["sample_id"].duplicated().any()
assert questions["prompt"].notna().all()
assert np.isfinite(scores.to_numpy(dtype=np.float64)).all()
assert np.isfinite(costs.to_numpy(dtype=np.float64)).all()
assert scores.ge(0).all().all() and scores.le(1).all().all()
assert costs.gt(0).all().all()


def extract_question_text(prompt):
    try:
        parts = ast.literal_eval(prompt)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parts, list):
        return None
    if not all(isinstance(part, str) for part in parts):
        return None
    return "\n\n".join(parts).strip()


questions["question_text"] = questions["prompt"].apply(extract_question_text)
assert questions["question_text"].notna().all()
assert questions["question_text"].ne("").all()
assert questions.index.equals(scores.index)
assert questions.index.equals(costs.index)


# 3. Group exact duplicate texts and recreate the original 60/10/10/20 splits
# This grouping does not perform semantic near-duplicate detection.
questions["question_group"] = pd.factorize(
    questions["question_text"], sort=False
)[0]
group_ids = questions["question_group"].drop_duplicates()
shuffled_groups = group_ids.sample(frac=1, random_state=42).reset_index(drop=True)

n_groups = len(shuffled_groups)
train_end = int(0.60 * n_groups)
calibration_end = train_end + int(0.10 * n_groups)
validation_end = calibration_end + int(0.10 * n_groups)

split_groups = {
    "train": shuffled_groups.iloc[:train_end],
    "calibration": shuffled_groups.iloc[train_end:calibration_end],
    "validation": shuffled_groups.iloc[calibration_end:validation_end],
    "test": shuffled_groups.iloc[validation_end:],
}
questions["split"] = "unassigned"
for split in SPLITS:
    mask = questions["question_group"].isin(split_groups[split])
    questions.loc[mask, "split"] = split

assert not questions["split"].eq("unassigned").any()
assert questions.groupby("question_group")["split"].nunique().max() == 1

tables = {}
for split in SPLITS:
    mask = questions["split"].eq(split)
    tables[split] = {
        "questions": questions.loc[mask].copy(),
        "scores": scores.loc[mask].copy(),
        "costs": costs.loc[mask].copy(),
    }


# 4. Calculate full-prompt character lengths and fit the scaler on training only
length_features = {
    split: tables[split]["questions"]["question_text"].str.len().to_frame("char_length")
    for split in SPLITS
}
length_scaler = StandardScaler()
length_scaled = {
    "train": length_scaler.fit_transform(length_features["train"]),
}
for split in SPLITS[1:]:
    length_scaled[split] = length_scaler.transform(length_features[split])


# 5. Fit a fresh character TF-IDF vectorizer on training prompts only
tfidf_vectorizer_50k = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    lowercase=False,
    min_df=3,
    max_features=50000,
)
tfidf_matrices = {
    "train": tfidf_vectorizer_50k.fit_transform(
        tables["train"]["questions"]["question_text"]
    ),
}
for split in SPLITS[1:]:
    tfidf_matrices[split] = tfidf_vectorizer_50k.transform(
        tables[split]["questions"]["question_text"]
    )


# 6. Append scaled length while keeping the matrices sparse and float64
feature_matrices = {
    split: hstack(
        [tfidf_matrices[split], csr_matrix(length_scaled[split])], format="csr"
    )
    for split in SPLITS
}


def check_alignment(matrix, question_table, score_table, cost_table):
    assert matrix.shape[0] == len(question_table) == len(score_table) == len(cost_table)
    assert question_table.index.equals(score_table.index)
    assert question_table.index.equals(cost_table.index)
    assert score_table.columns.tolist() == cost_table.columns.tolist() == model_names


# 7. Check and save the matrices plus the original split tables
for split in SPLITS:
    matrix = feature_matrices[split]
    split_tables = tables[split]
    check_alignment(
        matrix, split_tables["questions"], split_tables["scores"], split_tables["costs"]
    )
    assert matrix.shape[1] == tfidf_matrices["train"].shape[1] + 1
    assert matrix.dtype == np.float64
    assert np.isfinite(matrix.data).all()

    matrix_path = OUTPUT_DIR / f"X_{split}.npz"
    save_npz(matrix_path, matrix)
    assert (load_npz(matrix_path) - matrix).nnz == 0

    for name, table in split_tables.items():
        table_path = OUTPUT_DIR / f"{split}_{name}.pkl"
        table.to_pickle(table_path)
        assert pd.read_pickle(table_path).equals(table)

    active_counts = tfidf_matrices[split].getnnz(axis=1)
    print(
        split,
        "features:", matrix.shape,
        "scores:", split_tables["scores"].shape,
        "costs:", split_tables["costs"].shape,
        "zero TF-IDF rows:", int((active_counts == 0).sum()),
    )


# 8. Save the fitted vectorizer, scaler, model order, and feature order
preprocessing = {
    "tfidf_vectorizer": tfidf_vectorizer_50k,
    "length_scaler": length_scaler,
    "model_names": model_names,
    "feature_order": ["tfidf", "scaled_char_length"],
}
preprocessing_path = OUTPUT_DIR / "preprocessing.joblib"
joblib.dump(preprocessing, preprocessing_path)

restored_preprocessing = joblib.load(preprocessing_path)
assert restored_preprocessing["model_names"] == model_names
assert restored_preprocessing["tfidf_vectorizer"].vocabulary_ == tfidf_vectorizer_50k.vocabulary_
assert np.array_equal(restored_preprocessing["tfidf_vectorizer"].idf_, tfidf_vectorizer_50k.idf_)
assert np.array_equal(restored_preprocessing["length_scaler"].mean_, length_scaler.mean_)
assert np.array_equal(restored_preprocessing["length_scaler"].scale_, length_scaler.scale_)


# Useful names for the next notebook steps
X_train = feature_matrices["train"]
X_calibration = feature_matrices["calibration"]
X_validation = feature_matrices["validation"]
X_test = feature_matrices["test"]

train_questions, train_scores, train_costs = (
    tables["train"][name] for name in ("questions", "scores", "costs")
)
calibration_questions, calibration_scores, calibration_costs = (
    tables["calibration"][name] for name in ("questions", "scores", "costs")
)
validation_questions, validation_scores, validation_costs = (
    tables["validation"][name] for name in ("questions", "scores", "costs")
)
test_questions, test_scores, test_costs = (
    tables["test"][name] for name in ("questions", "scores", "costs")
)

print("Saved outputs:", OUTPUT_DIR.resolve())
