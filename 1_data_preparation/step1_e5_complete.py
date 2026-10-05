# Run this cell in the routerbench_embeddings Jupyter environment.
# Starts from the RouterBench split tables you already saved.
# Existing cached embeddings assume those question tables are unchanged.

import json
import re
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler


# 1. Paths and settings
INPUT_DIR = Path(
    "/Users/shaoguanhua/Desktop/fall-2026-right-for-less"
    "1_data_preparation/data/processed/char_tfidf_50k"
)
OUTPUT_DIR = INPUT_DIR.parent / "multilingual_e5_small_chunk480"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 480
BATCH_SIZE = 8
RECOMPUTE_EMBEDDINGS = False
SPLITS = ("train", "calibration", "validation", "test")
COUNT_COLUMNS = ["answer_choice_count", "numeric_value_count"]


# 2. Load and check the existing split tables
tables = {}
for split in SPLITS:
    tables[split] = {
        name: pd.read_pickle(INPUT_DIR / f"{split}_{name}.pkl")
        for name in ("questions", "scores", "costs")
    }

previous_preprocessing = joblib.load(INPUT_DIR / "preprocessing.joblib")
model_names = previous_preprocessing["model_names"]
length_scaler = previous_preprocessing["length_scaler"]

for split in SPLITS:
    questions = tables[split]["questions"]
    scores = tables[split]["scores"]
    costs = tables[split]["costs"]
    assert len(questions) == len(scores) == len(costs)
    assert questions.index.equals(scores.index)
    assert questions.index.equals(costs.index)
    assert scores.columns.tolist() == costs.columns.tolist() == model_names
    assert questions["question_text"].notna().all()
    assert questions["question_text"].str.strip().ne("").all()
    assert np.isfinite(scores.to_numpy(dtype=np.float64)).all()
    assert np.isfinite(costs.to_numpy(dtype=np.float64)).all()
    assert scores.ge(0).all().all() and scores.le(1).all().all()
    assert costs.gt(0).all().all()

all_questions = pd.concat([
    tables[split]["questions"].assign(_split=split) for split in SPLITS
])
assert all_questions["sample_id"].notna().all()
assert not all_questions["sample_id"].duplicated().any()
assert all_questions.groupby("question_group")["_split"].nunique().max() == 1
assert all_questions.groupby("question_text")["_split"].nunique().max() == 1


# 3. E5 inputs: preserve short prompts; chunk long prompts without overlap
encoder = SentenceTransformer(MODEL_NAME, device="cpu")


def prepare_encoder_inputs(text):
    prefixed_text = "query: " + text
    full_ids = encoder.tokenizer(
        prefixed_text, add_special_tokens=True, truncation=False
    )["input_ids"]
    if len(full_ids) <= encoder.max_seq_length:
        return [prefixed_text]

    content_ids = encoder.tokenizer(
        text, add_special_tokens=False, truncation=False
    )["input_ids"]
    inputs = [
        "query: " + encoder.tokenizer.decode(content_ids[start:start + CHUNK_SIZE])
        for start in range(0, len(content_ids), CHUNK_SIZE)
    ]
    for chunk_input in inputs:
        ids = encoder.tokenizer(
            chunk_input, add_special_tokens=True, truncation=False
        )["input_ids"]
        assert len(ids) <= encoder.max_seq_length
    return inputs


def prepare_batch_inputs(question_texts):
    inputs, owners = [], []
    for position, text in enumerate(question_texts):
        prompt_inputs = prepare_encoder_inputs(text)
        inputs.extend(prompt_inputs)
        owners.extend([position] * len(prompt_inputs))
    return inputs, owners


def pool_question_embeddings(input_vectors, input_owners, n_questions):
    owners = np.asarray(input_owners, dtype=np.int64)
    counts = np.bincount(owners, minlength=n_questions)
    assert (counts > 0).all()
    sums = np.zeros((n_questions, input_vectors.shape[1]), dtype=np.float32)
    np.add.at(sums, owners, input_vectors)
    means = sums / counts[:, None].astype(np.float32)
    lengths = np.linalg.norm(means, axis=1, keepdims=True)
    assert np.isfinite(lengths).all() and (lengths > 0).all()
    return means / lengths


def embed_prompt(text):
    inputs = prepare_encoder_inputs(text)
    vectors = encoder.encode(
        inputs, normalize_embeddings=True, convert_to_numpy=True
    )
    return pool_question_embeddings(vectors, [0] * len(inputs), 1)[0]


# 4. Prompt-only handcrafted features; these are the rules we developed
line_choice_pattern = r"(?m)^[ \t]*([A-Z])\)[ \t]+"
numeric_pattern = r"\d+(?:,\d{3})*(?:\.\d+)?"
programming_request_pattern = (
    r"\b(?:write|implement|debug|fix|complete|create)\b"
    r"[^\n.!?]{0,100}"
    r"\b(?:function|code|program|script)\b"
)
code_syntax_pattern = (
    r"(?m)^[ \t]*(?:def[ \t]+\w+[ \t]*\("
    r"|import[ \t]+\w+|assert[ \t]+)"
)
chinese_programming_pattern = (
    r"(?:编写|实现|调试|修复|补全|写)"
    r"[^\n。！？]{0,60}"
    r"(?:函数|代码|程序|脚本)"
)
code_content_patterns = [r"```", code_syntax_pattern]
code_request_patterns = [programming_request_pattern, chinese_programming_pattern]


def count_answer_choices(text):
    labels = re.findall(line_choice_pattern, text)
    if len(labels) < 2:
        return 0
    if labels != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(labels)]):
        return 0
    return len(labels)


def count_numeric_values(text):
    return len(re.findall(numeric_pattern, text))


def detect_code(text):
    if any(re.search(p, text, flags=re.IGNORECASE) for p in code_content_patterns):
        return 1
    if count_answer_choices(text) > 0:
        return 0
    return int(any(
        re.search(p, text, flags=re.IGNORECASE) for p in code_request_patterns
    ))


def build_handcrafted_features(question_table):
    texts = question_table["question_text"]
    features = pd.DataFrame(index=question_table.index)
    features["char_length"] = texts.str.len()
    features["answer_choice_count"] = texts.apply(count_answer_choices)
    features["numeric_value_count"] = texts.apply(count_numeric_values)
    features["code_indicator"] = texts.apply(detect_code)
    return features


# 5. Extract features and fit the new count scaler on training only
handcrafted_features = {
    split: build_handcrafted_features(tables[split]["questions"])
    for split in SPLITS
}
count_scaler = StandardScaler()
count_scaler.fit(handcrafted_features["train"][COUNT_COLUMNS])


# 6. Load or create embeddings, assemble features, and save each split
e5_embeddings, full_features = {}, {}
for split in SPLITS:
    questions = tables[split]["questions"]
    raw_path = OUTPUT_DIR / f"X_{split}_e5.npy"
    if raw_path.exists() and not RECOMPUTE_EMBEDDINGS:
        embeddings = np.load(raw_path)
    else:
        inputs, owners = prepare_batch_inputs(questions["question_text"])
        input_vectors = encoder.encode(
            inputs,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        embeddings = pool_question_embeddings(input_vectors, owners, len(questions))
        np.save(raw_path, embeddings)

    assert embeddings.shape == (len(questions), 384)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-6)

    features = handcrafted_features[split]
    assert features.index.equals(questions.index)
    length = length_scaler.transform(features[["char_length"]]).astype(np.float32)
    counts = count_scaler.transform(features[COUNT_COLUMNS]).astype(np.float32)
    code = features[["code_indicator"]].to_numpy(dtype=np.float32)

    baseline = np.hstack([embeddings, length])
    full = np.hstack([baseline, counts, code])
    assert full.shape == (len(questions), 388)
    assert full.dtype == np.float32 and np.isfinite(full).all()

    np.save(OUTPUT_DIR / f"X_{split}_e5_features.npy", baseline)
    full_path = OUTPUT_DIR / f"X_{split}_e5_full_features.npy"
    np.save(full_path, full)
    assert np.array_equal(np.load(full_path), full)

    e5_embeddings[split] = embeddings
    full_features[split] = full
    print(split, "features:", full.shape, "scores:", tables[split]["scores"].shape,
          "costs:", tables[split]["costs"].shape)

joblib.dump(length_scaler, OUTPUT_DIR / "length_scaler.joblib")
joblib.dump(count_scaler, OUTPUT_DIR / "count_scaler.joblib")


# 7. Save the recipe and column layout (the Python helpers remain in this cell)
metadata = {
    "model_name": MODEL_NAME,
    "embedding_dimension": 384,
    "max_input_tokens": encoder.max_seq_length,
    "input_prefix": "query: ",
    "long_prompt_chunk_tokens": CHUNK_SIZE,
    "chunk_overlap_tokens": 0,
    "chunking_method": "split content token IDs, then decode each chunk",
    "pooling": "equal-weight mean of chunk vectors",
    "normalize_each_input": True,
    "normalize_after_pooling": True,
    "dtype": "float32",
    "question_tables_directory": str(INPUT_DIR.resolve()),
    "row_order": "same positional order as the corresponding question table",
    "model_names": model_names,
    "combined_feature_dimension": 385,
    "combined_feature_order": [
        "normalized_e5_embedding: columns 0–383",
        "standardized_char_length: column 384",
    ],
    "combined_feature_filename_pattern": "X_{split}_e5_features.npy",
    "full_feature_dimension": 388,
    "full_feature_filename_pattern": "X_{split}_e5_full_features.npy",
    "full_feature_order": [
        "normalized_e5_embedding: columns 0–383",
        "standardized_char_length: column 384",
        "standardized_answer_choice_count: column 385",
        "standardized_numeric_value_count: column 386",
        "binary_code_indicator: column 387",
    ],
    "length_scaler_file": "length_scaler.joblib",
    "count_scaler_file": "count_scaler.joblib",
    "count_scaler_columns": COUNT_COLUMNS,
    "answer_choice_pattern": line_choice_pattern,
    "answer_choice_validation": "at least two labels, consecutive from A, in order",
    "numeric_value_pattern": numeric_pattern,
    "code_content_patterns": code_content_patterns,
    "code_request_patterns": code_request_patterns,
    "code_regex_flags": "IGNORECASE",
    "code_detection_priority": [
        "code-content match -> 1",
        "otherwise accepted answer choices -> 0",
        "otherwise programming-request match -> 1",
        "otherwise -> 0",
    ],
    "package_versions": {
        name: version(name) for name in [
            "sentence-transformers", "transformers", "tokenizers", "torch",
            "numpy", "pandas", "pyarrow", "scikit-learn", "joblib",
        ]
    },
}
metadata_path = OUTPUT_DIR / "embedding_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

X_train_e5_full_features = full_features["train"]
X_calibration_e5_full_features = full_features["calibration"]
X_validation_e5_full_features = full_features["validation"]
X_test_e5_full_features = full_features["test"]

print("Saved outputs:", OUTPUT_DIR.resolve())
