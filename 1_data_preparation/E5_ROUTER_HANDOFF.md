# RouterBench E5 notebook: teammate handoff

This guide describes the E5 preparation workflow we completed and how to use its outputs for the remaining routing steps. **Start modeling with `X_{split}_e5_full_features.npy`: each question has 388 features.** Benchmark scores remain fractional values in `[0, 1]`.

Suggested location when sharing: `Erdos-Router/1_data_preparation/E5_ROUTER_HANDOFF.md`. Notebook links below are relative to that location. The loading cell assumes the working directory is the repository root; change `ROOT` if necessary.

## 1. Notebooks and scope

| File | Purpose |
| --- | --- |
| [e5_emedding.ipynb](e5_emedding.ipynb) | Original notebook developed incrementally, with examples and checks. The filename has this spelling. |
| [step1_e5_complete.ipynb](step1_e5_complete.ipynb) | Consolidated preparation workflow in one cell. |
| [step1_e5_complete.py](step1_e5_complete.py) | The same consolidated workflow as a Python script. |

The E5 workflow starts from the question, score, and cost tables prepared during the TF-IDF stage. It does **not** download RouterBench again or create new splits. Those tables are shared inputs for both representations.

The saved `question_text` was extracted from the original stringified list of prompt parts using `ast.literal_eval`, joining the string parts with `"\n\n"`, then stripping outer whitespace. Keep this text unchanged when reusing cached embeddings. Do not replace it with a generated model response.

Completed: prompt extraction, grouping exact duplicate texts, splitting, E5 encoding, four handcrafted features, training-only scaling, saving, and save/reload checks. Remaining: score models, calibration, cost models, policy selection, and final evaluation. No trained router is included in these preparation outputs.

## 2. Files to share

Keep these two directories together under `1_data_preparation/data/processed/`:

```text
char_tfidf_50k/
    {split}_questions.pkl
    {split}_scores.pkl
    {split}_costs.pkl
    preprocessing.joblib

multilingual_e5_small_chunk480/
    X_{split}_e5.npy
    X_{split}_e5_features.npy
    X_{split}_e5_full_features.npy
    length_scaler.joblib
    count_scaler.joblib
    embedding_metadata.json
```

Here `{split}` means `train`, `calibration`, `validation`, or `test`; each pattern represents four files. Share the source notebook/script too. TF-IDF `.npz` matrices are only needed if teammates also want that baseline.

`preprocessing.joblib` supplies the original model order and training-fitted length scaler. The E5 directory contains a copy of that scaler and the new count scaler. The JSON records the embedding recipe and feature layout; it does not contain encoder weights or feature-extraction functions. Absolute paths inside metadata refer to the original computer.

| Split | Question groups | Rows | Full features | Scores and costs, each |
| --- | ---: | ---: | --- | --- |
| Train | 21,888 | 21,900 | `(21900, 388)` | `(21900, 11)` |
| Calibration | 3,648 | 3,648 | `(3648, 388)` | `(3648, 11)` |
| Validation | 3,648 | 3,649 | `(3649, 388)` | `(3649, 11)` |
| Test | 7,297 | 7,300 | `(7300, 388)` | `(7300, 11)` |

These are the recorded notebook results, not a fresh dataset regeneration for this guide. There are 36,497 rows and 36,481 exact-text groups. Sixteen pairs share a text; members of a group stay in one split. The split used approximately 60%/10%/10%/20% of groups and random seed 42. A comprehensive near-duplicate audit remains separate work.

## 3. Environment and running preparation

The original `erdos_ds_environment` crashed during PyTorch import because two OpenMP runtimes were loaded. A clean `routerbench_embeddings` environment resolved that problem. A separate environment is a dependency choice, not a mathematical requirement.

For teammates who need to run preparation and have Conda:

```bash
conda create --name routerbench_embeddings python=3.12 pip --no-default-packages
conda activate routerbench_embeddings
python -m pip install sentence-transformers pandas numpy scipy scikit-learn joblib pyarrow ipykernel
python -m ipykernel install --user --name routerbench_embeddings --display-name "RouterBench embeddings"
```

Select that kernel in VS Code and check `sys.executable`. `pyarrow` is needed to load these saved pandas tables. Do not bypass an OpenMP conflict with `KMP_DUPLICATE_LIB_OK`; use a consistent environment. Preserve the recorded package versions/environment when reproducing a run rather than assuming a fresh installation reproduces every bit.

Edit `INPUT_DIR` in the consolidated notebook/script to point to your copy of `char_tfidf_50k`. `OUTPUT_DIR` defaults to its sibling E5 directory. The supplied script contains the author's absolute path.

| Setting | Actual behavior |
| --- | --- |
| `RECOMPUTE_EMBEDDINGS = False` | Load existing raw `X_{split}_e5.npy`; encode a split if that file is missing. |
| `RECOMPUTE_EMBEDDINGS = True` | Encode all four splits again and save raw embeddings. |
| Either setting | Recreate handcrafted features, fit the training count scaler, rebuild the 385- and 388-column matrices, and save scalers and metadata. |

The script initializes the encoder even with cached vectors, so it may still need access to encoder weights. Teammates who only load prepared matrices for modeling do not need to initialize the encoder.

**Cache requirement:** question texts and positional order must be unchanged. Shape alone cannot prove this. If prompts, splits, encoder revision, chunking, or pooling change, regenerate in a new versioned directory and keep the matching tables. Current raw `.npy` files have no embedded sample-ID manifest. Preserve original metadata with cached vectors: rerunning the script records the current environment even if the vectors came from an earlier run.

## 4. What one feature vector contains

The encoder is `intfloat/multilingual-e5-small`, with 384 dimensions and a 512-token input limit. Every input receives `query: `, including non-English prompts. This prefix is also recommended for general embedding features. See the [official E5 model card](https://huggingface.co/intfloat/multilingual-e5-small).

Our long-prompt recipe is a project choice:

1. Encode the original prefixed prompt directly if it fits within 512 tokens, including special tokens.
2. Otherwise, tokenize the original text without prefix or special tokens and divide it into non-overlapping chunks of 480 content tokens.
3. Decode each chunk, add `query: `, and verify the resulting input fits the encoder limit.
4. Encode and normalize each input vector. Average a question's chunk vectors with **equal weight**, then normalize the average.

This preserves the tail of long prompts, where questions and options may appear. It is not token-weighted pooling. Chunking can separate related information and decoding can change whitespace; keeping every chunk does not guarantee perfect semantic representation. The training notebook found 120 prompts over the limit, about 0.55% of training rows.

| Zero-based column | Feature | Treatment |
| --- | --- | --- |
| 0–383 | E5 question vector | Unit-length, pooled if needed |
| 384 | Full prompt character length | Standardized with training length scaler |
| 385 | Answer-choice count | Standardized with training count scaler |
| 386 | Numeric-value count | Standardized with training count scaler |
| 387 | Code indicator | Kept as 0 or 1 |

Saved E5 matrices use `float32`. Only the 384-dimensional embedding block is unit-length. **Do not normalize the complete 388-column matrix.** The 384-column file is embeddings alone; the 385-column file adds length; the 388-column file includes all four handcrafted features.

The final feature rules must remain consistent between training and inference:

- **Character length:** Python string length of the complete `question_text`, including whitespace.
- **Answer-choice count:** match `(?m)^[ \t]*([A-Z])\)[ \t]+`, then accept only at least two labels consecutive from `A`, in order. Zero means no accepted sequence was detected; other formats may be missed. Count actual options, not letters mentioned in answer instructions.
- **Numeric-value count:** count matches of `\d+(?:,\d{3})*(?:\.\d+)?`. This counts occurrences of digit-based values, including some digits in dates, identifiers, and code. It is not a numerical-expression parser; scientific notation and spelled-out numbers are not handled as complete values.
- **Code indicator:** first flag fenced text or selected line-start Python syntax. Otherwise, return zero if accepted answer choices exist. Otherwise, flag the English/Chinese programming-request patterns. Exact patterns are in the notebook/script and metadata. This heuristic can miss code or flag non-code fenced text.

Recorded responses, scores, observed costs, oracle choices, and task identity are not input features. `eval_name` is retained for diagnostics and reporting.

## 5. Load the prepared data for modeling

Run this cell from the repository root or change `ROOT`. It loads development splits. Call `load_split("test")` after model and policy choices are frozen.

```python
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path.cwd()  # Change to your Erdos-Router root if necessary.
PROCESSED = ROOT / "1_data_preparation" / "data" / "processed"
TABLES_DIR = PROCESSED / "char_tfidf_50k"
E5_DIR = PROCESSED / "multilingual_e5_small_chunk480"

preprocessing = joblib.load(TABLES_DIR / "preprocessing.joblib")
model_names = preprocessing["model_names"]
metadata = json.loads((E5_DIR / "embedding_metadata.json").read_text())
assert metadata["full_feature_dimension"] == 388
if "model_names" in metadata:
    assert metadata["model_names"] == model_names

def load_split(split):
    assert split in {"train", "calibration", "validation", "test"}
    X = np.load(E5_DIR / f"X_{split}_e5_full_features.npy")
    questions = pd.read_pickle(TABLES_DIR / f"{split}_questions.pkl")
    scores = pd.read_pickle(TABLES_DIR / f"{split}_scores.pkl")
    costs = pd.read_pickle(TABLES_DIR / f"{split}_costs.pkl")

    assert X.shape == (len(questions), 388)
    assert X.dtype == np.float32 and np.isfinite(X).all()
    assert questions.index.equals(scores.index)
    assert questions.index.equals(costs.index)
    assert scores.shape == costs.shape == (len(questions), len(model_names))
    assert scores.columns.tolist() == costs.columns.tolist() == model_names
    assert questions["split"].eq(split).all()
    y = scores.to_numpy(dtype=np.float64)
    c = costs.to_numpy(dtype=np.float64)
    assert np.isfinite(y).all() and ((0 <= y) & (y <= 1)).all()
    assert np.isfinite(c).all() and (c > 0).all()
    assert np.allclose(np.linalg.norm(X[:, :384], axis=1), 1, atol=1e-6)
    assert np.isin(X[:, 387], [0, 1]).all()
    return {"X": X, "questions": questions, "scores": scores, "costs": costs}

data = {split: load_split(split)
        for split in ("train", "calibration", "validation")}

development_questions = pd.concat(
    [part["questions"] for part in data.values()]
)
assert development_questions["sample_id"].is_unique
assert development_questions.groupby("question_group")["split"].nunique().max() == 1
assert development_questions.groupby("question_text")["split"].nunique().max() == 1
print({split: part["X"].shape for split, part in data.items()})
```

**Row contract:** matrix row `i` corresponds to `questions.iloc[i]`, `scores.iloc[i]`, and `costs.iloc[i]`. Original pandas indices have gaps and are not matrix row numbers. Do not independently sort, filter, shuffle, or reset tables. Apply row selections to all four objects in the same positional order. These checks verify table alignment and feature structure; they cannot detect a previously shuffled cached matrix without a saved row manifest.

## 6. Candidate models and targets

The tables contain **11 candidates**: six open models and five proprietary models. The authoritative order comes from `preprocessing["model_names"]`:

```text
WizardLM/WizardLM-13B-V1.2
claude-instant-v1
claude-v1
claude-v2
gpt-3.5-turbo-1106
gpt-4-1106-preview
meta/code-llama-instruct-34b-chat
meta/llama-2-70b-chat
mistralai/mistral-7b-chat
mistralai/mixtral-8x7b-chat
zero-one-ai/Yi-34B-Chat
```

The repository's six-open-model experiment uses WizardLM, Code Llama, Llama 2, Mistral, Mixtral, and Yi. If the team chooses that scope, select those same columns in the same order from scores and costs, and use that order for prediction matrices. Question features stay unchanged. Decide the pool before policy tuning.

For question `i` and model `m`, `y[i, m]` is the recorded score in `[0, 1]`. We explicitly chose to keep fractional scores. Predict **expected benchmark score**, `q_hat[i, m]`, rather than claim a probability of binary correctness. Do not threshold targets to make a classifier accept them. A row-average score weights tasks according to their row counts; report per-task results because benchmarks use different scoring rubrics.

`costs[i, m]` is the recorded historical dollar cost of that response. It is a training/evaluation target, not a prompt feature or a statement about today's provider prices.

## 7. Continue with Steps 2–7

### Step 2: predict each candidate's score

Fit one score model per candidate on training questions only. A sigmoid linear model can use fractional cross-entropy:

```text
q_hat = sigmoid(w · x + b)
loss = mean(-y log(q_hat) - (1-y) log(1-q_hat)) + L2 penalty on w
```

This is a surrogate for bounded expected score. It does not turn fractional observations into binary outcomes. Do not directly call `LogisticRegression.fit(X, fractional_y)`: that estimator expects class labels. Implement the fractional objective explicitly, or use weighted binary copies: each original training row contributes a label-1 copy weighted by `y` and a label-0 copy weighted by `1-y`. Their combined loss is exactly the fractional cross-entropy term. [LogisticRegression supports sample weights](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

Select regularization with grouped cross-validation **within training**, using `question_group`. Split original questions before creating weighted copies. For strict fold isolation, fit length/count scalers on each fold's training rows and transform held-out rows. Saved 388-column matrices use full-training scalers, appropriate for the final training fit. The pretrained E5 block can be reused across folds.

Compare prediction error with a constant training-mean-score baseline. Inspect convergence and per-task errors. On calibration data, compare binned predicted scores with average observed scores. If needed, fit a calibration mapping that accepts continuous targets, such as isotonic regression. Binary classifier calibration utilities should not receive fractional labels directly. Freeze score models and calibration before policy tuning.

### Step 3: predict each candidate's cost

Fit one positive conditional-mean cost model per candidate on training data. Gamma regression with a log link is the intended baseline; tune regularization using training grouped CV. [GammaRegressor uses a log link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html).

Compare cost MAE and mean predicted versus observed cost against a constant training-mean-cost baseline. Exponentiating ordinary least squares predictions for log-cost does not automatically estimate mean dollar cost. Prepared costs are strictly positive, suitable for the Gamma baseline.

All 11 candidates require 11 score models and 11 cost models; the six-model pool requires six of each.

### Step 4: define the quality requirement

Before examining test results, agree on the pool, aggregation rule, fixed reference model, and either an absolute mean-score target `A` or an allowed mean-score loss `delta` relative to the reference. Under our fractional-score decision, this is a **quality requirement**, not a binary accuracy guarantee.

### Steps 5–6: apply the routing rule

Predict every candidate's quality and cost for each prompt, then choose:

```text
selected_model = argmin_m [c_hat_m + lambda × (1 - q_hat_m)]
```

Here `lambda >= 0` is dollars per unit of expected score loss. There is no additional routing classifier to train for this rule. Define deterministic ties, such as choosing the first model in the fixed order. At decision time, use predictions only; observed scores/costs evaluate the chosen candidate afterward.

For two candidates, the higher-quality option is preferred when its additional predicted cost is smaller than `lambda` times its additional predicted quality. This is the Step 6 upgrade interpretation.

### Step 7: choose the policy, freeze it, then test once

Evaluate a predeclared lambda grid on validation questions using the selected model's **observed** score and cost. Choose the cheapest policy meeting the agreed quality requirement. If none meets it, report that outcome and revisit development choices without using test data. Increasing lambda need not monotonically increase observed quality.

Freeze the encoder recipe, features, scalers, candidate order, score/cost models, calibration, reference, lambda, and tie rule. Then load test data and report:

- Mean observed routed score and its difference from the fixed reference.
- Mean observed cost, cost per 1,000 queries, and savings relative to the reference.
- Selection frequencies and per-`eval_name` results.
- Fixed-model baselines and the TF-IDF baseline if included.
- Paired uncertainty estimates resampling `question_group`, keeping each question's candidate outcomes together. A non-significant difference alone does not establish an allowed-loss requirement.

Benchmark dollar costs exclude our encoder's serving overhead. Measure that latency/cost separately for end-to-end deployment reporting. Keep one-time preparation expense separate from per-query inference expense.

## 8. Features for a new prompt

Use the exact `prepare_encoder_inputs`, `pool_question_embeddings`, `embed_prompt`, and handcrafted-feature helpers from the consolidated source. Initialize the same encoder once, load both scalers once, and concatenate:

```text
384-dimensional normalized E5 vector
+ training-scaled full character length
+ training-scaled answer-choice count
+ training-scaled numeric-value count
+ unscaled binary code indicator
= one float32 vector of length 388
```

Do not refit scalers on incoming prompts. Preserve prompt formatting and the same preprocessing contract. For deployment, move helpers into an importable module: importing the current complete script executes preparation and writes outputs, so it is not yet a library module.

## 9. Handoff checklist

1. Share both processed-data folders, source notebook/script, this guide, and the original environment/version record.
2. Run the loading cell and verify row counts, model order, and local paths.
3. Agree on 11 versus six candidates, fractional-score metrics, the quality target, and reference.
4. Develop score/cost models with grouped training CV; reserve calibration for score calibration and validation for policy selection.
5. Preserve provenance. Current preparation does not pin an encoder/dataset commit or attach row-ID hashes to cached matrices; record these for subsequent versions. Complete a near-duplicate audit if stronger separation is required.
6. Save frozen models and policy with their feature recipe, then perform final test evaluation.

The [repository README](../README.md) and [KPI notes](../kpis.md) provide project context. Where they describe binary labels or accuracy, apply the fractional-score decision here explicitly; do not silently change the saved targets.
