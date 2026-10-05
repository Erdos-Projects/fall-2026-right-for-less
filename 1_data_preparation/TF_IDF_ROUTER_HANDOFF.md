# Using the TF-IDF Step 1 data to build the router

This guide is for teammates continuing the experiment in [TF-IDF embedding.ipynb](<TF-IDF embedding.ipynb>). It explains what the notebook prepared, how to load its saved data, and how those data support Steps 2 through 7 of the supplied *The Right Answer, for Less* mathematical guide. The notebook completed data preparation and feature construction; it did not train correctness or cost regressions, select a routing policy, or measure router savings.

The saved feature matrices can be used directly for fitting the final regressions after choosing their settings. Three choices need to be settled first: which candidate models to include, how to turn fractional benchmark scores into a correctness target, and whether to retain the saved character features or rebuild them using the later unigram experiment. These are proposed next decisions, not decisions already implemented by the notebook.

## Step 1 What the notebook did

### Loaded and checked RouterBench

The notebook downloaded `routerbench_0shot.pkl` from `withmartian/routerbench` on Hugging Face. The downloaded table has **36,497 rows and 37 columns**, including question identifiers, prompts, evaluation names, model scores, generated responses, historical response costs, and an oracle routing field.

It separated the source into three aligned tables:

| Table | Contents | Later use |
| --- | --- | --- |
| `questions` | `sample_id`, original `prompt`, `eval_name`, then parsed text, group, and split | Features, grouping, audits, and subgroup reporting |
| `scores` | One column per candidate model | Targets for correctness after defining success; observed policy evaluation |
| `costs` | One column per candidate model, renamed from `<model>\|total_cost` | Targets for cost regression; observed policy evaluation |

The recorded notebook checks found no missing or repeated sample IDs, no missing prompts, no missing model scores or costs, no scores outside [0, 1], and no zero or negative costs. Reading back all four saved partitions also confirmed finite feature values, scores, and costs, and consistent table indexes. These checks establish usability of the saved records; they do not establish binary score semantics or independent questions.

Generated responses, the oracle model, observed scores, and observed costs do not enter the feature matrix. The router must make its decision using information available before an LLM answers.

### Included 11 historical candidate models

The notebook selected every nonmetadata column without `|`, so its saved score and cost tables contain the following models, in this order:

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

The [repository README](../README.md) describes a pool of **six open-weight candidates**. To follow that project scope, select those six columns from every score and cost table. The feature matrices are shared across models and need no column changes. If the team instead chooses all 11, document the broader experiment explicitly. Use model identifiers as recorded benchmark configurations; the saved costs describe historical responses.

### Converted serialized prompts into text

The source `prompt` is a string containing a Python list of strings. The notebook used `ast.literal_eval`, required a list containing only strings, joined its parts with two newlines, and stripped leading and trailing whitespace. All rows parsed successfully and produced nonempty text.

```python
import ast

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
```

Use the saved `question_text` for downstream work. For future benchmark records in the same serialized format, apply this function. A new request that is already plain text should be passed as plain text; parsing it as a serialized list would reject it.

### Grouped exact duplicates and created four partitions

There were **36,481 distinct question texts**. Sixteen exact duplicate groups each contained two rows, accounting for 32 rows. The notebook created `question_group` by factorizing exact `question_text`, shuffled the unique groups with `random_state=42`, and allocated approximately 60%, 10%, 10%, and 20% to the four partitions. It retained original source row order inside each partition.

| Partition | Question groups | Saved rows | Purpose |
| --- | ---: | ---: | --- |
| Training | 21,888 | 21,900 | Choose regression settings within training and fit the predictors |
| Calibration | 3,648 | 3,648 | Check and optionally adjust correctness probabilities |
| Policy validation | 3,648 | 3,649 | Select the reference model and routing tradeoff |
| Final test | 7,297 | 7,300 | Evaluate the frozen experiment |

No exact-text group crosses partitions. Rows and groups differ because duplicate rows were retained. In formulas that assume independent questions, use group counts to describe independence and state whether metrics weight source rows or unique question groups. The examples below use the notebook's source rows. For uncertainty estimates, resample complete question groups.

The notebook also checked whether whitespace normalization reduced the number of distinct texts; it did not. That check did not change `question_group`. Near-duplicates and alternative prompting versions were **not comprehensively grouped**. The PDF calls for keeping them together, so a broader duplicate audit remains a prerequisite for a strong generalization claim. If that audit changes group assignments, rebuild the partitions and every fitted preprocessing artifact before modeling.

### Built character TF-IDF features and standardized length

TF-IDF gives more weight to character sequences that are frequent in a prompt but less common across the training corpus. Here it is a sparse text representation, rather than a sentence encoder. Character sequences let the same procedure represent English, Chinese, and other text without relying on word boundaries.

The first trial used character sequences of length 2 through 4 and a maximum of 20,000 features. It left 206 training prompts with fewer than ten active features. The notebook expanded the vocabulary to 50,000 features and used this version for the saved matrices:

```python
TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    lowercase=False,
    min_df=3,
    max_features=50000,
)
```

`min_df=3` retains sequences appearing in at least three training documents; `max_features` caps the vocabulary. The saved vectorizer uses the default smoothed inverse document frequency and L2 normalization. Its vocabulary and IDF weights were fitted on training text only; the other partitions used `transform`. See the [official TF-IDF parameter documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

The notebook measured `char_length = len(question_text)` and fitted `StandardScaler` on training lengths only. The saved training mean is approximately 761.33 characters and its scale is approximately 429.62 characters. This is character count, not model-specific token count.

The final feature order is:

```text
50,000 TF-IDF columns | 1 standardized character-length column
```

Thus each question has **50,001 features**, stored as a SciPy CSR sparse matrix. The completed matrix is not itself L2 normalized: the standardized length is appended after TF-IDF normalization. Keep these large matrices sparse.

The PDF proposes a sentence embedding plus length, answer-choice count, numerical-expression count, and a code indicator. This notebook implements **character TF-IDF plus character length** instead. The later mathematical steps still accept this feature map, but describe the experiment using its actual features.

### Investigated a feature limitation after saving

All four saved TF-IDF partitions had no zero-feature prompts and no prompts with fewer than ten active features. However, a nearest-neighbor diagnostic found different Chinese riddles with TF-IDF cosine similarity 1.0. For one inspected pair, none of the 132 distinguishing character sequences survived in the saved vocabulary. Identical vectors therefore do not establish identical questions.

A later trial changed `ngram_range` to `(1, 4)`, retaining some distinguishing individual characters. The inspected pair's similarity fell to approximately **0.97093**. This supports investigating better coverage; it is not evidence of improved router accuracy.

**The unigram trial was not saved.** `preprocessing.joblib` and every `X_*.npz` file still use the `(2, 4)` vectorizer. To adopt `(1, 4)`, transform all four partitions with the newly fitted training vectorizer, append the same length feature, save the complete artifact set in a separate named directory, and repeat alignment and reconstruction checks. Matching matrix dimensions alone do not make two vocabularies interchangeable.

## Load the saved Step 1 artifacts

The current files are under:

```text
1_data_preparation/data/processed/char_tfidf_50k/
```

| File pattern | Contents |
| --- | --- |
| `X_<split>.npz` | CSR matrix with 50,001 feature columns |
| `<split>_questions.pkl` | Six columns: `sample_id`, `prompt`, `eval_name`, `question_text`, `question_group`, `split` |
| `<split>_scores.pkl` | Raw benchmark scores, with 11 model columns |
| `<split>_costs.pkl` | Historical response costs, with the same model columns |
| `preprocessing.joblib` | Fitted vectorizer, fitted length scaler, saved model order, and feature order |

There are 17 artifacts: four matrices, twelve tables, and one preprocessing bundle. The notebook reloaded training data, confirmed equality to the in-memory objects, and reconstructed one feature row with a maximum absolute difference of about `5.55e-17`.

Use `numpy`, `pandas`, `scipy`, `scikit-learn`, and `joblib`, listed in [requirements.txt](../requirements.txt). Reusing these files does not require downloading RouterBench. Rerunning the download cells also requires `huggingface_hub`, which the notebook installs separately. The README's `prepare_step1.py` and `config.json` entry point is absent in this checkout; use this notebook or its saved files.

The following blocks form a sequence of examples for a downstream notebook. Run from the repository root, or change `ROOT` to its absolute path. They load trusted project artifacts, preserve row alignment, and select the six candidates specified in the README.

```python
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, hstack, csr_matrix

ROOT = Path.cwd()  # Repository root; change if running elsewhere.
DATA = ROOT / "1_data_preparation/data/processed/char_tfidf_50k"
preprocessing = joblib.load(DATA / "preprocessing.joblib")
saved_models = preprocessing["model_names"]
models = [
    "WizardLM/WizardLM-13B-V1.2",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]
# To run the broader experiment instead: models = list(saved_models)

def load_split(split):
    X = load_npz(DATA / f"X_{split}.npz").tocsr()
    questions = pd.read_pickle(DATA / f"{split}_questions.pkl")
    scores = pd.read_pickle(DATA / f"{split}_scores.pkl")
    costs = pd.read_pickle(DATA / f"{split}_costs.pkl")
    assert len(questions) == len(scores) == len(costs) == X.shape[0]
    assert questions.index.equals(scores.index)
    assert questions.index.equals(costs.index)
    assert scores.columns.tolist() == costs.columns.tolist() == saved_models
    assert X.shape[1] == len(preprocessing["tfidf_vectorizer"].vocabulary_) + 1
    assert preprocessing["feature_order"] == ["tfidf", "scaled_char_length"]
    assert questions["split"].eq(split).all()
    assert np.isfinite(X.data).all()
    scores = scores.loc[:, models].astype(float)
    costs = costs.loc[:, models].astype(float)
    assert np.isfinite(scores.to_numpy()).all()
    assert np.isfinite(costs.to_numpy()).all()
    assert ((scores >= 0) & (scores <= 1)).all().all()
    assert (costs > 0).all().all()
    return X, questions, scores, costs

X_train, q_train, score_train, cost_train = load_split("train")
X_cal, q_cal, score_cal, cost_cal = load_split("calibration")
X_val, q_val, score_val, cost_val = load_split("validation")
# Load the test partition only after all experiment choices are frozen.
```

The matrices have positional rows but no pandas index. Matrix row `r` corresponds to `questions.iloc[r]`, `scores.iloc[r]`, and `costs.iloc[r]`. The preserved pandas index is the original source row label and is usually nonconsecutive. Do not sort, shuffle, filter, or drop rows from only one object. When selecting model columns, use the same `models` list everywhere.

## Step 2 Predict correctness for each candidate

### Define a binary target before fitting

The PDF's logistic regression assumes `y` is either 0 or 1. The saved tables contain scores such as 0.25, 0.5, and 0.75; the training table contains **51,159 fractional score entries across all 11 models**. These are benchmark scores, not automatically probabilities or binary correctness labels.

A proposed implementation of the PDF's binary experiment is **full-credit success**:

```python
Y_train = score_train.eq(1.0).astype("int8")
Y_cal = score_cal.eq(1.0).astype("int8")
Y_val = score_val.eq(1.0).astype("int8")
```

This defines success as a recorded score exactly equal to 1.0 and treats partial credit as failure. Document that definition and verify its suitability for the evaluation tasks before adoption. The notebook itself did not binarize scores. Use the same definition for training, calibration, policy selection, reference comparisons, and testing. An average raw benchmark score should be reported as mean score, not binary accuracy. Retaining partial credit would require an explicitly different target and prediction model.

### Fit and tune one regression per model

For each model `m`, fit `p_m(x) = sigmoid(a_m + beta_m @ phi(x))` using `X_train` and `Y_train[m]`. Retain continuous success probabilities for routing.

Choose regularization by grouped cross-validation **inside training only**, with `q_train["question_group"]` as the grouping variable. Refit the vectorizer and length scaler on each inner training fold, then transform its held-out fold. The saved `X_train` preprocessing was fitted on all outer training rows, so using it unchanged in inner CV would expose that fold's text distribution to vocabulary, IDF, and scaling. After choosing settings, fit the final regressions on all saved training features.

Check that each training fold contains both outcome classes. Score candidates using binary log loss; avoid automatic class weighting when the goal is an unadjusted probability of success. The [LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) describes sparse input support and the inverse regularization parameter `C`. Record the software settings rather than equating `C` directly to the PDF's penalty coefficient.

After training, predict on `X_cal`, report log loss and Brier score, and inspect reliability curves. If calibration is needed, fit a held-out sigmoid or isotonic calibrator on that partition while keeping the base regression frozen. Store the base model and calibrator together. Do not fit calibration on policy-validation or test outcomes.

**Deliverable:** one final probability predictor per candidate, plus calibration diagnostics and any fitted calibrators.

## Step 3 Predict historical cost for each candidate

Fit `c_m(x) = exp(b_m + gamma_m @ phi(x))` using `X_train` and the strictly positive `cost_train[m]`. These targets are recorded dollars per historical response, rather than token counts or present-day prices.

Use grouped training-only cross-validation to choose the cost regularization, with the same fold-local preprocessing discipline as Step 2. The [GammaRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html) specifies the log link and penalty parameter `alpha`. Use its `lbfgs` solver as an initial choice for this high-dimensional representation; its Newton solver explicitly constructs a Hessian with quadratic memory growth in the feature count. Check convergence and finite positive predictions.

Compare held-out dollar MAE, Gamma loss, and mean predicted versus observed cost against a simple baseline: always predict `cost_train[m].mean()`. Reserve the final test for the frozen experiment. Do not silently add constants to costs or fit ordinary least squares on log-cost as if it were the same Gamma model.

**Deliverable:** one positive mean-cost predictor per candidate and a training-mean baseline. With six candidates, Steps 2 and 3 yield six correctness regressions and six cost regressions; with all 11, they yield 11 of each.

### Fit the final regressions after choosing settings

This template expects dictionaries of per-model settings already selected by training-only cross-validation. It is a final fitting template, not the cross-validation procedure itself. Inspect convergence warnings before accepting the outputs.

```python
from sklearn.linear_model import LogisticRegression, GammaRegressor

def fit_final_predictors(X_train, Y_train, cost_train, C_by_model, alpha_by_model):
    probability_models, cost_models = {}, {}
    for m in models:
        assert Y_train[m].nunique() == 2
        probability_models[m] = LogisticRegression(
            C=C_by_model[m], solver="lbfgs", max_iter=2000,
        ).fit(X_train, Y_train[m].to_numpy())
        cost_models[m] = GammaRegressor(
            alpha=alpha_by_model[m], solver="lbfgs", max_iter=2000,
        ).fit(X_train, cost_train[m].to_numpy())
    return probability_models, cost_models

def predict_outputs(X, probability_models, cost_models):
    probabilities = []
    for m in models:
        predictor = probability_models[m]
        positive_column = list(predictor.classes_).index(1)
        probabilities.append(predictor.predict_proba(X)[:, positive_column])
    P = np.column_stack(probabilities)
    C_hat = np.column_stack([cost_models[m].predict(X) for m in models])
    return P, C_hat
```

If calibration is used, replace each entry in `probability_models` with its fitted calibrated predictor exposing `classes_` and `predict_proba` before generating policy-validation predictions. Otherwise the dictionary contains the fitted logistic regressions. Neither dictionary is present in the Step 1 files.

## Step 4 Fix the experiment objective

Choose the aggregate accuracy requirement before selecting the policy. Use either a prespecified absolute target `A`, or an allowed loss `delta` relative to a fixed reference candidate selected on development data. If using the repository convention, select that reference by highest policy-validation accuracy and freeze it; specify a tie rule and the tolerance before final testing.

The goal is to minimize average observed response cost while maintaining the chosen accuracy requirement across questions. This is an aggregate requirement; the router does not need every individual prediction to exceed `A`. The PDF's 0.85 accuracy example is illustrative and is not a measured or adopted project target.

Document the candidate pool, success definition, weighting of duplicate rows, accuracy target or tolerance, cost units, serving overhead treatment, and split limitations together. An unattainable target remains unattainable for the evaluated policies; do not lower it after looking at test results.

## Step 5 Construct the routing policy

Use the final predictors to create two arrays, each with shape `(number_of_questions, len(models))`:

- `P`: predicted success probabilities in `models` order, including calibration if used.
- `C_hat`: predicted historical dollar costs in the same order.

For a nonnegative tradeoff `lambda`, select the candidate minimizing:

```text
score_m(x) = predicted_cost_m(x) + lambda * (1 - predicted_success_m(x))
```

`lambda` has units of dollars per expected failure. It is a decision weight; the score is not a billed cost. There is no additional policy regression to train.

```python
def select_models(P, C_hat, tradeoff):
    P = np.asarray(P, dtype=float)
    C_hat = np.asarray(C_hat, dtype=float)
    assert P.shape == C_hat.shape and P.ndim == 2
    assert P.shape[1] == len(models)
    assert np.isfinite(P).all() and np.isfinite(C_hat).all()
    assert ((P >= 0) & (P <= 1)).all() and (C_hat > 0).all()
    assert np.isfinite(tradeoff) and tradeoff >= 0
    scores = C_hat + tradeoff * (1 - P)
    # Exact ties choose the first candidate in the fixed models list.
    return scores.argmin(axis=1)

def replay(selected, Y, observed_costs):
    assert Y.columns.tolist() == observed_costs.columns.tolist() == models
    assert Y.index.equals(observed_costs.index)
    assert len(selected) == len(Y)
    rows = np.arange(len(selected))
    success = Y.to_numpy(dtype=float)[rows, selected]
    dollars = observed_costs.to_numpy(dtype=float)[rows, selected]
    return success, dollars
```

For example, assemble `P_val` by calling each stored probability predictor on `X_val`, and assemble `C_hat_val` by calling each cost predictor, always iterating through `models`. Keep predictions and observed labels in distinct variables. `select_models` uses predictions; `replay` then looks up the selected response's actual benchmark outcome and cost.

## Step 6 Check the upgrade calculation

Before searching policies, check the score calculation using the PDF's illustrative example:

| Candidate | Predicted dollars | Predicted success | Score at lambda = 0.005 |
| --- | ---: | ---: | ---: |
| Small | 0.0002 | 0.78 | 0.00130 |
| Medium | 0.0006 | 0.87 | 0.00125 |
| Large | 0.0015 | 0.91 | 0.00195 |

Medium has the minimum score. Its predicted response cost is $0.0006. An upgrade from candidate `a` to a more accurate candidate `b` is worthwhile when:

```text
predicted_cost_b - predicted_cost_a
    < lambda * (predicted_success_b - predicted_success_a)
```

These numbers check arithmetic only. They are not predictions fitted by the notebook or measured project outcomes.

## Step 7 Select on validation and evaluate once on test

Prespecify a grid of nonnegative tradeoffs, including zero and positive values spanning inexpensive and accuracy-focused choices. Fit the predictors once, generate validation predictions once, and compare every grid value through observed validation replay. Do not refit models for each tradeoff or assume observed accuracy increases monotonically with it.

The following function requires an explicitly supplied grid and accuracy target; it does not invent project settings:

```python
def choose_policy(P_val, C_hat_val, Y_val, cost_val, lambda_grid, A):
    assert 0 <= A <= 1
    records = []
    for tradeoff in lambda_grid:
        chosen = select_models(P_val, C_hat_val, float(tradeoff))
        success, dollars = replay(chosen, Y_val, cost_val)
        records.append({
            "lambda": float(tradeoff),
            "accuracy": success.mean(),
            "mean_cost": dollars.mean(),
        })
    results = pd.DataFrame(records)
    feasible = results.loc[results["accuracy"] >= A]
    if feasible.empty:
        return results, None  # No evaluated policy meets the target.
    best = feasible.sort_values(["mean_cost", "lambda"]).iloc[0]
    return results, float(best["lambda"])
```

This example compares historical response costs only. If serving overhead differs by policy, incorporate it in the cost comparison and report it separately. For equal-cost feasible policies, this code chooses the smaller tradeoff. Keep that rule fixed.

Once selected, freeze the feature pipeline, candidate order, label rule, fitted regressions, calibrators, tradeoff, reference model, and tie rules. Then load `test`, apply the same label definition, create `P_test` and `C_hat_test`, and replay the chosen policy. Do not refit on calibration or validation data after selecting the policy without a new plan for calibration and selection.

Report test accuracy, mean historical dollars per question, dollars per 1,000 questions, savings relative to the frozen reference, and the paired accuracy difference. Compare with every fixed candidate and the cheapest fixed candidate; the PDF also requests a random mixture whose weights were chosen on validation data. Show routing frequencies, results by `eval_name`, and the validation cost-accuracy curve.

Use paired bootstrap uncertainty estimates that keep candidate outcomes and costs together. Because duplicate rows remain, sample `question_group` clusters and carry all their rows into each resample. Include confidence intervals for savings and for the router-minus-reference accuracy difference. If assessing tolerance `delta`, compare a prespecified lower confidence bound for that difference with `-delta`. A nonsignificant difference alone does not establish acceptable accuracy loss.

If the selected policy misses its target on test, report that result. Test labels must not be used to adjust the selected policy.

**Deliverable:** the complete validation search, frozen policy bundle, and final observed metrics with uncertainty. The notebook currently supplies none of these fitted-router results.

## Reuse the feature pipeline for a new question

After fitting and selecting the policy, use the saved vectorizer and length scaler without calling `fit` again:

```python
def features_for_texts(texts):
    assert all(isinstance(text, str) and text.strip() for text in texts)
    tfidf = preprocessing["tfidf_vectorizer"].transform(texts)
    lengths = pd.DataFrame({"char_length": [len(text) for text in texts]})
    scaled = preprocessing["length_scaler"].transform(lengths)
    return hstack([tfidf, csr_matrix(scaled)], format="csr")

X_new = features_for_texts(["A new question in the same input format"])
# Next: final correctness and cost predictors -> select_models -> one LLM call.
```

Preserve the same prompt assembly used in training. If using serialized list prompts, parse them first with `extract_question_text`. For a future live system, preserve candidate generation settings and revalidate against new behavior and billing; Appendix A of the PDF describes token-based cost modeling. The current saved artifacts support historical replay.

## Handoff checklist and source references

Before beginning model fitting, record the chosen model pool and binary success rule, resolve the feature version, and audit broader question grouping. Archive the raw dataset or record its exact revision and checksum: the notebook download did not specify a Hugging Face revision, and `requirements.txt` does not pin package versions. A fresh download or environment is therefore not guaranteed to reproduce these artifacts.

For the next experiment, record fold group assignments, preprocessing parameters, regression settings and convergence, calibration method, success definition, candidate configurations, historical cost provenance, accuracy target, tradeoff grid, selected policy, reference model, and tie rules. Store final regressions and calibrators with their model order and feature pipeline; the existing `preprocessing.joblib` contains preprocessing only.

Sources used for this handoff:

- [TF-IDF embedding.ipynb](<TF-IDF embedding.ipynb>): executed cells and recorded outputs establish completed notebook work.
- The saved files in `data/processed/char_tfidf_50k/`: inspected to confirm schema, dimensions, alignment, model order, and actual preprocessing settings.
- [The supplied seven-step mathematical guide](../../llm_routing_seven_steps_chat_fonts.pdf): Steps 1 through 7 and Appendices A and B define the proposed regression, routing, and evaluation procedure.
- [README.md](../README.md) and [kpis.md](../kpis.md): project candidate scope and reference comparison conventions.

The supplied PDF is a reference for the proposed experiment. Its examples and recommendations are distinguished here from operations actually completed in the notebook and from new implementation suggestions.
