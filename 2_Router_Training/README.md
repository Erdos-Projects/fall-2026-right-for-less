# Step 2: Five-Class Score and Cost Prediction

This folder contains [TF-IDF_E5_Score_Classification_Router.ipynb](TF-IDF_E5_Score_Classification_Router.ipynb). It compares three prediction methods using pure character TF-IDF and frozen E5 embeddings with four additional features.

For each question, the notebook predicts a score distribution and a dollar cost for **11 candidate LLMs**, using recorded RouterBench outcomes. You do not need to download these LLMs or call their APIs.

## Quick Start

### 1. Download the project

Clone the repository, or use GitHub's **Code > Download ZIP** and extract the project:

```bash
git clone https://github.com/Erdos-Projects/fall-2026-right-for-less.git
cd fall-2026-right-for-less
```

Keep this folder inside the downloaded project.

### 2. Complete Step 1

Run the preparation notebooks in this order:

1. [step1_TFIDF_complete.ipynb](../1_data_preparation/step1_TFIDF_complete.ipynb): download the raw data, create the splits, and save the question, score, cost and TF-IDF files.
2. [step1_e5_complete.ipynb](../1_data_preparation/step1_e5_complete.ipynb): read those split tables, generate E5 embeddings, and save the 388-dimensional features.

Use the **same kernel working directory** for both preparation notebooks. You can check it by running `Path.cwd()` after importing `Path` from `pathlib`. The preparation code writes files relative to that directory.

Initial preparation downloads RouterBench and E5. Prepared data are excluded from Git. Existing compatible outputs from both notebooks can be reused.

The [E5 environment guide](../1_data_preparation/README-e5.md) provides macOS setup instructions. On other operating systems, use an equivalent Python environment with the preparation dependencies installed.

### 3. Select the training environment

Open the training notebook in VS Code or Jupyter. In VS Code, select the Python environment at the top right of the notebook. Reuse the environment used for data preparation when possible.

Step 2 requires NumPy, pandas, SciPy, **scikit-learn >= 1.6**, **XGBoost >= 2.0**, joblib, IPython and Jinja2. If the preparation environment already contains the other packages, install the additional dependencies from a terminal using that environment:

```bash
python -m pip install "xgboost>=2.0" jinja2
```

Keep package versions compatible with the saved preprocessing objects.

### 4. Check the data path

The notebook automatically searches its working directory and parent directories for either supported layout:

```text
project/
  data/processed/
    char_tfidf_50k/
    multilingual_e5_small_chunk480/
  1_data_preparation/
  2_Router_Training/
```

Alternatively, both prepared-data folders may be under:

```text
project/1_data_preparation/data/processed/
```

For these layouts, the default configuration usually needs no changes:

```python
DATA_PREPARATION_DIR = None
```

If data are stored elsewhere, set this variable in the configuration cell to the **folder containing `data/`**. For example, on Windows:

```python
DATA_PREPARATION_DIR = Path(r"D:\ErdosData")
```

This expects both folders below:

```text
D:\ErdosData\data\processed\char_tfidf_50k\
D:\ErdosData\data\processed\multilingual_e5_small_chunk480\
```

Use your computer's actual path. On macOS or Linux, supply the corresponding absolute path.

If both supported layouts contain prepared data, explicitly select the intended directory. The current notebook otherwise uses the first matching location. Confirm the printed **Prepared data** path before training.

### 5. Run the notebook

Run all cells in order. The default configuration uses the complete prepared splits:

```python
RUN_MODE = "full"
```

For an initial execution check, use `RUN_MODE = "smoke"`. To run the complete experiment afterward, switch back to `"full"`, restart the kernel, and run all cells again. Smoke metrics are only an execution check.

The notebook prints the data and output directories and checks data alignment. Training defaults to CPU; a GPU is optional for XGBoost.

## Methods and Outputs

| Method | Score prediction | Cost prediction |
|---|---|---|
| L2 logistic | Class probabilities with L2 regularization | Ridge model on normalized costs |
| kNN | Class frequencies among cosine neighbors | Mean recorded neighbor cost |
| XGBoost | Multiclass probabilities from boosted trees | Squared-error model on normalized costs |

Both feature sets are evaluated with every method:

- **TF-IDF:** pure character TF-IDF. Step 1 saves an additional length column, which this notebook excludes.
- **E5 + features:** 384 frozen `intfloat/multilingual-e5-small` dimensions plus character length, answer-choice count, numeric-value count and a code indicator, for **388 dimensions**.

The five classes `{0, 0.25, 0.5, 0.75, 1}` cover **99.858% of training scores**. Coverage is the number of recorded question-LLM scores exactly equal to an anchor divided by the total number of training scores. Rare values map to the nearest anchor. Numerical errors use the original scores.

Each distribution produces two outputs:

- **Hard class:** the anchor with the highest probability.
- **Expected score:** the probability-weighted average of the five anchors.

Both outputs share the same cost predictions. Logistic regularization is selected through three-fold grouped cross-validation on training data. XGBoost settings are fixed. The main kNN comparison uses **k = 40** for both feature sets; the notebook also evaluates `1, 2, 3, 5, 10, 20, 40`.

## Evaluation Metrics

All reported metrics evaluate predictions on the same validation questions. Let $i=1,\ldots,n$ index questions and $m=1,\ldots,11$ index candidate LLMs:

- $s_{im}$: original recorded score, between 0 and 1.
- $c_{im}$: recorded dollar cost.
- $a_j$: score anchor, with $a=(0,0.25,0.5,0.75,1)$ and $j=0,\ldots,4$.
- $z_{im}=\arg\min_j|s_{im}-a_j|$: mapped class used for classification; midpoint ties use the lower anchor.
- $p_{imj}$: predicted probability of class $j$; the five probabilities sum to one.
- $\hat c_{im}$: predicted dollar cost.

The predicted class and the two numerical score outputs are:

$$
\hat z_{im}=\arg\max_j p_{imj},\qquad
\hat s^{\mathrm{hard}}_{im}=a_{\hat z_{im}},\qquad
\hat s^{\mathrm{expected}}_{im}=\sum_{j=0}^{4}a_jp_{imj}.
$$

An exact probability tie selects the lower anchor. Hard scores belong to the five anchors; expected scores can take intermediate values.

### Classification and probability metrics

**Accuracy** is the fraction of questions whose predicted score class matches the mapped recorded class:

$$
\mathrm{Accuracy}_m=\frac{1}{n}\sum_{i=1}^{n}
\mathbf{1}[\hat z_{im}=z_{im}].
$$

Here, $\mathbf{1}[\cdot]$ is 1 when the condition is true and 0 otherwise. Accuracy measures score-category prediction, not question-answering correctness or successful LLM selection. Higher is better.

**Balanced accuracy** gives equal weight to each true score class present in validation. For $J_m=\{z_{im}:i=1,\ldots,n\}$, first compute class recall, then average it:

$$
\mathrm{Recall}_{mj}=
\frac{\sum_i\mathbf{1}[z_{im}=j\;\mathrm{and}\;\hat z_{im}=j]}
{\sum_i\mathbf{1}[z_{im}=j]},\qquad
\mathrm{BalancedAccuracy}_m=
\frac{1}{|J_m|}\sum_{j\in J_m}\mathrm{Recall}_{mj}.
$$

Recall is the proportion of examples of a particular true class that are correctly classified. Balanced accuracy helps expose performance hidden by class imbalance. All five classes are present for every LLM in the reference validation run. Higher is better.

**Negative log-likelihood (NLL)** measures the probability assigned to the observed class:

$$
\mathrm{NLL}_m=-\frac{1}{n}\sum_i
\log\left(\max(p_{im,z_{im}},10^{-12})\right).
$$

The logarithm is natural. Confident incorrect predictions receive a large penalty. The probability floor keeps the calculation finite when a class receives zero probability. Lower is better; a lower NLL alone does not establish calibrated probabilities.

**Zero true-class probability rate** counts predictions that assign exactly zero probability to the observed class, before applying the NLL floor:

$$
\mathrm{ZeroRate}_m=\frac{1}{n}\sum_i
\mathbf{1}[p_{im,z_{im}}=0].
$$

This is particularly useful for unsmoothed kNN frequencies: the rate is nonzero when the true class does not occur among the neighbors. Lower is better.

### Numerical score and cost errors

For the following formulas, $(y_{im},\hat y_{im})$ is either an **original score** and its hard/expected prediction, or a **recorded dollar cost** and its prediction. Numerical score errors use the original $s_{im}$.

**Mean squared error (MSE)** averages squared prediction errors and puts more weight on large errors:

$$
\mathrm{MSE}_m=\frac{1}{n}\sum_i(\hat y_{im}-y_{im})^2.
$$

**Mean absolute error (MAE)** averages absolute prediction errors:

$$
\mathrm{MAE}_m=\frac{1}{n}\sum_i|\hat y_{im}-y_{im}|.
$$

Score MAE is measured on the 0-to-1 score scale; cost MAE is in dollars. MSE uses squared units, including dollars squared for costs. Lower is better for both metrics.

**Weighted absolute percentage error (WAPE)** expresses aggregate absolute error relative to aggregate recorded value:

$$
\mathrm{WAPE}_m=
\frac{\sum_i|\hat y_{im}-y_{im}|}{\sum_i|y_{im}|}.
$$

Tables display this ratio as a percentage. The calculation avoids division by individual zero scores. WAPE is neither a fraction of wrong predictions nor the average of per-question percentage errors. Lower is better.

### Aggregation across LLMs and undefined values

Compute each metric separately for each candidate LLM, then average equally across the 11 LLMs:

$$
\overline T=\frac{1}{11}\sum_{m=1}^{11}T_m.
$$

In particular, reported WAPE is the mean of the 11 per-LLM ratios; it is not a single ratio obtained by pooling every LLM's scores or dollar costs.

If a target column has $\sum_i|y_{im}|=0$, its WAPE is undefined. The notebook excludes undefined WAPEs from the mean and reports the number of defined columns:

$$
N_{\mathrm{defined}}(y)=\sum_{m=1}^{11}
\mathbf{1}\left[\sum_i|y_{im}|>0\right].
$$

`Defined_score_WAPE_models` and `Defined_cost_WAPE_models` apply this count to scores and costs respectively. Both counts are **11** throughout the reference results. For a run with undefined columns, the mean is:

$$
D_y=\left\{m:\sum_i|y_{im}|>0\right\},\qquad
\overline{\mathrm{WAPE}}=
\frac{1}{|D_y|}\sum_{m\in D_y}\mathrm{WAPE}_m.
$$

If no columns have a defined WAPE, the mean is also undefined.

Hard and expected outputs share cost predictions, so their cost errors are identical.

## Reference Validation Results

The completed full run used **21,900 training**, **3,648 calibration** and **3,649 validation** questions, with seed 42. Calibration predictions were saved without fitting a calibrator. The test split was not evaluated.

Training-only cross-validation selected **C = 1** for both feature sets. XGBoost used **100 boosting rounds** with fixed settings. All main kNN results below use **k = 40**.

The **training-frequency baseline** predicts each LLM's training class frequencies for every question. Its hard score is the most frequent mapped training class, its expected score is the average of mapped training scores, and its cost is the mean recorded training cost. It uses no question features.

Bold values mark the best unrounded result in each metric column of a table.

### 1. Classification and probability quality

| Method | Features | Accuracy | Balanced accuracy | NLL | Zero true-class probability rate |
|---|---|---|---|---|---|
| Training frequency | Baseline | 53.8% | 20.0% | 1.1813 | 0.00% |
| L2 logistic | TF-IDF | 68.6% | 45.1% | 0.6335 | 0.00% |
| L2 logistic | E5 + features | 69.1% | 44.7% | **0.6321** | 0.00% |
| kNN, k = 40 | TF-IDF | 68.0% | 43.9% | 0.7586 | 0.42% |
| kNN, k = 40 | E5 + features | 68.9% | **45.5%** | 0.6875 | 0.24% |
| XGBoost | TF-IDF | **69.5%** | 45.2% | 0.6499 | 0.00% |
| XGBoost | E5 + features | 68.6% | 43.9% | 0.6654 | 0.00% |

XGBoost + TF-IDF has the highest accuracy; L2 logistic + E5 has the lowest NLL; E5-kNN has the highest balanced accuracy. The baseline's 20% balanced accuracy comes from always predicting one class: recall is 1 for that class and 0 for the other four. kNN can assign zero probability to a true class absent from its neighbors.

### 2. Numerical score errors: hard versus expected

Both tables compare predictions with **original recorded scores**. Bold values indicate the lowest error within each table. The baseline uses the same hard/expected definitions as the learned classifiers.

**Hard class**

| Method | Features | Score MSE | Score MAE | Score WAPE |
|---|---|---|---|---|
| Training frequency | Baseline | 0.30006 | 0.34141 | 72.34% |
| L2 logistic | TF-IDF | 0.24292 | 0.26217 | 55.14% |
| L2 logistic | E5 + features | 0.23780 | 0.25706 | 54.01% |
| kNN, k = 40 | TF-IDF | 0.24176 | 0.26316 | 55.75% |
| kNN, k = 40 | E5 + features | 0.23800 | 0.25784 | 54.17% |
| XGBoost | TF-IDF | **0.23497** | **0.25389** | **53.14%** |
| XGBoost | E5 + features | 0.24284 | 0.26211 | 55.09% |

**Expected score**

| Method | Features | Score MSE | Score MAE | Score WAPE |
|---|---|---|---|---|
| Training frequency | Baseline | 0.17798 | 0.38109 | 82.35% |
| L2 logistic | TF-IDF | 0.15528 | **0.32267** | **67.88%** |
| L2 logistic | E5 + features | 0.15389 | 0.32968 | 69.43% |
| kNN, k = 40 | TF-IDF | 0.15624 | 0.32777 | 68.94% |
| kNN, k = 40 | E5 + features | 0.15327 | 0.32546 | 68.29% |
| XGBoost | TF-IDF | **0.15307** | 0.33246 | 69.50% |
| XGBoost | E5 + features | 0.15630 | 0.33922 | 71.49% |

Across all six learned configurations, **hard scores have lower MAE and WAPE**, while **expected scores have lower MSE**. XGBoost + TF-IDF has the lowest hard-score MAE (0.25389) and expected-score MSE (0.15307). L2 logistic + TF-IDF has the lowest expected-score MAE (0.32267). Expected-score WAPE remains about 68% to 71%, so substantial score error remains. The preferred output depends on the evaluation objective; these diagnostics alone do not establish which output routes better.

### 3. Dollar-cost prediction

| Method | Features | Cost MSE (USD squared) | Cost MAE (USD) | Cost WAPE |
|---|---|---|---|---|
| Training frequency | Baseline | 1.753e-06 | $0.00058106 | 61.48% |
| L2 logistic | TF-IDF | 3.371e-07 | $0.00016267 | 16.80% |
| L2 logistic | E5 + features | 2.948e-07 | $0.00015382 | 14.09% |
| kNN, k = 40 | TF-IDF | 5.027e-07 | $0.00022048 | 25.33% |
| kNN, k = 40 | E5 + features | **2.216e-07** | **$0.00009858** | **10.28%** |
| XGBoost | TF-IDF | 2.262e-07 | $0.00011740 | 12.42% |
| XGBoost | E5 + features | 2.467e-07 | $0.00011330 | 11.71% |

At k = 40, E5-kNN has the lowest cost MSE, MAE and WAPE in the main comparison. Its MAE is approximately **$0.00009858** and its WAPE is **10.28%**, compared with **61.48%** for the mean-cost baseline. WAPE gives context for small dollar errors. Costs reflect the dataset's historical benchmark pricing.

### 4. kNN neighbor-count experiment

The sweep uses the same validation questions for every k. These tables report **expected-score** errors; classification metrics and costs are shared by the hard and expected outputs. Full hard-score results for every k are displayed in the notebook and exported to `knn_score_classification_grid.csv`.

**TF-IDF**

| k | Accuracy | NLL | Score MSE | Score MAE | Score WAPE | Cost WAPE |
|---|---|---|---|---|---|---|
| 1 | 61.1% | 10.7512 | 0.29381 | **0.31870** | **67.68%** | 24.267% |
| 2 | 58.5% | 6.1788 | 0.22371 | 0.32119 | 67.76% | 22.974% |
| 3 | 63.5% | 4.1228 | 0.19982 | 0.32250 | 68.10% | **22.885%** |
| 5 | 65.2% | 2.5076 | 0.17980 | 0.32277 | 67.91% | 22.887% |
| 10 | 66.6% | 1.3621 | 0.16552 | 0.32371 | 68.15% | 23.230% |
| 20 | 67.6% | 0.8969 | 0.15883 | 0.32556 | 68.45% | 24.046% |
| 40 | **68.0%** | **0.7586** | **0.15624** | 0.32777 | 68.94% | 25.326% |

**E5 + features (388 dimensions)**

| k | Accuracy | NLL | Score MSE | Score MAE | Score WAPE | Cost WAPE |
|---|---|---|---|---|---|---|
| 1 | 61.6% | 10.6162 | 0.29272 | 0.31637 | 66.77% | 13.130% |
| 2 | 59.4% | 5.9256 | 0.22011 | 0.31578 | **66.48%** | 11.528% |
| 3 | 64.4% | 3.8178 | 0.19454 | **0.31570** | 66.50% | 10.814% |
| 5 | 65.8% | 2.1431 | 0.17535 | 0.31800 | 66.82% | 10.281% |
| 10 | 66.9% | 1.1003 | 0.16177 | 0.32042 | 67.35% | **10.024%** |
| 20 | 68.2% | 0.7670 | 0.15572 | 0.32284 | 67.77% | 10.035% |
| 40 | **68.9%** | **0.6875** | **0.15327** | 0.32546 | 68.29% | 10.281% |

Among the tested values, **k = 40** gives the highest accuracy, lowest NLL and lowest expected-score MSE for both representations. Cost WAPE is lowest at **k = 3 for TF-IDF (22.885%)** and **k = 10 for E5 (10.024%)**. Smaller k can reduce expected-score MAE while worsening MSE or NLL. At k = 1, predictions are one-hot, so every incorrect class prediction receives the NLL floor penalty. There is no single k that minimizes every error measure.

These comparisons are exploratory validation results. Further configuration selection should be completed before evaluating the untouched test split. Routing benefits at a fixed budget have not been evaluated here. E5 includes auxiliary features absent from TF-IDF, so the comparison does not isolate the embedding representation alone.

## Saved Results

With a standard layout, each run creates a new timestamped folder under:

```text
2_Router_Training/experiment_outputs/score_classification_5class/
```

With external data, the default output directory is derived from that data location. Set `OUTPUT_BASE_OVERRIDE` to choose another destination. Check the printed **Outputs** path.

Saved files include calibration and validation probabilities, hard scores, expected scores, dollar costs, all tested kNN settings, evaluation tables, fitted Logistic/Ridge/XGBoost models, split identifiers and an experiment manifest. Probability arrays have shape `(questions, 11, 5)`.

Outputs are generated automatically; they do not need to be uploaded before running. Saved predictions support further evaluation without retraining.

## If Files Cannot Be Found

Check the following before changing model code:

1. Both Step 1 notebooks completed successfully, including the E5 metadata export.
2. TF-IDF and E5 outputs are under the same `data/processed/` directory.
3. The kernel working directory is inside the downloaded project, or `DATA_PREPARATION_DIR` points to the folder containing `data/`.
4. The selected kernel contains the required packages.

Correcting a path does not require regenerating existing compatible embeddings. After adjusting the configuration, restart the kernel and run the notebook from the beginning.
