# Step 2: Five-Class Score and Cost Prediction

## Changes from the Previous Version

Compared with the [previous version (2026-10-17)](https://github.com/Erdos-Projects/fall-2026-right-for-less/blob/0e9cacbdfb3d5f82e176ae42010bb509342ca799/2_Router_Training/README.md), before the LogNormal/Gamma extension, there are three main experimental changes. CV means cross-validation.

| Area | Previous version | Current version |
|---|---|---|
| Logistic CV diagnostics | Records NLL and class accuracy | Adds hard- and expected-score MSE/MAE for each C; NLL still selects C |
| LogNormal costs | Not included | Adds log-cost Ridge with training-only smearing and grouped training CV for alpha |
| Gamma costs | Not included | Adds Gamma regression with a log link and grouped training CV for alpha |

The two added cost families are included in the final calibration comparison and saved prediction outputs.

This folder contains [TF-IDF_E5_Score_Classification_Router.ipynb](TF-IDF_E5_Score_Classification_Router.ipynb). It compares three score classifiers and five cost predictors using pure character TF-IDF and frozen E5 embeddings with four additional features.

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

Step 2 requires NumPy, pandas, SciPy, **scikit-learn >= 1.6**, **XGBoost >= 2.0**, joblib, threadpoolctl, matplotlib, IPython and Jinja2. If the preparation environment already contains the other packages, install the additional dependencies from a terminal using that environment:

```bash
python -m pip install "xgboost>=2.0" jinja2 matplotlib threadpoolctl
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

### 5. Restart the kernel and run all cells

Select **Restart Kernel**, then **Run All**. The default configuration uses all prepared training, calibration and validation rows:

```python
RUN_MODE = "full"
```

A complete run repeats training CV for Logistic C, Ridge alpha and the two kNN neighbor counts, fits the selected predictors and XGBoost, then runs LogNormal and Gamma cost experiments. Each additional cost family searches alpha once per representation. LogNormal E5 uses the requested final-fit alpha of 0; its CV selection remains recorded separately. XGBoost uses 100 boosting rounds in full mode. Prepared TF-IDF and frozen E5 features are reused.

**Full results use the fixed `full/` folder.** A rerun overwrites the same generated filenames and preserves `archive_before_*/` folders. Ownership checks guard existing outputs. Use the current manifests to identify this run's files; unrelated or obsolete files are not listed as new results. Archive any active results you want to keep before another full run.

For an initial execution check, set `RUN_MODE = "smoke"`, restart the kernel and run all cells. Smoke uses a small subset and 10 XGBoost rounds, writes a separate timestamped folder, and leaves `full/` unchanged. To train the complete experiment afterward, switch back to `"full"`, restart the kernel and run all cells again. Smoke metrics are only an execution check.

The notebook prints **Prepared data** and **Outputs**, checks alignment and stops if a Logistic CV or final head fails to converge. A completed run has `status == "completed"` in `run_status.json`. A failed or interrupted run may contain intermediate files. Completion is recorded only after all predictors, exports and final comparisons pass the notebook's checks.

Training defaults to CPU; a GPU is optional for XGBoost.

## Experiment Overview

For a walkthrough, follow the sections in this order: prediction targets, evaluation metrics, input representations, prediction methods, and predictor results on calibration data. Metrics are defined before they are used for model selection.

The experiment has **three score classifiers** and **five cost predictors**, each with two representations: six score configurations and ten cost configurations. The three original pipelines export score/cost bundles. LogNormal and Gamma add cost predictions that can be paired with any existing score head. Hard and expected scores summarize the same class distribution.

| Method | Score prediction | Cost prediction |
|---|---|---|
| L2 logistic | Linear logits, softmax probabilities and L2-regularized cross-entropy | Linear Ridge model on normalized costs |
| kNN | Uniform class frequencies among cosine neighbors | Mean recorded neighbor cost |
| XGBoost | Boosted-tree logits and multiclass cross-entropy | Separate boosted trees with squared-error loss |
| LogNormal | Reuse an existing score head | Ridge on log cost, exponentiation and training-only smearing |
| Gamma | Reuse an existing score head | L2-regularized Gamma mean with a log link |

### Split roles

| Split | Role in the revised workflow |
|---|---|
| **Train** | Fit preprocessing and predictors; choose Logistic C, Ridge/LogNormal/Gamma alpha and separate kNN score/cost neighbor counts through grouped training CV |
| **Calibration** | Compare the selected base predictors and inspect prediction errors |
| **Validation** | Develop and assess downstream routing policies after fixing the base predictors |
| **Test** | Final assessment of the selected complete routing system |

Calibration is the development split for predictor comparisons and diagnostics; no probability calibrator is fitted. Earlier work, including the supplied cost notebooks, inspected validation and test outcomes. This run excludes test from fitting and evaluation; those earlier inspections remain part of the experimental history.

**About the numerical tables:** CV results and unchanged predictor results come from completed full run `20261028T010931_284078Z` (2026-10-27 local / 2026-10-28 UTC). LogNormal E5 calibration results use a targeted alpha-0 refit completed **2026-10-28 at 02:15 PDT**, which took **56 s**. Only its final fit, training-only smearing and dependent outputs were recomputed; the original CV results were preserved. For the original complete run, smoke took **4 min 19 s** and full training and checks took **2 h 24 min 51 s**, including **42 min 27 s for LogNormal** and **25 min 51 s for Gamma**. Settings and results are recorded in `full/experiment_manifest.json` and its CSVs.

## Prediction Targets and Shared Notation

The task is to predict how each candidate LLM would perform on a question, using that question's features. The router's predictors learn from recorded question-LLM outcomes. They do not generate answers during this experiment.

The following notation is shared by all methods:

| Symbol | Meaning |
|---|---|
| $i$ | Question-row index in the split currently being discussed |
| $N$ | Number of training questions; training sums run from $i=1$ to $N$ |
| $n$ | Number of calibration questions evaluated here; predictor-evaluation sums run from $i=1$ to $n$ |
| $m\in\{1,\ldots,11\}$ | Candidate LLM index |
| $j,j'\in\{0,1,2,3,4\}$ | Score-class indices; $j'$ is used when summing over classes |
| $q_i$ | Text of question $i$ |
| $d$ | Number of input features for the chosen representation |
| $\phi$ | Feature transformation: TF-IDF or E5 with additional features |
| $x_i=\phi(q_i)\in\mathbb R^d$ | Real-valued feature vector for question $i$; $x$ denotes a new question's vector |
| $s_{im}$ | Original recorded score of LLM $m$ on question $i$, between 0 and 1 |
| $c_{im}$ | Original recorded dollar cost of that LLM on that question |
| $a_j$ | Numerical score represented by class $j$: $a_0=0$, $a_1=0.25$, $a_2=0.5$, $a_3=0.75$, $a_4=1$ |
| $z_{im}$ | Mapped class **index**, assigned to the original score $s_{im}$ |

A hat marks a prediction. For example, $\hat c_m(x)$ is the predicted dollar cost of calling LLM $m$ on a question with features $x$. Subscripts identify the question, LLM or class; they do not imply multiplication.

Some mathematical conventions used below:

- `argmin` and `argmax` return the **index** giving the smallest or largest value. `min` and `max` return the value itself. Exact class ties select the lower index.
- $\mathbf1[B]$ is an indicator: it equals 1 when condition $B$ is true and 0 otherwise.
- $\log$ is the natural logarithm; $\exp$ is the exponential function.
- $\|x\|_2$ is a vector's Euclidean length, the square root of the sum of its squared components. For equally sized vectors $w$ and $x$, a superscript $\mathsf T$ means transpose, and $w^{\mathsf T}x$ is the sum of their component-by-component products.
- Single vertical bars mean absolute value for a number and the number of elements for a set. Summation limits identify which rows, classes or other elements are added.

### Map recorded scores to five classes

The anchor vector is $a=(0,0.25,0.5,0.75,1)$. Assign each recorded score to its nearest anchor, and store that anchor's index as $z_{im}$:

$$
z_{im}=\arg\min_{j\in\{0,\ldots,4\}}|s_{im}-a_j|.
$$

An exact midpoint selects the lower anchor. The five anchors cover **99.858% of training scores**: 240,559 of 240,900 recorded question-LLM scores. The remaining 341 scores are quantized; all questions are retained. Coverage is the number of original scores exactly equal to an anchor divided by the total number of scores.

#### Why use five classes?

The original training data already concentrate on these five values. The table counts **question-LLM score entries before mapping**: 21,900 training questions multiplied by 11 candidate LLMs gives 240,900 entries. Training share is each row's count divided by 240,900, displayed as a rounded percentage.

| Original recorded score | Training count | Training share |
|---|---:|---:|
| 0 | 93,268 | 38.716% |
| 0.25 | 15,064 | 6.253% |
| 0.5 | 8,075 | 3.352% |
| 0.75 | 27,679 | 11.490% |
| 1 | 96,473 | 40.047% |
| Other values | 341 | 0.142% |
| **Total** | **240,900** | **100.000%** |

| Consideration | Why it supports this choice |
|---|---|
| Most labels already match the anchors | **99.858%** of original training score entries keep their numerical value. |
| Other labels have little support | Only **341 entries (0.142%)** fall outside the anchors. Nearest-anchor mapping avoids fitting separate classes supported by very few observations. |
| One common output format | Every LLM predictor returns probabilities over the same five ordered anchors, supporting both hard and expected scores. |
| Information loss and interpretation | Rare values are merged into nearby anchors. Five-class accuracy evaluates this coarser target; this simplification does **not guarantee better predictive accuracy** or remove grading uncertainty. |

This **label quantization** simplifies sparse categories and loses some information. For example, 0.6 maps to 0.5 and 0.9 maps to 1. The probability of the class anchored at 1 therefore refers to the quantized class. Numerical error metrics, defined in the next section, continue to use the original scores. Original score and cost files remain unchanged.

### Turn class probabilities into numerical predictions

Let $p_{mj}(x)$ be the estimated probability that LLM $m$ receives mapped class $j$ on a question with features $x$. For each fixed LLM, its five class probabilities sum to one; the 11 LLMs each have their own distribution. For recorded question $i$, write $p_{imj}=p_{mj}(x_i)$.

Define $\hat z_m(x)$ as the predicted class index, $\hat s_m^{\mathrm{hard}}(x)$ as its numerical anchor, and $\hat s_m^{\mathrm{expected}}(x)$ as the probability-weighted score:

$$
\hat z_m(x)=\arg\max_{j\in\{0,\ldots,4\}}p_{mj}(x),\qquad
\hat s_m^{\mathrm{hard}}(x)=a_{\hat z_m(x)},\qquad
\hat s_m^{\mathrm{expected}}(x)=\sum_{j=0}^{4}a_jp_{mj}(x).
$$

The hard output selects the most probable anchor; the expected output retains information from all five probabilities. Both use the same predicted cost. For calibration rows, $\hat z_{im}=\hat z_m(x_i)$, and the corresponding numerical outputs are written $\hat s_{im}^{\mathrm{hard}}$ and $\hat s_{im}^{\mathrm{expected}}$. Likewise, $\hat c_{im}=\hat c_m(x_i)$ denotes the predicted dollar cost for that recorded question.

## Evaluation Metrics

The predictor diagnostics in this section evaluate the same calibration questions. Thus $i=1,\ldots,n$, while $N$ remains the training-set size. The recorded scores, mapped classes, probabilities and predictions use the notation defined above.

### Negative log-likelihood (NLL)

**NLL** evaluates the probability assigned to the observed class. For question $i$ and LLM $m$, $p_{im,z_{im}}$ means: select the probability entry whose class index equals the mapped recorded label $z_{im}$. It is a probability, not the numerical score anchor.

Let $\epsilon=10^{-12}$ be the probability floor used for numerical evaluation. Define $\mathrm{NLL}_m$ as the average negative log-probability for LLM $m$:

$$
\mathrm{NLL}_m=-\frac{1}{n}\sum_{i=1}^{n}
\log\left(\max(p_{im,z_{im}},\epsilon)\right).
$$

**Lower is better.** Assigning probability 0.8 to the true class gives a per-example loss of about 0.223; assigning 0.1 gives about 2.303. A confident incorrect prediction receives a large penalty. The floor keeps the result finite when the predicted true-class probability is zero.

For a single observed class per example, this is the categorical cross-entropy evaluation criterion with the stated numerical floor. Training objectives below use cross-entropy and may add regularization; that penalty is excluded from calibration NLL. A lower NLL alone does not establish calibrated probabilities.

Cross-validation applies the same negative-log-probability criterion to held-out **training** rows. Its use for choosing logistic regularization is explained under Prediction Methods.

### Accuracy, recall and balanced accuracy

#### General definitions: TP, TN, FP and FN

Start with a **binary classification** task, where one class is called **positive** and the other **negative**. These names identify the two classes; they carry no judgment about whether an outcome is desirable. Compare each predicted label with its actual label, then count the four possible cases:

| Count | Full name | Actual label | Predicted label | Interpretation |
|---|---|---|---|---|
| **TP** | True positive | Positive | Positive | A positive example correctly identified |
| **TN** | True negative | Negative | Negative | A negative example correctly identified |
| **FP** | False positive | Negative | Positive | A negative example incorrectly labeled positive |
| **FN** | False negative | Positive | Negative | A positive example missed by the classifier |

Each symbol is a **number of examples**. Their sum is the total number of evaluated examples.

**Accuracy** is the fraction of all examples classified correctly, counting both correctly identified positives and correctly identified negatives. **Recall**, also called the true positive rate, is the fraction of actual positive examples identified as positive. When at least one actual positive example exists, their binary formulas are

$$
\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN}.
$$

The recall denominator counts all actual positives, including those missed by the classifier. If no actual positives exist, this recall fraction is undefined.

#### Apply the definitions to five score classes

In this project, $m$ identifies one of the 11 candidate LLMs, $j$ identifies a score class, and $n$ is the number of calibration questions. The recorded mapped class is $z_{im}$ and the predicted class is $\hat z_{im}$ for question $i$.

For a fixed LLM $m$ and class $j$, treat **class $j$ as positive and all other classes as negative**. This is a one-versus-rest view. For example, to evaluate detection of the class anchored at score 1, choose $j=4$. Positive then means membership in that mapped score class.

Attach subscripts $m,j$ to the four counts to identify the chosen LLM and class:

| Count | Condition for question $i$ |
|---|---|
| $TP_{mj}$ | $z_{im}=j$ and $\hat z_{im}=j$ |
| $TN_{mj}$ | $z_{im}\ne j$ and $\hat z_{im}\ne j$ |
| $FP_{mj}$ | $z_{im}\ne j$ and $\hat z_{im}=j$ |
| $FN_{mj}$ | $z_{im}=j$ and $\hat z_{im}\ne j$ |

**Five-class accuracy for LLM $m$**, denoted $\mathrm{Accuracy}_m$, counts exact matches across all five classes:

$$
\mathrm{Accuracy}_m=\frac{1}{n}\sum_{i=1}^{n}
\mathbf{1}[\hat z_{im}=z_{im}].
$$

The binary accuracy formula above evaluates a positive-versus-negative decision. The five-class metric here requires the specific score-class label to match. For example, confusing the 0.25 and 0.5 score classes counts as a true negative when detecting the score-1 class, but remains an incorrect five-class prediction. Compute the reported five-class accuracy directly from exact matches.

Accuracy measures score-category prediction, not question-answering correctness or successful LLM selection. Higher is better.

**Recall for LLM $m$ and class $j$**, denoted $\mathrm{Recall}_{mj}$, measures how many actual class-$j$ examples are correctly identified. For a class present in calibration, its denominator is nonzero:

$$
\mathrm{Recall}_{mj}=
\frac{TP_{mj}}{TP_{mj}+FN_{mj}}=
\frac{\sum_{i=1}^{n}\mathbf{1}[z_{im}=j\;\mathrm{and}\;\hat z_{im}=j]}
{\sum_{i=1}^{n}\mathbf{1}[z_{im}=j]}.
$$

Thus $\mathrm{Accuracy}_m$ describes overall class-prediction performance for one LLM, while $\mathrm{Recall}_{mj}$ describes detection of one particular score class for that LLM.

#### Balanced accuracy: give each of the five classes equal weight

**Balanced accuracy**, denoted $\mathrm{BalancedAccuracy}_m$, gives equal weight to the five score classes. For LLM $m$, average the class recalls $\mathrm{Recall}_{mj}$ over class indices $j=0,\ldots,4$:

$$
\mathrm{BalancedAccuracy}_m=
\frac{1}{5}\sum_{j=0}^{4}\mathrm{Recall}_{mj}.
$$

This formula assumes that **all five true score classes have samples in calibration** for the LLM being evaluated. This assumption holds for every LLM in the reported calibration run. A class with actual samples but no correct predictions has recall 0.

Balanced accuracy helps expose performance hidden by class imbalance. Higher is better. As with the other reported metrics, compute it separately for each candidate LLM before averaging across the 11 LLMs.

### Zero true-class probability rate

**Zero true-class probability rate** counts predictions that assign exactly zero probability to the observed class, before applying the NLL floor:

$$
\mathrm{ZeroRate}_m=\frac{1}{n}\sum_i
\mathbf{1}[p_{im,z_{im}}=0].
$$

This is particularly useful for unsmoothed kNN frequencies: the rate is nonzero when the true class does not occur among the neighbors. Lower is better.

### Numerical score and cost errors

For MSE and MAE, $(y_{im},\hat y_{im})$ is either an **original score** and its hard/expected prediction, or a **recorded dollar cost** and its prediction. Numerical score errors use the original $s_{im}$.

**Mean squared error (MSE)** averages squared prediction errors and puts more weight on large errors:

$$
\mathrm{MSE}_m=\frac{1}{n}\sum_i(\hat y_{im}-y_{im})^2.
$$

**Mean absolute error (MAE)** averages absolute prediction errors:

$$
\mathrm{MAE}_m=\frac{1}{n}\sum_i|\hat y_{im}-y_{im}|.
$$

Score MAE is measured on the 0-to-1 score scale; cost MAE is in dollars. MSE uses squared units, including dollars squared for costs. Lower is better for both metrics.

**Cost weighted absolute percentage error (Cost WAPE)** expresses aggregate absolute dollar error relative to aggregate recorded dollars:

$$
\mathrm{CostWAPE}_m=
\frac{\sum_i|\hat c_{im}-c_{im}|}{\sum_i c_{im}}.
$$

WAPE is used **only for cost prediction**. All recorded costs are positive. Tables display this ratio as a percentage; it is neither a fraction of wrong predictions nor the average of per-question percentage errors. Lower is better. Numerical score predictions are evaluated using MSE and MAE on the original 0-to-1 scale.

### Aggregation across LLMs

For any one metric, let $U_m$ denote its value for LLM $m$, and let $\overline U$ denote the overall mean. Compute the metric separately for each candidate LLM, then average equally across the 11 LLMs:

$$
\overline U=\frac{1}{11}\sum_{m=1}^{11}U_m.
$$

Reported Cost WAPE is the equal-weight mean of the 11 per-LLM cost ratios. Compute each LLM's ratio separately before averaging. Score MSE and MAE are also averaged equally across the 11 LLMs.

Hard and expected outputs share cost predictions, so their cost errors are identical.

## Input Representations

Both representations turn each question into a numerical vector. The model receives these vector components as input.

| Representation used in this experiment | Input for one question | Number of components |
|---|---|---:|
| **Character TF-IDF** | A sparse vector of weighted character sequences of lengths 2 to 4. Most components are zero; vocabulary and weights are learned from training text. | Up to **50,000**, determined by the fitted vocabulary |
| **E5 + additional features** | A dense, frozen E5 semantic embedding followed by four scalar features describing the question. | **384 + 4 = 388** |

TF-IDF stands for term frequency-inverse document frequency. It weights character patterns using their occurrence counts and their frequency across training questions. Each nonzero TF-IDF vector is normalized to Euclidean length 1. Step 1 also saves a length column alongside TF-IDF; this notebook uses the **pure TF-IDF columns** for that representation.

E5 uses `intfloat/multilingual-e5-small` to produce a **384-dimensional semantic vector** for each question. These 384 components are learned representations; individual components have no assigned meaning such as "length" or "difficulty." E5 remains frozen in this experiment.

The **388-dimensional input** consists of that same 384-dimensional embedding plus the following four features:

| Additional feature | What it measures | How it enters the vector |
|---|---|---|
| Character length | Number of characters in the question text | Standardized using training statistics |
| Answer-choice count | Number of detected multiple-choice answer options | Standardized using training statistics |
| Numeric-value count | Number of numeric values detected in the question text | Standardized using training statistics |
| Code indicator | Whether the question is identified as code-related | Binary value: **1** for detected code-related text, **0** otherwise |

Choice counts, numeric counts and code detection use the preparation notebook's heuristic rules. Standardization subtracts the training mean and divides by the training standard deviation; constant features use scale 1. The same training statistics are reused on calibration and validation.

The main experiment uses the full **388-dimensional E5 input**. It receives no further normalization after the four features are appended, and kNN computes cosine distances using all 388 components.

## Prediction Methods

In each method below, $x_i$ means the chosen TF-IDF or 388-dimensional E5 feature vector. Each method is applied separately to the two representations. Score and cost are separate prediction targets.

### 1. L2 logistic classification and Ridge cost prediction

#### Score classifier

For each candidate LLM, **multinomial logistic regression** learns five linear scores, called logits. A logit is a raw real-valued score before conversion to a probability.

For class $j$ of LLM $m$, define $w_{mj}\in\mathbb R^d$ as its learned feature-weight vector, $b_{mj}$ as its scalar intercept (a learned constant offset), and $g_{mj}(x)$ as its logit. Stack the five weight vectors as the rows of $W_m\in\mathbb R^{5\times d}$, and collect the five intercepts in $b_m$.

The linear calculation and the softmax conversion are

$$
g_{mj}(x)=w_{mj}^{\mathsf T}x+b_{mj},\qquad
p_{mj}(x)=\frac{\exp(g_{mj}(x))}{\sum_{j'=0}^{4}\exp(g_{mj'}(x))}.
$$

Softmax produces five nonnegative probabilities that sum to one. Let $C>0$ be scikit-learn's inverse regularization-strength parameter. Let $\mathcal L_m$ denote the training objective for LLM $m$, and let $\|W_m\|_F^2$ denote the sum of squares of all entries in $W_m$, equivalently the sum of squared lengths of its five row-weight vectors. Intercepts are unpenalized.

For the unweighted scikit-learn formulation used here, minimize

$$
\mathcal L_m(W_m,b_m;C)=
-\frac1N\sum_{i=1}^{N}\log p_{m,z_{im}}(x_i)
+\frac{1}{2NC}\|W_m\|_F^2,\qquad
\|W_m\|_F^2=\sum_{j=0}^{4}\|w_{mj}\|_2^2.
$$

Cross-entropy rewards probability assigned to the observed class. The L2 term discourages large weights; smaller `C` gives stronger regularization. If $\lambda$ denotes the coefficient in the alternative convention 'mean cross-entropy plus $\lambda\|W_m\|_F^2/2$', then $\lambda=1/(NC)$ for this unweighted formulation. Evaluation NLL omits the penalty. Standard cross-entropy uses class probabilities without an explicit penalty based on the numerical distance between anchors.

#### Selecting C within training data

For each representation, search **C = 0.01, 0.1, 1, 10, 100, 1000 and 10000** using three shuffled `GroupKFold` folds with seed 42. This broad grid uses tenfold spacing across six orders of magnitude. Exact-duplicate question groups stay in one fold. Within each fold, fit the TF-IDF vectorizer or the E5 auxiliary-feature scalers on the fitting rows; reuse the frozen E5 vectors.

Let $f\in\{1,2,3\}$ be the fold index, $H_f$ the set of held-out training-row indices in fold $f$, and $|H_f|$ the number of such rows. Each training row is held out once, so $\sum_f|H_f|=N$. The superscript $(-f,C)$ means the classifier was fitted on the other two folds using candidate value $C$. In each fold's training objective, replace $N$ by the fitting-row count $N-|H_f|$ and sum only over those fitting rows.

Define $\mathrm{CV}(C)$ as the held-out NLL, averaged over questions and all 11 LLMs. Using the previously defined probability floor $\epsilon=10^{-12}$, the criterion is

$$
\mathrm{CV}(C)=
-\frac{1}{11\sum_{f=1}^{3}|H_f|}
\sum_{f=1}^{3}\sum_{i\in H_f}\sum_{m=1}^{11}
\log\max\left(p^{(-f,C)}_{m,z_{im}}(x_i),\epsilon\right).
$$

This averages equally over LLMs and held-out questions; fold means are weighted by fold size. Choose the lowest value, favoring smaller C on an exact tie, then refit all 11 score classifiers on the full training split. The recorded three-fold CV results are below. Hard and expected MSE/MAE use the original recorded scores and average equally over the 11 LLMs; both outputs come from the same held-out class probabilities.

**TF-IDF**

| C | CV NLL | CV accuracy | Hard MSE | Hard MAE | Expected MSE | Expected MAE |
|---|---|---|---|---|---|---|
| 0.01 | 0.79056 | 66.30% | 0.26127 | 0.28215 | 0.16657 | 0.36172 |
| 0.1 | 0.67271 | 67.92% | 0.24562 | 0.26628 | 0.15657 | 0.34110 |
| 1 | **0.64416** | 68.18% | 0.24623 | 0.26588 | 0.15732 | 0.32625 |
| 10 | 0.74019 | 65.80% | 0.26722 | 0.28753 | 0.18119 | 0.31767 |
| 100 | 1.05925 | 63.77% | 0.28137 | 0.30301 | 0.21546 | 0.31492 |
| 1000 | 1.46015 | 63.47% | 0.28264 | 0.30464 | 0.23133 | 0.31370 |
| 10000 | 1.53591 | 63.26% | 0.28372 | 0.30601 | 0.23321 | 0.31388 |

**E5 + features**

| C | CV NLL | CV accuracy | Hard MSE | Hard MAE | Expected MSE | Expected MAE |
|---|---|---|---|---|---|---|
| 0.01 | 0.70581 | 65.91% | 0.26202 | 0.28354 | 0.16681 | 0.36007 |
| 0.1 | 0.66615 | 67.64% | 0.24676 | 0.26767 | 0.15866 | 0.34455 |
| 1 | **0.64187** | 68.40% | 0.24355 | 0.26333 | 0.15622 | 0.33221 |
| 10 | 0.64269 | 68.07% | 0.24747 | 0.26708 | 0.15821 | 0.32791 |
| 100 | 0.64623 | 67.88% | 0.24920 | 0.26885 | 0.15957 | 0.32660 |
| 1000 | 0.64743 | 67.85% | 0.24938 | 0.26905 | 0.15990 | 0.32688 |
| 10000 | 0.64674 | 67.84% | 0.24970 | 0.26933 | 0.15980 | 0.32683 |

The lowest recorded CV NLL occurs at **C = 1 for both representations**. Bold NLL marks the implemented selection. Accuracy and numerical errors are auxiliary diagnostics.

For comparison, choosing C by another metric from the same saved CV results would give:

| Selection criterion | TF-IDF best C | E5 best C |
|---|---|---|
| NLL minimum (implemented) | 1 | 1 |
| Hard-class accuracy maximum | 1 | 1 |
| Hard-score MSE minimum | 0.1 | 1 |
| Expected-score MSE minimum | 0.1 | 1 |
| Hard-score MAE minimum | 1 | 1 |
| Expected-score MAE minimum | 1000 | 100 |

These alternative criteria do not change the fitted models: hard and expected outputs both use the NLL-selected C = 1. Expected-score MSE would instead select C = 0.1 for TF-IDF and C = 1 for E5. C is inverse regularization strength, so smaller C means stronger L2 regularization.

All **462 CV heads** and **22 final score heads** converged in the reported results, with **0 convergence warnings**. Warning counts include optimization attempts; an earlier warning can be followed by a successful retry. The unconverged count concerns the final outcome of each fit.

Optimization uses `lbfgs`, tolerance `1e-3`, initially 300 iterations, and continuation retries with limits of 1,000 and 2,000. Each candidate C starts a separate fit; a retry continues that same fit. The seven-C search fits **462 CV heads**, followed by **22 final score heads**. CV records also include hard- and expected-score MSE/MAE against original scores; both outputs share the C selected by NLL. Per-head records are in `linear_cv_heads.csv`, with final diagnostics in `linear_final_fit.csv`. No class or sample weighting is applied.

#### Ridge cost predictor

Cost is a separate numerical target. Let $\mu_m$ be LLM $m$'s mean training dollar cost, and let $r_{im}$ be its normalized recorded cost on training question $i$. First divide each recorded cost by that LLM's training mean:

$$
\mu_m=\frac1N\sum_{i=1}^{N} c_{im},\qquad r_{im}=\frac{c_{im}}{\mu_m}.
$$

For each candidate LLM, let $\beta_m\in\mathbb R^d$ be the learned cost-weight vector, $\beta_{0m}$ its intercept, and $\alpha\geq0$ its L2 penalty strength. The predicted normalized cost is $\hat r_m(x)=\beta_{0m}+x^{\mathsf T}\beta_m$. Ridge fits these coefficients by minimizing

$$
\sum_{i=1}^{N}\left(r_{im}-\beta_{0m}-x_i^{\mathsf T}\beta_m\right)^2
+\alpha\|\beta_m\|_2^2.
$$

The code fits one multi-output Ridge object per representation, with separate coefficients for the 11 cost columns. Intercepts are unpenalized. The solver is `lsqr`, with tolerance `1e-3` and at most 1,000 iterations.

**Ridge alpha selection:** Use three-fold grouped cross-validation on the training split to choose Ridge alpha separately for each representation. Each fold refits its feature preprocessing and computes cost-normalization means using only its fitting rows. Each training question receives one out-of-fold prediction; calibration, validation and test labels are excluded from selection. Choose the lowest mean Cost WAPE; exact ties favor the larger alpha.

The table reports training cross-validation errors in dollars. MSE and MAE are computed over all out-of-fold questions for each LLM, then averaged across the 11 LLMs. For WAPE, divide each LLM's total absolute dollar error by its total recorded dollars before averaging the 11 ratios. The metric definitions above apply. Lower is better for all three metrics; bold values mark the lowest error within each representation.

| Features | Alpha | CV Cost WAPE | CV Cost MAE (USD) | CV Cost MSE (USD squared) |
|---|---|---|---|---|
| TF-IDF | 0.01 | 28.655% | $0.00026508 | 5.119e-07 |
| TF-IDF | 0.1 | 22.026% | $0.00021296 | 3.979e-07 |
| TF-IDF | 1 | 17.710% | $0.00017437 | **3.565e-07** |
| TF-IDF | 10 | **17.143%** | **$0.00016572** | 3.840e-07 |
| TF-IDF | 100 | 21.130% | $0.00019832 | 4.885e-07 |
| TF-IDF | 1000 | 37.441% | $0.00034136 | 8.347e-07 |
| TF-IDF | 10000 | 56.701% | $0.00053619 | 1.566e-06 |
| E5 + features | 0.01 | 14.858% | $0.00016266 | **3.117e-07** |
| E5 + features | 0.1 | 14.770% | $0.00016149 | 3.118e-07 |
| E5 + features | 1 | 14.210% | $0.00015438 | 3.148e-07 |
| E5 + features | 10 | **13.293%** | **$0.00014230** | 3.360e-07 |
| E5 + features | 100 | 14.120% | $0.00014891 | 3.851e-07 |
| E5 + features | 1000 | 15.713% | $0.00016237 | 4.444e-07 |
| E5 + features | 10000 | 23.836% | $0.00022698 | 5.559e-07 |

The lowest recorded CV Cost WAPE occurs at **alpha 10 for both representations**. The other metric minima are TF-IDF MAE: alpha 10; TF-IDF MSE: alpha 1; E5 + features MAE: alpha 10; E5 + features MSE: alpha 0.1. Each complete run selects alpha within training and refits the chosen cost heads on the full training split. Cost WAPE gives equal weight to each LLM's absolute error relative to its recorded cost scale; its minimizer need not minimize squared error. Full results are saved in `cost_ridge_cv/cost_cv_summary.csv`.

Convert predictions back to dollars using

$$
\hat c_m(x)=\mu_m\max\left(\hat r_m(x),10^{-6}\right).
$$

The floor is applied in normalized units before restoring dollars, so the dollar floor is $10^{-6}\mu_m$. XGBoost cost predictions use this same normalization and restoration rule.

### 2. Cosine kNN: local class frequencies and mean cost

kNN (k-nearest neighbors) predicts from similar training questions. For each representation, build a brute-force cosine-neighbor index. Define $d_{\cos}(x,x_i)$ as the cosine distance between query vector $x$ and training vector $x_i$. For nonzero feature vectors, it is

$$
d_{\cos}(x,x_i)=1-\frac{x^{\mathsf T}x_i}{\|x\|_2\|x_i\|_2}.
$$

Let $k$ be the number of neighbors, and let $N_k(x)$ be the set of **training-row indices** of the $k$ questions with the smallest cosine distances to $x$. Using the indicator defined earlier, estimate each class probability from **uniform class counts** and cost from original dollar values:

$$
p_{mj}(x;k)=\frac1k\sum_{i\in N_k(x)}\mathbf1[z_{im}=j],\qquad
\hat c_m(x;k)=\frac1k\sum_{i\in N_k(x)}c_{im}.
$$

Distance determines membership in the neighborhood; all selected neighbors receive equal weight. The same neighbor questions supply outcomes for all 11 LLMs. The expected-score calculation consequently equals the mean of the neighbors' mapped scores:

$$
\hat s_m^{\mathrm{expected}}(x;k)=\frac1k\sum_{i\in N_k(x)}a_{z_{im}}.
$$

The hard output is the modal mapped class. A class absent from the neighborhood receives probability zero. NLL evaluates these frequency estimates; it is not used to fit a parametric score classifier.

#### Select separate neighbor counts within training

For each representation, let $k_{\mathrm{score}}$ be the neighbor count used for score-class probabilities, and let $k_{\mathrm{cost}}$ be the count used for dollar costs. Search the fixed set $\mathcal K=\{1,2,3,5,10,20,40,60,80,100\}$ using three grouped folds within training. Exact-duplicate question groups stay together. Each fold refits TF-IDF or the E5 auxiliary-feature scalers on its fitting rows and builds the cosine index from those rows only. Frozen E5 vectors are reused.

For the formulas below, $N$ is the number of training questions, $f=1,2,3$ indexes a fold, and $H_f$ contains its held-out row indices. Every training question belongs to one $H_f$. The superscript $(-f)$ means predictions from the other two folds. For candidate $k$, $p_{mj}^{(-f)}(x_i;k)$ is LLM $m$'s predicted class-$j$ probability on held-out question $i$, and $\hat c_m^{(-f)}(x_i;k)$ is its predicted dollar cost.

Choose $k_{\mathrm{score}}$ by the highest class accuracy across the 11 LLMs and all held-out training questions:

$$
\mathrm{CVAccuracy}(k)=\frac{1}{11N}
\sum_{f=1}^{3}\sum_{i\in H_f}\sum_{m=1}^{11}
\mathbf1\!\left[
\arg\max_{j\in\{0,\ldots,4\}}p_{mj}^{(-f)}(x_i;k)=z_{im}
\right].
$$

Choose $k_{\mathrm{cost}}$ by the lowest mean per-LLM Cost WAPE. First pool each LLM's dollar errors across all out-of-fold questions, then compute its ratio and average the 11 ratios:

$$
\mathrm{CVCostWAPE}(k)=\frac1{11}\sum_{m=1}^{11}
\frac{\sum_{f=1}^{3}\sum_{i\in H_f}
|\hat c_m^{(-f)}(x_i;k)-c_{im}|}
{\sum_{i=1}^{N}c_{im}}.
$$

For either selection, an exact tie favors the larger $k$.

After selection, rebuild the neighbor index using all training rows. Main-comparison probabilities use $k_{\mathrm{score}}$, and dollar costs use $k_{\mathrm{cost}}$. Hard and expected scores summarize the same selected probability distribution and share the selected cost prediction. Calibration and validation labels do not participate in either selection.

Retrieve the largest requested neighbor count once per query and reuse its prefixes for all candidates, processing queries in batches of 128. All candidate calibration and validation predictions are exported for reuse; calibration metrics are reported only for the selected main predictor.

### 3. XGBoost: boosted trees for scores and costs

XGBoost builds shallow decision trees sequentially. Each round uses the loss's first- and second-order derivatives to choose trees that reduce the current ensemble's errors. Small updates accumulate into a nonlinear predictor.

The score classifier uses these quantities:

- $R$: total number of boosting rounds; $t=1,\ldots,R$ indexes a round.
- $F^{(t)}_{mj}(x)$: accumulated raw score, or margin, for LLM $m$ and class $j$ after round $t$; $F^{(0)}_{mj}(x)$ is the initial margin.
- $f_{tmj}(x)$: output of the new tree added for that LLM and class in round $t$.
- $\eta$: learning rate, which scales the contribution of the new tree.

Add the tree's contribution, then convert the final class margins into probabilities:

$$
F^{(t)}_{mj}(x)=F^{(t-1)}_{mj}(x)+\eta f_{tmj}(x),\qquad
p_{mj}(x)=\frac{\exp(F^{(R)}_{mj}(x))}
{\sum_{j'=0}^{4}\exp(F^{(R)}_{mj'}(x))}.
$$

One score classifier and one separate scalar cost model are fitted per LLM and representation. The cost model predicts the normalized target $r_{im}$ defined in the Ridge section; here $\hat r_m(x)$ denotes the boosted-tree model's normalized cost prediction. Let $\ell_{\mathrm{score},i}$ and $\ell_{\mathrm{cost},i}$ be their respective per-example losses for question $i$ and a fixed LLM $m$:

$$
\ell_{\mathrm{score},i}=-\log p_{m,z_{im}}(x_i),\qquad
\ell_{\mathrm{cost},i}=\tfrac12\left(r_{im}-\hat r_m(x_i)\right)^2.
$$

Each model adds a tree-complexity penalty to its summed loss. For the next equation, define:

- $\mathcal L$: total training objective for one score or cost model.
- $\ell_i$: the corresponding per-example loss just defined; sum it over $i=1,\ldots,N$.
- $\mathcal T$: that model's set of trees; $f$ denotes one tree in this set.
- $\Omega(f)$: complexity penalty for tree $f$.
- $L_f$: number of leaves in tree $f$; $b=1,\ldots,L_f$ indexes a leaf.
- $w_{fb}$: learned output value stored at leaf $b$ of tree $f$.
- $\gamma$: penalty per leaf, controlling the leaf-count contribution.
- $\lambda_{\mathrm{leaf}}$: L2 penalty on leaf values, corresponding to XGBoost's `reg_lambda`.

The objective is

$$
\mathcal L=\sum_{i=1}^{N}\ell_i+\sum_{f\in\mathcal T}\Omega(f),\qquad
\Omega(f)=\gamma L_f+\frac{\lambda_{\mathrm{leaf}}}{2}
\sum_{b=1}^{L_f}w_{fb}^2.
$$

The notebook sets `reg_lambda = 1`; `gamma = 0` is the library default. The active penalty here therefore discourages large leaf values. The score objective is `multi:softprob`; the cost objective is `reg:squarederror`, using the normalized costs and dollar-restoration formula above.

| Parameter | Setting |
|---|---|
| Boosting rounds | 100 in full mode; 10 in smoke mode |
| Maximum depth | 3 |
| Learning rate | 0.05 |
| Training-row subsample | 0.8 |
| Feature subsample per tree | 0.8 |
| Histogram bins | At most 64 per feature |
| Tree method | `hist` |
| Leaf L2 penalty | `reg_lambda = 1.0` |
| Device / CPU threads per fit | `cpu` / 4 |
| Random seed | 42 |

These settings are fixed: no XGBoost cross-validation or early stopping is performed. Under the default strategy, five-class classification adds one tree per class per round: 100 rounds produce 500 class-specific trees per score classifier. Each scalar cost model uses 100 trees. The reported final score classifiers observed all five classes.

### 4. LogNormal cost regression with smearing

LogNormal fits normalized **log cost**, then restores positive dollar predictions. Both added families require finite, strictly positive recorded costs ($c_{im}>0$). Reuse the Ridge section's $\mu_m$, $r_{im}=c_{im}/\mu_m$, coefficients $\beta_m$, intercept $\beta_{0m}$ and penalty $\alpha$. Each cost family learns its own coefficients. The equations describe the full-training fit; within CV, restrict sums and cost means to the fitting rows.

Let $\widehat\eta_m(x)$ be the fitted log-normalized cost. Its Ridge objective is

$$
\min_{\beta_{0m},\beta_m}
\sum_{i=1}^{N}\left[\log r_{im}-\beta_{0m}-x_i^{\mathsf T}\beta_m\right]^2
+\alpha\|\beta_m\|_2^2,
\qquad \widehat\eta_m(x)=\beta_{0m}+x^{\mathsf T}\beta_m.
$$

Fitting log cost reduces the influence of large dollar values, but exponentiation alone does not recover the mean dollar cost. Define $\widetilde c_{im}^{\mathrm{OOF}}$ as an **unsmeared dollar prediction** made without question $i$'s group. Let $S_m$ be the mean observed-to-predicted ratio. Smearing adjusts the exponentiated prediction:

$$
S_m=\frac1N\sum_{i=1}^{N}\frac{c_{im}}{\widetilde c_{im}^{\mathrm{OOF}}},
\qquad \widehat c_m(x)=\mu_m S_m\exp\!\left(\widehat\eta_m(x)\right).
$$

Each outer CV fit estimates $S_m$ through **three grouped inner folds using only its fitting rows**. Inner models restore their own fitting cost means before entering the ratio. Final smearing uses three folds of the full training set. Calibration and validation outcomes do not enter this correction; one global factor need not remove bias for every question.

### 5. Gamma cost regression with a log link

Gamma regression models positive normalized costs directly. Its deviance responds to the observed-to-predicted ratio. For this separately fitted model, let $\widehat r_m(x)$ be the predicted normalized mean. For positive values $y$ (observed) and $u$ (predicted), define Gamma deviance $d_\Gamma(y,u)$:

$$
\widehat r_m(x)=\exp(\beta_{0m}+x^{\mathsf T}\beta_m),
\qquad d_\Gamma(y,u)=2\left[\log\left(\frac{u}{y}\right)+\frac{y}{u}-1\right].
$$

Minimize the regularized loss, then restore dollars:

$$
\min_{\beta_{0m},\beta_m}
\frac1{2N}\sum_{i=1}^{N}d_\Gamma\left(r_{im},\widehat r_m(x_i)\right)
+\frac{\alpha}{2}\|\beta_m\|_2^2,
\qquad \widehat c_m(x)=\mu_m\widehat r_m(x).
$$

The log link makes predictions positive; Gamma needs no smearing. LogNormal uses `Ridge(solver="lsqr")`; Gamma uses `GammaRegressor`. Both use tolerance `1e-6`, at most 5,000 iterations and unpenalized intercepts. They export cost point predictions. Their loss scalings differ, so alpha magnitudes are not directly comparable across families.

#### Training-only alpha selection for LogNormal and Gamma

Use the same **three grouped training folds, seed 42**, with fold-local TF-IDF/scalers and fitting-only cost means. Each family/representation runs one search. Candidate grids are:

- **LogNormal / TF-IDF:** 0.01, 0.1, 1, 10, 100, 1000, 10000.
- **LogNormal / E5:** 0, 1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000.
- **Gamma / both:** 1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1.

Let $\widehat c_{im}^{(\alpha,\mathrm{OOF})}$ be the dollar prediction for training question $i$ while its group is held out, including inner smearing for LogNormal. With $N$ training questions, select alpha by

$$
\mathrm{CVCostWAPE}(\alpha)=\frac1{11}\sum_{m=1}^{11}
\frac{\sum_{i=1}^{N}|\widehat c_{im}^{(\alpha,\mathrm{OOF})}-c_{im}|}
{\sum_{i=1}^{N}c_{im}}.
$$

Pool each LLM's three-fold predictions before computing its ratio, then average the 11 ratios. Choose the lowest unrounded value; exact ties favor larger alpha. **One alpha per family/representation is shared by its 11 separately fitted LLM heads.** Refit on all training rows, using the requested alpha-0 override for LogNormal E5. Calibration, validation and test labels do not select alpha.

| Cost family | Features | CV selected alpha | Final-fit alpha | Training OOF Cost WAPE at CV selection |
|---|---|---|---|---|
| LogNormal | TF-IDF | 1 | 1 | 13.0008% |
| LogNormal | E5 + features | 0.0001 | 0 | 12.6638% |
| Gamma | TF-IDF | 0.0001 | 0.0001 | 13.1005% |
| Gamma | E5 + features | 0.0001 | 0.0001 | 13.0613% |

LogNormal E5 is nearly tied at alpha 0 and 1e-4: CV WAPE is 12.663942% and 12.663831%, respectively. **The CV winner remains 1e-4.** The requested final fit uses alpha 0 for all 11 E5 heads and for their training-only smearing fits. This explicit final-fit override leaves the CV selection unchanged. The manifests distinguish `cv_selected_alpha` from `fit_alpha`; the existing `selected_alpha` field retains the CV result. All original candidates and CV results remain in `lognormal_cv_summary.csv` and `gamma_cv_summary.csv`.

### 6. Training-frequency baseline

The baseline ignores question features. Define $p^{\mathrm{base}}_{mj}$ as LLM $m$'s class-$j$ frequency in training, and $\hat c_m^{\mathrm{base}}$ as its constant dollar-cost prediction. Reuse $\mu_m$, the mean training dollar cost defined above:

$$
p^{\mathrm{base}}_{mj}=\frac1N\sum_{i=1}^{N}\mathbf1[z_{im}=j],\qquad
\hat c_m^{\mathrm{base}}=\mu_m.
$$

These values are constant across questions. The hard score is the most frequent mapped class; the expected score is $\sum_j a_jp^{\mathrm{base}}_{mj}$. This measures how much question-dependent predictions add beyond each LLM's average behavior.

#### How the baseline predicts a new question

For each candidate LLM, first compute its five class frequencies and mean dollar cost using **training data only**. For a new question, copy those stored values as its predicted probabilities and cost. Then calculate the hard and expected scores from those probabilities. The question's text does not change the baseline prediction.

**Illustrative example with hypothetical values.** Suppose one LLM has the following training class frequencies and a mean training cost of `$0.0002`. These hypothetical values illustrate the calculation.

| Mapped score anchor | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---:|---:|---:|---:|---:|
| Training frequency / predicted probability | 0.1 | 0.1 | 0.2 | 0.2 | 0.4 |

For **every new question**, this LLM's baseline predictions are:

- **Class probabilities:** `[0.1, 0.1, 0.2, 0.2, 0.4]`.
- **Hard score:** **1**, the anchor with the largest probability, 0.4.
- **Expected score:** **0.675**, calculated as shown below.
- **Dollar cost:** **`$0.0002`**, copied from the mean training cost independently of the score probabilities.

$$
0\times0.1+0.25\times0.1+0.5\times0.2+0.75\times0.2+1\times0.4=0.675.
$$

Repeat this procedure separately for all 11 candidate LLMs. Each LLM has its own training frequencies and mean cost. Compare the resulting predictions with recorded calibration outcomes using the same metrics as the learned methods. This tests how much the question-dependent predictors improve over each LLM's overall historical behavior.

Score and cost predictors are fitted separately. Cost is a prediction target here; no routing-cost penalty is added to the score classifier's cross-entropy.

Implementation references: [TF-IDF weighting](https://scikit-learn.org/1.7/modules/feature_extraction.html#tfidf-term-weighting), [multilingual E5 model card](https://huggingface.co/intfloat/multilingual-e5-small), [Logistic objective in scikit-learn 1.7.2](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/linear_model/_logistic.py), [Ridge objective](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.Ridge.html), and [XGBoost objectives and tree regularization](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).

## Predictor Results on Calibration Data

The reported experiment used **21,900 training questions** and evaluated **3,648 calibration questions**. Its fitted settings are below; LogNormal E5 uses the requested alpha-0 override while its CV winner remains 0.0001:

| Features | Logistic C | Ridge alpha | kNN score k / cost k | LogNormal fit alpha | Gamma alpha |
|---|---|---|---|---|---|
| TF-IDF | 1 | 10 | 80 / 5 | 1 | 0.0001 |
| E5 + features | 1 | 10 | 80 / 10 | 0 | 0.0001 |

XGBoost used 100 boosting rounds. Calibration compares the selected predictors; the **3,649 validation predictions** support downstream routing-policy development. Neither split selects the training-CV hyperparameters. No probability calibrator was fitted and test was not evaluated. Bold values mark the best unrounded metric within each main-comparison table.

### 1. Classification and probability quality

| Method | Features | Accuracy | Balanced accuracy | NLL | Zero true-class probability rate |
|---|---|---|---|---|---|
| Training frequency | Baseline | 53.6% | 20.0% | 1.1800 | **0.00%** |
| L2 logistic | TF-IDF | 68.7% | **45.3%** | 0.6372 | **0.00%** |
| L2 logistic | E5 + features | 69.0% | 45.0% | **0.6341** | **0.00%** |
| kNN (train-CV selected) | TF-IDF | 67.8% | 43.6% | 0.7134 | 0.22% |
| kNN (train-CV selected) | E5 + features | 68.6% | 44.9% | 0.6752 | 0.17% |
| XGBoost | TF-IDF | **69.1%** | 44.8% | 0.6426 | **0.00%** |
| XGBoost | E5 + features | 69.1% | 44.3% | 0.6549 | **0.00%** |

XGBoost + TF-IDF has the highest accuracy (69.1%); L2 logistic + TF-IDF has the highest balanced accuracy (45.3%); L2 logistic + E5 + features has the lowest NLL (0.6341). These metrics assess score-category prediction, not successful routing. All five true classes occur for every LLM in calibration; the training-frequency baseline has 20% balanced accuracy.

### 2. Numerical score errors: hard versus expected

Both outputs are compared with the **original recorded calibration scores**. They summarize the same class distribution and share dollar-cost predictions.

**Hard class**

| Method | Features | Score MSE | Score MAE |
|---|---|---|---|
| Training frequency | Baseline | 0.30310 | 0.34426 |
| L2 logistic | TF-IDF | 0.24258 | 0.26189 |
| L2 logistic | E5 + features | 0.23950 | 0.25871 |
| kNN (train-CV selected) | TF-IDF | 0.24316 | 0.26499 |
| kNN (train-CV selected) | E5 + features | 0.24119 | 0.26106 |
| XGBoost | TF-IDF | **0.23839** | **0.25762** |
| XGBoost | E5 + features | 0.23928 | 0.25842 |

**Expected score**

| Method | Features | Score MSE | Score MAE |
|---|---|---|---|
| Training frequency | Baseline | 0.17883 | 0.38213 |
| L2 logistic | TF-IDF | 0.15571 | **0.32245** |
| L2 logistic | E5 + features | 0.15427 | 0.32957 |
| kNN (train-CV selected) | TF-IDF | 0.15652 | 0.33046 |
| kNN (train-CV selected) | E5 + features | 0.15512 | 0.33063 |
| XGBoost | TF-IDF | **0.15414** | 0.33337 |
| XGBoost | E5 + features | 0.15598 | 0.33907 |

Among the six learned configurations, hard scores have lower MAE in **6/6**, while expected scores have lower MSE in **6/6**. Compare both score summaries on calibration before fixing the routing inputs.

### 3. Dollar-cost prediction

All five cost predictors use the same calibration questions and equal weighting across the 11 LLMs. **Ridge is the cost head paired with Logistic score classification.** LogNormal and Gamma add separate cost predictions; they do not change the score heads. Metrics are defined above; lower is better.

| Cost method | Features | Cost MSE (USD squared) | Cost MAE (USD) | Cost WAPE |
|---|---|---|---|---|
| Training mean | Baseline | 1.873e-06 | $0.00058923 | 61.46% |
| Ridge | TF-IDF | 4.211e-07 | $0.00016767 | 17.24% |
| Ridge | E5 + features | 3.680e-07 | $0.00014571 | 13.49% |
| kNN | TF-IDF | 5.764e-07 | $0.00021138 | 23.41% |
| kNN | E5 + features | **2.227e-07** | **$0.00009879** | **10.42%** |
| XGBoost | TF-IDF | 2.844e-07 | $0.00012104 | 12.65% |
| XGBoost | E5 + features | 3.003e-07 | $0.00011780 | 12.18% |
| LogNormal | TF-IDF | 3.829e-07 | $0.00012570 | 12.97% |
| LogNormal | E5 + features | 5.236e-07 | $0.00012865 | 13.44% |
| Gamma | TF-IDF | 3.899e-07 | $0.00012787 | 13.20% |
| Gamma | E5 + features | 5.889e-07 | $0.00013386 | 14.23% |

**LogNormal E5 before and after the requested refit:**

| Final-fit alpha | Cost MSE (USD squared) | Cost MAE (USD) | Cost WAPE |
|---|---|---|---|
| 0.0001 (previous) | 5.243562e-07 | $0.000128666466 | 13.442423% |
| 0 (current) | 5.236276e-07 | $0.000128653225 | 13.440900% |

The calibration WAPE decrease is 0.001524 percentage points; both fits round to 13.44% in the main table.

- **E5 + kNN is best on all three summary metrics.** Its WAPE is also lowest for each of the 11 LLMs individually in this run.
- With TF-IDF, LogNormal and Gamma improve Ridge WAPE from 17.24% to 12.97% and 13.20%; XGBoost remains lower at 12.65%.
- With E5, the alpha-0 LogNormal fit has WAPE 13.44% and MSE 5.236e-07; Ridge has 13.49% and 3.680e-07, respectively. Gamma WAPE is 14.23%. Assess both typical absolute errors and the stronger penalty on large errors.

These are predictor diagnostics using historical benchmark prices. Actual routing savings at a chosen quality level require evaluating a routing policy with recorded scores and costs. The full comparison is in `calibration_cost_comparison_summary.csv`, with per-LLM rows in `calibration_cost_comparison_per_llm.csv`.

### 4. Training-only kNN neighbor-count selection

These are three-fold **training cross-validation** results. Each training question is held out once. Accuracy, NLL and numerical errors are computed from out-of-fold predictions and averaged equally across the 11 LLMs. Score MSE and MAE use expected scores and original recorded scores. Cost WAPE pools all held-out dollar errors within each LLM before averaging the 11 ratios.

Select score k by the highest CV accuracy and cost k by the lowest CV Cost WAPE. Exact ties favor larger k. Selections use unrounded values. Accuracy is shown to three decimal places to distinguish close candidates. Bold values identify the selected criterion in its column; other metrics are diagnostics.

| Features | Selected score k | Selected cost k |
|---|---|---|
| TF-IDF | 80 | 5 |
| E5 + features | 80 | 10 |

**TF-IDF**

| k | CV accuracy | CV NLL | CV Score MSE | CV Score MAE | CV Cost WAPE |
|---|---|---|---|---|---|
| 1 | 60.392% | 10.9441 | 0.29944 | 0.32470 | 25.632% |
| 2 | 58.409% | 6.2737 | 0.22727 | 0.32502 | 24.450% |
| 3 | 63.243% | 4.2248 | 0.20246 | 0.32503 | 24.032% |
| 5 | 64.622% | 2.5455 | 0.18272 | 0.32588 | **23.952%** |
| 10 | 66.051% | 1.3703 | 0.16794 | 0.32670 | 24.252% |
| 20 | 67.003% | 0.9165 | 0.16129 | 0.32844 | 25.217% |
| 40 | 67.146% | 0.7600 | 0.15862 | 0.33034 | 26.711% |
| 60 | 67.209% | 0.7207 | 0.15796 | 0.33149 | 27.559% |
| 80 | **67.215%** | 0.7061 | 0.15778 | 0.33230 | 28.138% |
| 100 | 67.199% | 0.6999 | 0.15777 | 0.33317 | 28.586% |

**E5 + features**

| k | CV accuracy | CV NLL | CV Score MSE | CV Score MAE | CV Cost WAPE |
|---|---|---|---|---|---|
| 1 | 60.633% | 10.8776 | 0.29888 | 0.32355 | 13.504% |
| 2 | 58.458% | 6.0764 | 0.22450 | 0.32276 | 11.815% |
| 3 | 63.269% | 3.9503 | 0.19968 | 0.32307 | 11.160% |
| 5 | 64.659% | 2.2353 | 0.18032 | 0.32371 | 10.697% |
| 10 | 66.042% | 1.1365 | 0.16604 | 0.32525 | **10.351%** |
| 20 | 67.224% | 0.7947 | 0.15969 | 0.32780 | 10.399% |
| 40 | 67.789% | 0.7035 | 0.15716 | 0.33044 | 10.547% |
| 60 | 67.891% | 0.6860 | 0.15675 | 0.33213 | 10.680% |
| 80 | **67.919%** | 0.6793 | 0.15685 | 0.33345 | 10.856% |
| 100 | 67.875% | 0.6744 | 0.15696 | 0.33429 | 11.018% |

Calibration evaluates the selected predictor pair above. No all-k calibration metric sweep is used here. Fix the base predictors before choosing routing cost weights, budgets or thresholds on validation, then evaluate the selected complete system on test. E5 includes auxiliary features absent from TF-IDF, so this comparison does not isolate the embedding alone.


## Saved Results

A successful full run writes its complete result set to the fixed directory:

```text
2_Router_Training/experiment_outputs/score_classification_5class/full/
```

The overwrite, archive-preservation and ownership checks are described in [Quick Start](#quick-start). Smoke runs use separate timestamped folders.

With external data, the default output base is derived from that data location. Set `OUTPUT_BASE_OVERRIDE` to choose another base; full results are written to its `full/` child. Confirm the printed **Outputs** path.

| Files | Contents and use |
|---|---|
| `run_status.json` | Identifies the owned output directory and records running/completed status, timing and the completed prediction-file count. |
| `experiment_manifest.json` | LLM/class order, split sizes, actual selected Logistic C, Ridge alpha, separate kNN counts, model settings, versions and execution scope. |
| `predictions_manifest.csv` | Main predictions for three methods, two representations and both calibration/validation splits. kNN records `score_k` and `cost_k`; its single `k` field is populated only when both are equal. |
| `knn_predictions_manifest.csv` | Prediction prefixes for all ten k values, both representations and both splits; `selected_for_score` and `selected_for_cost` identify the selected components. |
| `*_probability.npy` | Class probabilities with shape `(questions, 11, 5)`; preserve the manifest's LLM and anchor order. |
| `*_hard_score.pkl`, `*_expected_score.pkl`, `*_dollar_cost.pkl` | Question-by-LLM score and dollar-cost DataFrames. |
| `calibration_classification_summary.csv` and `calibration_classification_per_llm.csv` | Classification/probability metrics for the selected main predictors, overall and per LLM. |
| `calibration_numeric_summary.csv` and `calibration_numeric_per_llm.csv` | Score MSE/MAE and cost MSE/MAE/WAPE for hard and expected scores. |
| `linear_cv_summary.csv`, `linear_cv_folds.csv` and `linear_cv_heads.csv` | Seven-C training CV, including NLL, accuracy, hard/expected MSE/MAE and convergence records. Full mode fits 462 CV heads. |
| `linear_final_fit.csv` | Fit/convergence diagnostics for the 22 selected final Logistic score heads. |
| `cost_ridge_cv/` | Seven-alpha training CV, including summary, fold/per-LLM errors, out-of-fold dollar predictions, training rows and the CV manifest. |
| `knn_cv/` | Ten-k training CV summaries, fold/per-LLM records, the selected score/cost pair, training rows, manifest and selected out-of-fold predictions. |
| `linear_models.joblib` and `xgb_models.joblib` | Fitted Logistic/Ridge and XGBoost score/cost estimators for both representations. |
| `lognormal_models.joblib`, `gamma_models.joblib` | Fitted cost heads, normalization means, LogNormal smearing factors and preprocessing metadata. |
| `lognormal_predictions_manifest.csv`, `gamma_predictions_manifest.csv` | Four cost-only prediction prefixes per family, covering both inputs and both output splits. |
| `lognormal_cv_*.csv` / `gamma_cv_*.csv`, `*_cv_oof_*.npz` | Cost-family CV summaries, folds, per-LLM errors, training rows and all candidate OOF dollar predictions. |
| `lognormal_experiment_manifest.json`, `gamma_experiment_manifest.json` | Grids, CV-selected and fitted alpha, model order, data fingerprints and run/update provenance. LogNormal records `cv_selected_alpha` and `fit_alpha`; `selected_alpha` retains its original CV meaning. |
| `lognormal_run_status.json`, `gamma_run_status.json` | Progress of each additional cost family. Global `run_status.json` completes after the final combined check. |
| `calibration_cost_comparison_summary.csv`, `calibration_cost_comparison_per_llm.csv` | Common calibration comparison of five cost families and the training-mean baseline. |
| `lognormal_calibration_*.csv`, `gamma_calibration_*.csv`, `*_residual_*.png` / `*_residual_length_*.csv` | Additional cost metrics and residual/length diagnostics. |
| `train_rows.csv`, `calibration_rows.csv`, `validation_rows.csv` | Row identifiers preserving split order. |
| `raw_score_value_counts.csv` and `raw_score_anchor_coverage.csv` | Training/calibration score-support counts and five-anchor coverage. |

With the current grids, a complete run exports **216 prediction files**: 208 core files (four types for 12 main bundles and 40 single-k bundles), plus **eight LogNormal/Gamma dollar-cost files**. Main kNN bundles combine the selected score/cost counts; each single-k bundle uses one k for both. All-k predictions are exported without an all-k calibration metric sweep.

The main manifests cover six score/cost configurations; the two additional manifests identify their entries as `cost_only`. The training-frequency baseline appears in calibration metric tables; it is not a separate exported prediction bundle. Test results are reserved for the later complete routing evaluation.

For routing-policy development, filter the manifests to `split == "validation"`. Load a score bundle from `predictions_manifest.csv` and its default cost, or pair it with a LogNormal/Gamma cost frame. Preserve question indices, LLM columns and class order; select policy settings again when changing the cost predictor. Saved predictions support these experiments without retraining.

LogNormal exports `pred_12_calibration_dollar_cost.pkl` through `pred_15_validation_dollar_cost.pkl`; Gamma uses `pred_16_calibration_dollar_cost.pkl` through `pred_19_validation_dollar_cost.pkl`. Consult their manifests for the exact input/split mapping.

Model files require the matching Step 1 inputs for new-text inference: the TF-IDF vectorizer or E5 encoder, auxiliary-feature scalers and compatible packages. Rebuild the kNN index from the prepared training features and mapped score/recorded-cost labels; the index is not serialized as a fitted estimator. Keep these prepared inputs with the project.

## If Files Cannot Be Found

Check the following before changing model code:

1. Both Step 1 notebooks completed successfully, including the E5 metadata export.
2. TF-IDF and E5 outputs are under the same `data/processed/` directory.
3. The kernel working directory is inside the downloaded project, or `DATA_PREPARATION_DIR` points to the folder containing `data/`.
4. The selected kernel contains the required packages.

Correcting a path does not require regenerating existing compatible embeddings. After adjusting the configuration, restart the kernel and run the notebook from the beginning.
