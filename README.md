# The Right answer, for Less (LLM routing)

Different language models (LLMs) offer different tradeoffs between performance and cost, so choosing the right model for each problem can significantly improve efficiency. Given a problem set, instead of always using the LLM with the best performance with a high cost, we want to devise a router that will automatically recommend the least expensive LLM while still maintaining a high accuracy on each problem. We hope this router can benefit both Companies and individuals.

Our stakeholders will be any company or individual that needs to use AI to manage a big number of tasks or problems of varied difficulties. Using an efficient router, they can save a lot of cost while still achieving a satisfactory efficiency and accuracy.

The primary question is: **Can prompt-specific routing reduce average cost while
keeping accuracy within a prespecified tolerance of the strongest fixed model?**

## Repository structure

```text
.
├── 1_data_preparation/
│   ├── step1_TFIDF_complete.ipynb
│   ├── step1_e5_complete.ipynb
│   ├── loading_template.py
│   ├── README-e5.md
|   └── Exploratory_Data_Analysis.ipynb
├── 2_Router_Training/
│   ├── TF-IDF_E5_Score_Classification_Router.ipynb
├── 3_policy_evaluation/
│   ├── DecisionPolicy.py
│   ├── loading_utils.py
│   ├── viz_utils.py
│   └── viz_notebook.ipynb
├── 4_Final_Test/
│   ├── Final_Test_kNN_E5_Frozen_Policy_KPIs_TieFix.ipynb
│   └── performance.md
└── Key_Performance_Indicators.md
```

Prepared data and fitted experiment outputs are not included in this repository. The preparation notebooks download the data files; the training notebook generates predictors, predictions and run metadata.

## Data and candidate models

This project studies prompt-specific LLM routing using recorded outcomes from [RouterBench](https://huggingface.co/datasets/withmartian/routerbench). The preparation workflow uses `withmartian/routerbench`'s `routerbench_0shot.pkl`. For each prompt, it extracts the task label, recorded score and historical dollar cost for every candidate model.

| Candidate model |
|---|
| `WizardLM/WizardLM-13B-V1.2` |
| `claude-instant-v1` |
| `claude-v1` |
| `claude-v2` |
| `gpt-3.5-turbo-1106` |
| `gpt-4-1106-preview` |
| `meta/code-llama-instruct-34b-chat` |
| `meta/llama-2-70b-chat` |
| `mistralai/mistral-7b-chat` |
| `mistralai/mixtral-8x7b-chat` |
| `zero-one-ai/Yi-34B-Chat` |

These names identify historical benchmark configurations. The recorded costs do not establish current API prices or savings.

### Splits

Prompt lists are parsed into text. Exact duplicate texts receive the same group ID, and groups are shuffled before splitting. The approximate group proportions are 60% training, 10% calibration, 10% validation and 20% test.

| Split | Recorded rows | Role in the current workflow |
|---|---:|---|
| Training | 21,900 | Fit predictors; select hyperparameters through grouped cross-validation |
| Calibration | 3,648 | Compare predictor quality and inspect errors |
| Validation | 3,649 | Select and analyze routing policies |
| Test | 7,300 | Evaluate the recorded frozen policy |

Exact duplicate groups remain together in training cross-validation. 

### KPIs and Exploratory data analysis

See [Key performance indicators](Key_Performance_Indicators.md) for metric definitions. The [Exploratory Data Analysis notebook](Exploratory_Data_Analysis.ipynb) uses only prompts belonging to the saved training split. It examines task imbalance, fractional scores, historical cost distributions, prompt-length relationships, and duplicate prompts, including comparisons within tasks. It also compares the 11 models and calculates a hindsight cost-saving benchmark using known outcomes. These analyses provide context for feature and modeling choices.


## Results

See [performance.md](4_Final_Test/performance.md) for the current test results, model-selection breakdown, cost-prediction diagnostics and assessment against the [Key performance indicators](Key_Performance_Indicators.md).

## References

- Hu et al. (2024), [RouterBench: A Benchmark for Multi-LLM Routing System](https://arxiv.org/abs/2403.12031).
- [RouterBench dataset](https://huggingface.co/datasets/withmartian/routerbench).
- [Multilingual E5-small model and usage](https://huggingface.co/intfloat/multilingual-e5-small).
- [Predictor implementation and recorded results](2_Router_Training/TF-IDF_E5_Score_Classification_Router.ipynb).

## Details about features, predictors, and routing policy

### Prompt representations

| Representation | Features used by the training notebook |
|---|---|
| Character TF-IDF | Case-sensitive character 2–4-grams; `min_df=3`; at most 50,000 features. Preparation saves an extra length column, but training removes it. |
| E5 + features | Frozen `intfloat/multilingual-e5-small` embeddings: 384 dimensions, plus character length, answer-choice count, numeric-value count and a code indicator, for 388 dimensions total. |

E5 inputs use the `query: ` prefix. Long prompts are split into non-overlapping chunks of 480 content tokens; normalized chunk vectors are averaged with equal weight and normalized again. Length and count features use training-fitted standardization; the code indicator is binary. E5 is used as a frozen encoder, not fine-tuned.

### Score prediction

Each candidate LLM has a predicted distribution over five score anchors:

```text
0, 0.25, 0.50, 0.75, 1.00
```

Original grades are mapped to the nearest anchor for classification; exact midpoint ties use the lower anchor. The anchors cover 99.858% of the recorded training score entries. All examples remain in the experiment.

The same distribution produces two numerical outputs:

- **Expected score:** the probability-weighted average of the anchors.
- **Hard score:** the anchor with the highest predicted probability.

Class accuracy and negative log-likelihood (NLL) use mapped labels. Numerical score MSE and MAE use the original recorded grades. An expected grade is not automatically a calibrated probability of binary correctness.

### Models and selection

| Model | Score prediction | Cost prediction | Selection/settings |
|---|---|---|---|
| L2 logistic + Ridge | One regularized classifier per candidate | Ridge on mean-normalized costs, restored to dollars | Grouped training CV selects logistic C by NLL and Ridge alpha by mean per-model WAPE |
| kNN | Uniform class frequencies among cosine neighbors | Mean historical cost among cosine neighbors | Grouped training CV selects score k by class accuracy and cost k separately by WAPE |
| XGBoost | One boosted-tree classifier per candidate | One boosted-tree regressor per candidate | Fixed settings: 100 rounds, depth 3, learning rate 0.05 in full mode |
| LogNormal | Uses an existing score head | Ridge on log cost, exponentiated with training-only smearing | Grouped CV selects alpha; E5 final fitting explicitly uses alpha 0 instead of its recorded CV winner 0.0001 |
| Gamma | Uses an existing score head | Gamma regression with a log link | Grouped CV selects alpha by mean per-model WAPE |


### Routing policy

For a prompt x, candidate m has predicted score ŝₘ(x) and predicted dollar cost ĉₘ(x). The implemented policy is:

```math
\pi_\lambda(x)=\arg\max_m\left[\hat{s}_m(x)-\lambda\hat{c}_m(x)\right],\qquad\lambda\ge0.
```

At λ = 0, the policy selects the highest predicted score. Larger λ penalizes expensive models more strongly; λ is measured in score per dollar. Exact utility ties follow candidate-column order. `DecisionPolicy` evaluates individual λ values or a grid. Its threshold-selection method returns the cheapest **observed validation policy** whose mean recorded score reaches the chosen threshold, or `None` if none qualifies. 

## Running the project

### 1. Create an environment

Follow the [E5 Python environment setup guide](1_data_preparation/README-e5.md) once. It creates `.venv-e5-clean` with Python 3.12 on macOS, installs the preparation/embedding dependencies, and verifies the imports. 

### 2. Prepare the data

Run these notebooks in order:

1. [TF-IDF preparation](1_data_preparation/step1_TFIDF_complete.ipynb).
2. [E5 preparation](1_data_preparation/step1_e5_complete.ipynb).

Both preparation kernels must run from `1_data_preparation/`. Check their working directory with `Path.cwd()`. This produces the canonical layout expected by the policy/final-evaluation loaders:

```text
1_data_preparation/data/
├── raw/routerbench_0shot.pkl
└── processed/
    ├── char_tfidf_50k/
    └── multilingual_e5_small_chunk480/
```

The first notebook downloads RouterBench if the raw file is missing. The second downloads the E5 encoder and generates features; its default setting recomputes embeddings. Initial preparation needs network access. Both workflows use CPU by default.

### 3. Train and compare predictors

Open [the training notebook](2_Router_Training/TF-IDF_E5_Score_Classification_Router.ipynb). The notebook searches for prepared data automatically. Full outputs are written to:

```text
2_Router_Training/experiment_outputs/score_classification_5class/full/
```

| Output | Contents |
|---|---|
| `experiment_manifest.json` | Settings, model order, versions, selected hyperparameters and run metadata |
| `predictions_manifest.csv` | Main score/cost prediction bundles for calibration and validation |
| `knn_predictions_manifest.csv` | Single-k prediction bundles for every tested neighbor count |
| `*_probability.npy`, `*_expected_score.pkl`, `*_hard_score.pkl`, `*_dollar_cost.pkl` | Aligned prediction arrays/tables |
| `calibration_*summary.csv`, `calibration_*per_llm.csv` | Aggregate and per-model evaluation metrics |
| `cost_ridge_cv/`, `knn_cv/`, `linear_cv_*.csv` | Training cross-validation records |
| `lognormal_*`, `gamma_*` | Additional cost models, diagnostics, prediction manifests and provenance |


### 4. Analyzing routing on validation and evaluating a frozen policy

Run [the policy notebook](3_policy_evaluation/viz_notebook.ipynb) with its kernel working directory set to `3_policy_evaluation/`. Run [the final-evaluation notebook](4_Final_Test/Final_Test_kNN_E5_Frozen_Policy_KPIs_TieFix.ipynb). It loads the test outcomes and calculates the scorecard. Its exported files go under `4_Final_Test/experiment_outputs/`.


