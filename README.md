# The Right answer, for Less (LLM routing)

Different language models (LLMs) offer different tradeoffs between performance and cost, so choosing the right model for each problem can significantly improve efficiency. Given a problem set, instead of always using the LLM with the best performance with a high cost, we want to devise a router that will automatically recommend the least expensive LLM while still maintaining a high accuracy on each problem. We hope this router can benefit both Companies and individuals.

Our stakeholders will be any company or individual that needs to use AI to manage a big number of tasks or problems of varied difficulties. Using an efficient router, they can save a lot of cost while still achieving a satisfactory efficiency and accuracy.

The primary question is: **Can prompt-specific routing reduce average cost while
keeping accuracy within a prespecified tolerance of the strongest fixed model?**

## Repository structure

```text
.
├── 0_environment/
│   ├── README_env.md
│   ├── requirements.txt
│   └── check.py
├── 1_data_preparation/
│   ├── TFIDF.ipynb
│   └── Exploratory_Data_Analysis.ipynb
├── 2a_Score_training_calibration/
│   ├── score_model_xgb.py
│   ├── score_training_xgb.ipynb
│   ├── score_model_knn.py
│   └── score_training_knn.ipynb
├── 2b_cost_training_calibration/
│   ├── cost_model_xgb.py
│   └── cost_training_xgb.ipynb
├── 3_policy_evaluation/
│   ├── policy.py
│   ├── policy_eval_xgb_xgb.ipynb
│   └── policy_eval_knn_xgb.ipynb
├── 4_Final_Test/
│   └── testing_knn_xgb.ipynb
├── Key_Performance_Indicators.md
└── README.md
```

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

## Routing policy

For each prompt $x$, predict each model’s expected score $\hat q_m(x)$ using kNN or XGBoost and expected cost $\hat c_m(x)$ in dollars using XGBoost. Route to

$$
\pi_\lambda(x)=\arg\max_m\left[\hat q_m(x)-\lambda\hat c_m(x)\right],
$$

with deterministic tie-breaking. At $\lambda=0$, routing favors the highest predicted score; increasing $\lambda$ gives cost more weight.

Evaluate a nonnegative grid of $\lambda$ values on the policy-validation partition. Select the policy with the lowest observed average cost whose observed mean score meets GPT-4’s validation mean score, breaking cost ties by the smallest $\lambda$. If none qualifies, report this. Freeze the selected policy and predictors before evaluating the held-out test set.

## Performance statement

On the held-out test set, the fixed kNN-score/XGBoost-cost router reduced average cost by 37.66% relative to always using GPT-4 (95% paired prompt-group bootstrap CI: 36.39%–38.91%). Its observed mean score was 0.780534, compared with GPT-4’s 0.779795. The estimated quality difference was +0.000740 (95% CI: −0.002973 to +0.004541).

## References

- Hu et al. (2024), [RouterBench: A Benchmark for Multi-LLM Routing System](https://arxiv.org/abs/2403.12031).
- [RouterBench dataset](https://huggingface.co/datasets/withmartian/routerbench).
- [Multilingual E5-small model and usage](https://huggingface.co/intfloat/multilingual-e5-small).

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | Follow `0_environment/README_env.md`, then run `python 0_environment/check.py`. | Creates the environment, installs dependencies, and checks the installation. |
| 2 | `1_data_preparation/TFIDF.ipynb` | Loads the data, creates partitions with duplicate prompts kept together, and saves TF–IDF features and prompt length. |
| 3 | `1_data_preparation/Exploratory_Data_Analysis.ipynb` | Explores the training data, scores, costs, and task distribution. |
| 4 | `2a_Score_training_calibration/score_training_xgb.ipynb` | Trains and saves XGBoost score predictors and inspects calibration-partition diagnostics. |
| 5 | `2b_cost_training_calibration/cost_training_xgb.ipynb` | Trains and saves XGBoost cost predictors using each LLM’s mean training cost for normalization, then inspects diagnostics. |
| 6 | `2a_Score_training_calibration/score_training_knn.ipynb` | Selects the neighbor count through grouped cross-validation, saves the kNN score predictor, and compares its calibration results with XGBoost. |
| 7 | `3_policy_evaluation/policy_eval_xgb_xgb.ipynb` | Evaluates XGBoost-score/XGBoost-cost routing policies on validation prompts and measures quality and savings. |
| 8 | `3_policy_evaluation/policy_eval_knn_xgb.ipynb` | Evaluates kNN-score/XGBoost-cost policies, selects the cheapest evaluated policy meeting the validation quality target, and saves its configuration. |
| 9 | `4_Final_Test/testing_knn_xgb.ipynb` | Evaluates the saved, fixed kNN-score/XGBoost-cost policy on held-out test prompts and reports quality, savings, and bootstrap confidence intervals. |
