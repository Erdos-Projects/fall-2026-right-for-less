# Performance of the frozen kNN-score / XGBoost-cost router

Summary: **On 7,300 historical test prompts, the validation-selected kNN-score/XGBoost-cost TF-IDF router reduced response cost by 37.97% relative to GPT-4, with a mean-grade difference of +0.000911.**

## Evidence and configuration

This report follows [Key performance indicators](../Key_Performance_Indicators.md). It uses the saved outputs in [the final-evaluation notebook](Final_Test_kNN_E5_Frozen_Policy_KPIs_TieFix.ipynb) and its [exported scorecard](experiment_outputs/knn_xgb_tfidf_test_20261029T150954_190306Z/test_kpi_scorecard.csv). 

| Setting | Recorded value |
|---|---|
| Evaluation rows | 7,300 test prompts |
| Candidate pool | All 11 historical LLMs, including GPT-4 |
| Router representation | Character TF-IDF vocabulary columns, float32 sparse matrix; the preparation length column is excluded |
| Score estimator | Uniform cosine kNN, k = 100 |
| Cost estimator | Saved XGBoost regressors, one per candidate; predictions restored to dollar units |
| Score output | Expected grade over five score anchors |
| Policy | `argmax(predicted_score - lambda * predicted_dollar_cost)` |
| λ | 2.7142066911362983 |
| Fixed reference | `gpt-4-1106-preview` |
| Validation score / cost bundles | `pred_05_validation` / `pred_09_validation` |
| Source training run | `20261029T023425_449901Z` |
| Notebook-linked export | `knn_xgb_tfidf_test_20261029T150954_190306Z` |

Notebook 3 selected the configuration and λ by minimizing observed validation response cost while meeting GPT-4's validation mean score of 0.787859688. That validation threshold is distinct from GPT-4's observed test mean. 

## Which KPIs are satisfied?

| KPI | Recorded evidence |
|---|---|
| P1: Mean recorded score | Router 0.780705; reference 0.779795 |
| P2: Quality retention | Score difference +0.000911; 95% interval [−0.002809, +0.004651] | 
| P3: Lower historical response cost | 37.9703% savings; 95% interval [36.6687%, 39.2549%] |
| P4: Joint cost–quality objective | Lower observed cost and slightly higher observed grade |
| R1: Fixed-candidate comparison | No fixed candidate has at least the router's mean grade and at most its cost | 
| R2: Selection shares | GPT-4 handles 69.4932%; all 11 candidates receive prompts | 
| D1: Final graded-score/classification errors | No final graded-score MSE/MAE or five-class NLL table |
| D2: Cost prediction errors | Candidate-level and routed-selection errors are reported below |

## Quality and historical cost

Source: final notebook Sections 4–5 and the exported scorecard.

| Outcome | Router | Fixed GPT-4 reference |
|---|---:|---:|
| Mean recorded grade (0–1) | 0.780705 | 0.779795 |
| Mean historical response cost (USD/prompt) | $0.00203463 | $0.00328008 |

The positive score difference is an observed mean-grade improvement. 

## What the router selects

Source: [per-prompt selections](experiment_outputs/knn_xgb_tfidf_test_20261029T150954_190306Z/selected_models_test.csv) and final notebook Section 3.

| Selected LLM | Test prompts |
|---|---:|
| `gpt-4-1106-preview` | 5,073 |
| `claude-v2` | 898 |
| `claude-instant-v1` | 561 |
| `zero-one-ai/Yi-34B-Chat` | 550 |
| `gpt-3.5-turbo-1106` | 86 |
| `mistralai/mistral-7b-chat` | 42 |
| `mistralai/mixtral-8x7b-chat` | 32 |
| `meta/code-llama-instruct-34b-chat` | 29 |
| `claude-v1` | 25 |
| `WizardLM/WizardLM-13B-V1.2` | 3 |
| `meta/llama-2-70b-chat` | 1 |
| **Total** | **7,300** |

GPT-4 handles **69.49%** of prompts; **30.51%** go to other candidates. These shares describe reference reliance. They do not establish that diverted prompts are easier or that every candidate improves routing.

The router has a slightly higher observed grade and lower cost than fixed GPT-4. It also has higher grade and lower cost than fixed Claude-v1 and Claude-v2. 
## Cost-prediction quality

Source: [cost diagnostics](experiment_outputs/knn_xgb_tfidf_test_20261029T150954_190306Z/test_cost_diagnostics_per_llm.csv). These evaluate the **XGBoost cost predictions** used by the router.

| Metric | Equal-weight mean across 11 candidates | Routed selections |
|---|---:|---:|
| MSE (USD²) | 2.341222e-07 | 8.606794e-07 |
| MAE (USD) | 1.165786e-04 | 3.034419e-04 |
| WAPE | 12.3664% | 14.9139% |
| Gamma loss (not doubled deviance) | 0.01545156 | 0.02102245 |

Routed WAPE of **14.91%** means total absolute predicted-cost error is that fraction of total recorded routed cost. All four routed errors exceed the equal-candidate averages here. These aggregates cover different populations, with routing concentrated on GPT-4. 

## What is not evaluated in this project

The following areas are outside the current KPIs and assessment.

| Area | What has not been evaluated |
|---|---|
| Engineering: latency and throughput | End-to-end response time, p50/p95 latency, concurrent request capacity and sustained throughput. Notebook execution times do not establish serving performance; router overhead and LLM response time would need separate measurements. |
| Engineering: resource use and total serving cost | Peak memory, CPU/GPU utilization, feature-generation cost and operating cost per request. Reported savings include historical model response costs only. |
| Engineering: service reliability | Request success rates, timeouts, retries, fallback behavior, rate-limit handling and recovery from model-provider failures. No production service-level targets are assessed. |
| Live user outcomes and online learning | User satisfaction, causal effects from live A/B tests, and learning from feedback available only for the selected model. The benchmark records outcomes for every candidate, so it does not test exploration or contextual-bandit behavior. |
| Generalization to changed workloads or model pools | Performance on genuinely new tasks, future prompt distributions, updated models or current provider prices. Results for the fixed historical candidate pool do not establish adaptation to these changes. |
| Final-policy subgroup guarantees | Predefined quality-retention requirements within individual tasks or prompt-length bands. Training EDA and validation diagnostics explore such differences, but they do not establish subgroup guarantees for the final router. |
| Per-prompt quality guarantees and response suitability | Guaranteed quality for each request, calibrated probabilities of binary correctness, or independent human assessment of usefulness and response safety. Mean benchmark grades and expected five-anchor scores do not establish these properties. |

