# Understanding the final test results

On **7,300 test problems**, the router saves **37.76% of recorded response cost** while achieving almost the same observed quality as always using GPT-4.

| Metric | Always GPT-4 | Router |
|---|---:|---:|
| Mean score | 0.779795 | 0.780671 |
| Cost per 1,000 problems | $3.28 | $2.04 |

**Where the savings come from.** The router keeps GPT-4 for 69.60% of problems and assigns the remaining 30.40% to other models. Those reassigned problems account for **62.64% of the original GPT-4 bill**, so replacing them has a large effect on total cost.

| Task | Share of total dollars saved |
|---|---:|
| Grade-school math | 73.47% |
| HellaSwag | 14.17% |
| All other tasks | 12.36% |

Grade-school math drives most of the savings: the router sends 1,454 of its 1,483 problems to other models, mainly Claude v2 and Claude Instant. Across all tasks, **Claude Instant, Claude v2, and Yi 34B generate 88.71% of the savings**. The overall saving therefore depends strongly on the task mix.

**Why the router's score is slightly higher.** GPT-4 has the highest overall score among the fixed models, but other models sometimes score better on individual problems. On the 2,219 problems assigned away from GPT-4, the selected alternatives score:

- **Higher:** 389 problems.
- **The same:** 1,505 problems.
- **Lower:** 325 problems.

After accounting for fractional grades, gains exceed losses by **6.4 total grade points**. Dividing by 7,300 gives the small mean-score increase of **0.000877**, or about **0.088 percentage points**. Improvements on `consensus_summary` and HellaSwag slightly outweigh losses on math and other tasks.

**How to interpret the result.** The 95% confidence interval for cost savings is **36.47%–39.01%**. The interval for the score difference is **−0.288 to +0.467 percentage points**, which includes zero. The results support substantial cost savings with similar observed quality; a reliable quality improvement has not been established.

Source: [final-test notebook](testing_knn_xgb.ipynb), using the policy fixed on validation. Costs are historical benchmark response costs and exclude router overhead.
