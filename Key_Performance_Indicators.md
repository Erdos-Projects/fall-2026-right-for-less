# Key performance indicators

The project aims to reduce response cost while keeping quality acceptably close to a strong fixed LLM. 

This document defines the metrics and how to interpret the performance. 


## Evaluation scope

- RouterBench supplies **graded scores in [0, 1]**, including fractional values, and historical response costs for all 11 candidates.
- The current frozen reference is `gpt-4-1106-preview`.

For prompt $x_i$, let $s_{im}$ be candidate $m$'s recorded grade and $c_{im}$ its historical response cost. The router selects model $\pi_\lambda(x_i)$ using predicted scores and costs:

```math
\pi_\lambda(x_i)=\operatorname*{arg\,max}_m
\left(\widehat{s}_{im}-\lambda\widehat{c}_{im}\right).
```

The selected model's recorded outcomes are:

```math
s_i^{R}=s_{i,\pi_\lambda(x_i)},\qquad
c_i^{R}=c_{i,\pi_\lambda(x_i)}.
```

For the fixed reference model $b$, use $s_i^{b}=s_{ib}$ and $c_i^{b}=c_{ib}$. A hat denotes a **prediction**; quantities without hats are recorded outcomes. All averages below use the same $n$ evaluation prompts.

## Primary policy KPIs

| ID | KPI | Definition | What counts as success? |
|---|---|---|---|
| P1 | Mean recorded score | $Q^{R}=\frac{1}{n}\sum_i s_i^{R}$; also report $Q^{b}$ | Higher is better. |
| P2 | Score difference / quality retention | $\Delta Q=\frac{1}{n}\sum_i(s_i^{R}-s_i^{b})$ | Report the point estimate and paired interval. |
| P3 | Mean historical response cost and savings | $C^{R}=\frac{1}{n}\sum_i c_i^{R}$; $S=1-C^{R}/C^{b}$ | Positive S means lower response cost. |
| P4 | Joint cost–quality objective | Cost reduction and quality above threshold | Both conditions must hold. |


## Routing behavior

| ID | Diagnostic | Calculation and interpretation | 
|---|---|---|
| R1 | Fixed-candidate comparison | Compare the router's score/cost point with every fixed candidate. A candidate dominates it if it has at least as much quality and at most as much cost, with one strict improvement. | 
| R2 | Selection shares | Number of prompts assigned to each LLM divided by n; include the reference share. | 

## Predictor diagnostics

All errors below are lower-is-better.

### D1. Graded-score and five-class prediction

For numerical predictions:

```math
\mathrm{MSE}_{S,m}=\frac1n\sum_i(\hat s_{im}-s_{im})^2,
\qquad
\mathrm{MAE}_{S,m}=\frac1n\sum_i|\hat s_{im}-s_{im}|.
```


### D2. Dollar-cost prediction

Use MSE, MAE, WAPE:

```math
\mathrm{MSE}_{C,m}=\frac1n\sum_i(\hat c_{im}-c_{im})^2,
\qquad
\mathrm{MAE}_{C,m}=\frac1n\sum_i|\hat c_{im}-c_{im}|,
```

```math
\mathrm{WAPE}_{C,m}=\frac{\sum_i|\hat c_{im}-c_{im}|}{\sum_i c_{im}}.
```

MAE is in USD, MSE in USD²; WAPE is dimensionless.


