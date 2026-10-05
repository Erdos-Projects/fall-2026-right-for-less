# KPIs (Key Performance Indicators)

All final metrics use the untouched test partition. 

Let there be $n$ test questions and $M$ candidate LLMs. For question $i$ and model $m$, $y_{im}$ is the recorded binary correctness outcome and $c_{im}$ is the recorded historical response cost in dollars. The router selects model $\pi(x_i)$. Define:
```math
y_i^R=y_{i,\pi(x_i)},\qquad c_i^R=c_{i,\pi(x_i)}.
```

The reference model $b$ is selected by highest **policy-validation accuracy**, then
frozen. 

## Primary outcomes

### Routed accuracy - higher is better

```math
\mathrm{Acc}_R=\frac{1}{n}\sum_{i=1}^n y_i^R.
```

### Accuracy change - higher is better

```math
\Delta_A=\frac{1}{n}\sum_i(y_i^R-y_{ib}).
```

### Mean response cost - lower is better

```math
\bar C_R=\frac{1}{n}\sum_i c_i^R.
```

### Cost savings - higher is better, subject to acceptable accuracy

```math
\mathrm{Savings}=1-\frac{\bar C_R}{\bar C_b}.
```

Multiply by 100 for percent. Negative values mean the router costs more. 

## Predictor diagnostics: MSE and other errors

### Correctness-probability MSE / Brier score - lower is better

```math
\mathrm{Brier}_m=\frac1n\sum_i(\hat p_m(x_i)-y_{im})^2.
```

### Binary log loss - lower is better

```math
-\frac1n\sum_i\{y_{im}\log\hat p_m(x_i)+(1-y_{im})\log[1-\hat p_m(x_i)]\}.
```

### Cost MSE, MAE and Gamma loss - lower is better

```math
\mathrm{MSE}_{C,m}=\frac1n\sum_i(\hat c_m(x_i)-c_{im})^2,
```

```math
\mathrm{MAE}_{C,m}=\frac1n\sum_i|\hat c_m(x_i)-c_{im}|,
```

```math
\mathrm{GammaLoss}_m=\frac1n\sum_i\left\{\frac{c_{im}}{\hat c_m(x_i)}-\log\frac{c_{im}}{\hat c_m(x_i)}-1\right\}.
```


## Secondary metrics

### Precision and recall 

??

### Latency - lower is better, subject to acceptable quality

```math
T_{\mathrm{total}}=T_{\mathrm{features}}+T_{\mathrm{predictors}}+
T_{\mathrm{decision}}+T_{\mathrm{LLM}}.
```

## Tradeoff and uncertainty reporting

??