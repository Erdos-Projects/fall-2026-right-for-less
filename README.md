# Erdos-Router

Different language models (LLMs) offer different tradeoffs between performance and cost, so choosing the right model for each problem can significantly improve efficiency. Given a problem set, instead of always using the LLM with the best performance with a high cost, we want to devise a router that will automatically recommend the least expensive LLM while still maintaining a high accuracy on each problem. We hope this router can benefit both Companies and individuals.

Our stakeholders will be any company or individual that needs to use AI to manage a big number of tasks or problems of varied difficulties. Using an efficient router, they can save a lot of cost while still achieving a satisfactory efficiency and accuracy.

The primary question is: **Can prompt-specific routing reduce average cost while
keeping accuracy within a prespecified tolerance of the strongest fixed model?**
This is an empirical study of established routing methods. 


# Start here

- [KPI definitions](kpis.md): Key performance indicators
- [TF-IDF Step 1 handoff](1_data_preparation/TF_IDF_ROUTER_HANDOFF.md): What the notebook prepared and how to use its saved data for router Steps 2 through 7.

## Python setup and data preparation
Install python. 

## Dataset and model pool

Source: [RouterBench on Hugging Face](https://huggingface.co/datasets/withmartian/routerbench).

The six open-weight candidates are:

1. `WizardLM/WizardLM-13B-V1.2`
2. `meta/code-llama-instruct-34b-chat`
3. `meta/llama-2-70b-chat`
4. `mistralai/mistral-7b-chat`
5. `mistralai/mixtral-8x7b-chat`
6. `zero-one-ai/Yi-34B-Chat`

These identify historical benchmark configurations. The project makes no claim
about current model versions, API prices, or currently available endpoints.

## Experimental procedure

Describe experiment

### Mathematical model
Our two main estimators will be
```math
\hat{p}_m(x_i) := \text{Estimator of the probability that the candidate model m solves the problem  } x_i \text{ correctly.}
```
and 
```math
\hat{c}_m(x) : = \text{Estimator of the expected cost of the candidate model m solving problem } x_i.
```

For $\hat p_m(x)$, we use logistic regression:

```math
\hat p_m(x)=\sigma(a_m+b_m^T\phi(x)),
```

where $\sigma(t)=(1+e^{-t})^{-1}$ is the sigmoid function.
We use the log-loss:

```math
\ell(y,p)=-y\log p-(1-y)\log(1-p).
```

The regularized empirical risk is

```math
R_m(A;a_m,b_m)
=
\frac{1}{N}\sum_{i=1}^{N}
\ell\left(y_{im},\hat p_m(x_i)\right)
+
\frac{A}{2}\|b_m\|_2^2,
```

where $N$ is the number of training prompts.

For $c_m(x)$, the expected response cost, we use Gamma
regression with a log link:

```math
\hat c_m(x)=e^{\alpha_m+\beta_m^T\phi(x)}.
```

For strictly positive observed costs, the loss function is

```math
d_\Gamma(c,\mu)
=
\frac{c}{\mu}
-
\log\left(\frac{c}{\mu}\right)
-1.
```

The regularized empirical risk is

```math
Q_m(B;\alpha_m,\beta_m)
=
\frac{1}{N}\sum_{i=1}^{N}
d_\Gamma\left(c_{im},\hat c_m(x_i)\right)
+
\frac{B}{2}\|\beta_m\|_2^2.
```

Here, $A$ and $B$ are chosen by cross-validation within
the training partition. For each fixed candidate value,
we fit the regression coefficients on the training folds
and evaluate prediction loss on the held-out fold.
The intercepts are not penalized.

For each prompt $x$, we assign every candidate model a score:

```math
S_m(x;\lambda)
=
\hat c_m(x)+\lambda\left[1-\hat p_m(x)\right].
```

The decision rule is

```math
\pi_\lambda(x)
=
\arg\min_{m\in\{1,\ldots,M\}}S_m(x;\lambda).
```

We choose $\lambda$ by constrained grid search on a
held-out policy-validation set. Among the candidate values
whose observed routing accuracy is at least $\tau$, we
select the one with the lowest observed average cost.
The accuracy target $\tau$ is fixed before training.

If no candidate value meets the target, we report that
no evaluated policy is feasible. The selected policy is
then evaluated on an untouched test set; meeting the
validation target does not guarantee the same accuracy
on new prompts. See [KPI definitions](kpis.md) for KPIs. 




## References

- Hu et al. (2024), [RouterBench](https://arxiv.org/abs/2403.12031).
- Shnitzer et al. (2023), [Large Language Model Routing with Benchmark Datasets](https://arxiv.org/abs/2309.15789). 
- Dunn and Smyth (2018), [Generalized Linear Models With Examples in R, Chapter 11](https://link.springer.com/chapter/10.1007/978-1-4419-0118-7_11).
