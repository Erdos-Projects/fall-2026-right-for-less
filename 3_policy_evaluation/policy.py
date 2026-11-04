import numpy as np
import pandas as pd


def evaluate_policy_grid(
    q_hat,
    c_hat,
    observed_scores,
    observed_costs,
    lambdas,
    tau,
):
    """Evaluate q_hat - lambda * c_hat policies on observed outcomes."""

    for table in (c_hat, observed_scores, observed_costs):
        if not table.index.equals(q_hat.index):
            raise ValueError("Prompt rows must match.")
        if not table.columns.equals(q_hat.columns):
            raise ValueError("Model columns must match.")

    quality_predictions = q_hat.to_numpy()
    cost_predictions = c_hat.to_numpy()
    scores = observed_scores.to_numpy()
    costs = observed_costs.to_numpy()

    rows = np.arange(len(q_hat))
    results = []

    for value in lambdas:
        if not np.isfinite(value) or value < 0:
            raise ValueError("Lambda must be finite and nonnegative.")

        utility = quality_predictions - value * cost_predictions
        chosen = utility.argmax(axis=1)

        mean_score = scores[rows, chosen].mean()
        mean_cost = costs[rows, chosen].mean()

        results.append({
            "lambda": float(value),
            "observed_mean_score": mean_score,
            "mean_cost_dollars": mean_cost,
            "quality_gap": mean_score - tau,
            "meets_target": bool(mean_score >= tau),
        })

    return pd.DataFrame(results)


# Calculate observed cost savings against a fixed-model baseline.
def add_cost_savings(policy_results, reference_mean_cost):
    """Add observed savings relative to a fixed-model baseline."""
    if not np.isfinite(reference_mean_cost) or reference_mean_cost <= 0:
        raise ValueError("Reference mean cost must be finite and positive.")

    results = policy_results.copy()
    results["cost_per_1000_prompts"] = 1000 * results["mean_cost_dollars"]
    results["savings_per_1000_prompts"] = (
        1000 * (reference_mean_cost - results["mean_cost_dollars"])
    )
    results["cost_savings_pct"] = (
        100 * (1 - results["mean_cost_dollars"] / reference_mean_cost)
    )
    return results


# Select the cheapest qualifying policy, breaking cost ties by lambda.
def select_best_policy(policy_results):
    """Return qualifying policies and the cheapest, breaking ties by lambda."""
    eligible = (
        policy_results.loc[policy_results["meets_target"]]
        .sort_values(["mean_cost_dollars", "lambda"])
        .copy()
    )
    best_policy = None if eligible.empty else eligible.iloc[0].copy()
    return eligible, best_policy
