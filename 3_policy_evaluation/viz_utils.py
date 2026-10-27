import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
    Helper functions for visualizing the prediction results.
"""

# ===========================================================
# Helper functions for visualizing the prediction data
# ===========================================================

def get_bin_counts(score_data, normal=True):
    binned = get_binned_data(score_data)

    # count the occurrences of each bin
    counts = binned.apply(pd.Series.value_counts).T.fillna(0)

    if normal:
        counts = normalize_counts(counts)

    return counts

def get_binned_data(score_data):
    # find the nearest scorebin for each score
    SCORE_BINS = np.array([0,0.25,0.5,0.75,1])
    return score_data.map(lambda s: SCORE_BINS @ (np.abs(SCORE_BINS - s) < 0.125))

def normalize_counts(data):
    return data.div(data.sum(axis=1), axis=0)

def heatmap_comparison(pred, truth):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5,5))
    sns.heatmap(pred, 
                cmap='Blues', annot=True, fmt='.0%',
                vmin=0, vmax=1, cbar=False,
                ax=ax1)
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Model')
    
    sns.heatmap(truth, 
                cmap='Blues', annot=True, fmt='.0%',
                vmin=0, vmax=1, cbar=False,
                ax=ax2)
    ax2.set_xlabel('Score')
    ax2.tick_params(left=False)
    ax2.set(yticklabels=[])
    
    plt.suptitle('Distribution of predicted vs true scores', 
                 x=0.7, fontweight="bold")
    plt.tight_layout()
    
    return fig

# ===========================================================
# Plot the pairwise win rates / average score differences
# ===========================================================

def plot_pairwise_win_rates(scores, title_prefix, tie_tol=0):
    """
    scores: DataFrame (n_prompts, n_models)
    tie_tol: Scores within this distance count as ties.
    """
    models = scores.columns

    S = scores.to_numpy()

    # Shape: (n_prompts, n_models, n_models)
    diff = S[:, :, None] - S[:, None, :]

    wins = (diff > tie_tol).mean(axis=0)
    ties = (np.abs(diff) <= tie_tol).mean(axis=0)

    win_rates = wins + 0.5 * ties

    win_rates = pd.DataFrame(
        win_rates, index=models, columns=models
    )

    tie_rates = pd.DataFrame(
        ties, index=models, columns=models
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        win_rates,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        center=0.5,
        square=True,
        ax=axes[0]
    )
    axes[0].set_title(title_prefix + " pairwise win rates")

    sns.heatmap(
        tie_rates,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        ax=axes[1]
    )
    axes[1].set_title(title_prefix + " pairwise tie rates")

    for ax in axes:
        ax.set_xlabel("Model j")
        ax.set_ylabel("Model i")

    plt.tight_layout()

    return fig, win_rates, tie_rates

def plot_pairwise_differences(scores, title_prefix):
    models = scores.columns
    S = scores.to_numpy()

    diff = S[:, :, None] - S[:, None, :]

    mean_diff = diff.mean(axis=0)

    mean_diff = pd.DataFrame(
        mean_diff,
        index=models,
        columns=models
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    sns.heatmap(
        mean_diff,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True
    )

    plt.title("Mean pairwise " + title_prefix + " score differences")
    plt.xlabel("Model j")
    plt.ylabel("Model i")
    plt.tight_layout()
    plt.show()

    return fig, mean_diff