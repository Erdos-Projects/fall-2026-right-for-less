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
    ax1.set_title(r'Distribution of $\underline{predicted}$ scores')
    
    sns.heatmap(truth, 
                cmap='Blues', annot=True, fmt='.0%',
                vmin=0, vmax=1, cbar=False,
                ax=ax2)
    ax2.set_xlabel('Score')
    ax2.tick_params(left=False)
    ax2.set(yticklabels=[])
    ax2.set_title(r'Distribution of $\underline{actual}$ scores')
    
    return fig