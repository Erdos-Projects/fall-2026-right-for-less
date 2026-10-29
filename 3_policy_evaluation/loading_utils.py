from pathlib import Path
import pickle
import sys

"""
    Helper functions for visualizing the prediction results.
"""

# add data prep folder to the system path to access loading_template module
sys.path.insert(0, str(Path.cwd().parent / "1_data_preparation"))
from loading_template import load_split

# ==========================================================================
MODEL_NAMES = ['logistic', 'knn', 'xgb']
COST_MODEL_NAMES = ['ridge', 'knn', 'xgb', 'lognormal', 'gamma']
DEFAULT_COST_MODELS = {'logistic': 'ridge', 'knn': 'knn', 'xgb': 'xgb'}
EMBED_NAMES = ['e5', 'tfidf']
SPLIT_NAMES = ['calibration', 'validation']
SCORE_NAMES = ['expected', 'hard']
# ==========================================================================

def load_predictions(model, embed, split, score_type, cost_model=None):
    """Load aligned score/cost predictions from the active full training run.

    `model` selects the score predictor; `cost_model` independently selects
    the cost predictor. Both use `embed`. Omitting `cost_model` preserves
    the original logistic/Ridge, kNN/kNN and XGBoost/XGBoost pairings.
    """
    score_file_name, cost_file_name = get_filenames(
        model, embed, split, score_type, cost_model,
    )

    # Use one active run for both heads; do not search archived output folders.
    run_dir = (Path(__file__).resolve().parents[1] / "2_Router_Training" /
               "experiment_outputs" / "score_classification_5class" / "full")
    with (run_dir / score_file_name).open('rb') as file:
        score_data = pickle.load(file)
    with (run_dir / cost_file_name).open('rb') as file:
        cost_data = pickle.load(file)

    if not (score_data.index.is_unique and cost_data.index.is_unique
            and score_data.index.equals(cost_data.index)
            and score_data.columns.equals(cost_data.columns)):
        raise ValueError("Score and cost predictions must have matching prompt and LLM order.")
    return score_data, cost_data


def validate_config(model, embed, split, score_type, cost_model=None):
    choices = (
        ('score model', model, MODEL_NAMES),
        ('embedding', embed, EMBED_NAMES),
        ('split', split, SPLIT_NAMES),
        ('score type', score_type, SCORE_NAMES),
    )
    for label, value, allowed in choices:
        if value not in allowed:
            raise ValueError(f"Allowed {label} options are: {allowed}; got {value!r}")
    if cost_model is not None and cost_model not in COST_MODEL_NAMES:
        raise ValueError(f"Allowed cost models are: {COST_MODEL_NAMES}; got {cost_model!r}")


def get_filenames(model, embed, split, score_type, cost_model=None):
    """Return the score and cost filenames for independently selected heads."""
    validate_config(model, embed, split, score_type, cost_model)
    if cost_model is None:
        cost_model = DEFAULT_COST_MODELS[model]

    # Prefix offsets follow the training notebook's exported prediction bundles.
    score_offsets = {'logistic': 0, 'knn': 4, 'xgb': 8}
    cost_offsets = {'ridge': 0, 'knn': 4, 'xgb': 8, 'lognormal': 12, 'gamma': 16}
    embed_offset = {'tfidf': 0, 'e5': 2}
    split_offset = {'calibration': 0, 'validation': 1}
    offset = embed_offset[embed] + split_offset[split]
    score_num = score_offsets[model] + offset
    cost_num = cost_offsets[cost_model] + offset

    return (f"pred_{score_num:02d}_{split}_{score_type}_score.pkl",
            f"pred_{cost_num:02d}_{split}_dollar_cost.pkl")
