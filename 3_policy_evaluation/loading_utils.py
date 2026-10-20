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
EMBED_NAMES = ['e5', 'tfidf']
SPLIT_NAMES = ['calibration', 'validation']
SCORE_NAMES = ['expected', 'hard']
# ==========================================================================

def load_predictions(model, embed, split, score_type):
    """
    returns: score_data, cost_data
    """
    # first check the configuration is valid
    validate_config(model, embed, split, score_type)

    # next reconstruct the file name
    score_file_name, cost_file_name  = get_filenames(model, 
                                                     embed, 
                                                     split,
                                                     score_type)
    
    # search recursively for the file name in the parent directory
    root_dir = Path.cwd().parent
    score_files = list(root_dir.rglob("*/full/" + score_file_name))
    cost_files = list(root_dir.rglob("*/full/" + cost_file_name))
    score_data = None
    cost_data = None

    if score_files:
        file_path = score_files[0] # Open the first matching file
        with open(file_path, 'rb') as file:
            score_data = pickle.load(file)
            
    if cost_files:
        file_path = cost_files[0]
        with open(file_path, 'rb') as file:
            cost_data = pickle.load(file)
    
    return score_data, cost_data

def validate_config(model, embed, split, score_type):
    assert model in MODEL_NAMES, f"Allowed models are: {MODEL_NAMES}"
    assert embed in EMBED_NAMES, f"Allowed embeddings are: {EMBED_NAMES}"
    assert split in SPLIT_NAMES, f"Allowed split types are: {SPLIT_NAMES}"
    assert score_type in SCORE_NAMES, f"Allowed score types are: {SCORE_NAMES}"

def get_filenames(model, embed, split, score_type):
    """
    output the file names for the selected configuration.

    returns: score_file_name, cost_file_name
    """
    models = {
        "logistic": 0,
        "knn": 4,
        "xgb": 8
    }

    embed_offset = {
        "tfidf": 0,
        "e5": 2
    }

    split_offset = {
        "calibration": 0,
        "validation": 1
    }

    num = models[model] + embed_offset[embed] + split_offset[split]

    return (f"pred_{num:02d}_{split}_{score_type}_score.pkl",
            f"pred_{num:02d}_{split}_dollar_cost.pkl")