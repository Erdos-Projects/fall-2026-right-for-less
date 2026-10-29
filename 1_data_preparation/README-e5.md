# Project Python environment setup

Use one environment for data preparation, frozen E5 embeddings, exploratory analysis, router training and evaluation. This guide includes **Matplotlib, Seaborn and XGBoost** as well as the preparation dependencies.

These instructions use **macOS and Python 3.12**. Run the shell commands in a terminal, such as VS Code's integrated terminal.

## 1. Open the project folder

Open your current `test-data-1` folder in VS Code, then choose **Terminal → New Terminal**. Check that the terminal is at the project root:

```bash
pwd
python3.12 --version
```

The directory should contain `1_data_preparation/`, `2_Router_Training/`, `3_policy_evaluation/` and `4_Final_Test/`. Its name may differ in another checkout. If Python 3.12 is missing, install it from [Python's macOS downloads](https://www.python.org/downloads/macos/) and reopen the terminal.

## 2. Create or reuse the environment

**For a new environment only:**

```bash
python3.12 -m venv .venv-e5-clean
```

**For both new and existing environments:**

```bash
source .venv-e5-clean/bin/activate
python -c "import sys; print(sys.executable)"
```

The printed path should end with `.venv-e5-clean/bin/python`. If you already followed the earlier E5 guide, reuse that environment and continue below. Keep notebooks and data outside the environment folder.

## 3. Install the macOS OpenMP runtime

XGBoost's native library needs the OpenMP runtime. If the import reports `Library not loaded: @rpath/libomp.dylib`, installing the Python package alone does not resolve it. Install the runtime into that existing Miniforge installation:

```bash
"$HOME/miniforge3/bin/conda" install -n base -c conda-forge llvm-openmp
```

**Restart the notebook kernel after installation**, then retry `import xgboost`. Reinstalling the already-present XGBoost package with pip does not supply this missing native runtime.

## 4. Install the project dependencies

With `.venv-e5-clean` activated, run:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy pyarrow joblib threadpoolctl \
  "scikit-learn>=1.6" "xgboost>=2.0" \
  torch sentence-transformers huggingface_hub \
  matplotlib seaborn jinja2 ipython ipykernel
python -m pip check
```

The command installs missing dependencies and checks the stated minimum versions. Packages already satisfying the requirements can be reused. Sentence Transformers installs its required Transformers/tokenizer dependencies.

| Packages | Project use |
|---|---|
| NumPy, pandas, SciPy, PyArrow | Arrays, tables, sparse features, statistics and data storage |
| scikit-learn, XGBoost, joblib, threadpoolctl | Predictors, preprocessing, model persistence and thread control |
| PyTorch, Sentence Transformers, Hugging Face Hub | Frozen E5 features and dataset/model downloads |
| Matplotlib, Seaborn, Jinja2 | Plots and styled result tables |
| IPython, ipykernel | Notebook display and execution in VS Code |

`pip check` should print `No broken requirements found.` This checks declared dependencies; the import checks below also test native-library loading.

These are setup requirements, not a locked reproduction environment. Reproducing an existing saved run may require its recorded package versions.

## 5. Verify imports before running notebooks

First check XGBoost:

```bash
python -c "import xgboost; print('XGBoost:', xgboost.__version__)"
```

Then check the project imports together:

```bash
python -X faulthandler - <<'PY'
import numpy as np
import pandas
import scipy
import pyarrow
import sklearn
import xgboost
import joblib
import threadpoolctl
import torch
import huggingface_hub
import matplotlib.pyplot as plt
import seaborn
import jinja2
import IPython
import ipykernel
from sentence_transformers import SentenceTransformer

print("All project imports OK")
print("PyTorch to NumPy:", torch.rand(2, 3).numpy())
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
fig.canvas.draw()
plt.close(fig)
print("Matplotlib rendering OK")
PY
```

Expected output includes `All project imports OK` and `Matplotlib rendering OK`. This checks imports, PyTorch/NumPy interoperability and basic plotting; it does not download E5 weights or train models.

## 6. Select the notebook kernel

In VS Code, open a notebook and select its kernel at the top right. Choose **Select Another Kernel → Python Environments**, then the environment whose executable ends with `.venv-e5-clean/bin/python`.

**Restart the notebook kernel after installing packages or libomp**, then rerun imports. To confirm the notebook uses the intended environment, run this in a notebook cell:

```python
import sys
print(sys.executable)
```

JupyterLab is optional if you prefer a browser interface instead of VS Code:

```bash
python -m pip install jupyterlab
python -m jupyterlab
```

## Use the environment again

From the project root in a new terminal:

```bash
source .venv-e5-clean/bin/activate
```

You do not need to recreate the environment or reinstall dependencies each time. To leave it:

```bash
deactivate
```

Follow the root README for notebook execution order and working-directory requirements.

## References

- [XGBoost installation](https://xgboost.readthedocs.io/en/stable/install.html)
- [Matplotlib installation](https://matplotlib.org/stable/install/index.html)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [Sentence Transformers installation](https://sbert.net/docs/installation.html)
- [pip dependency checks](https://pip.pypa.io/en/stable/cli/pip_check/)
- [VS Code notebook kernel selection](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management)
