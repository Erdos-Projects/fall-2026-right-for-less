# macOS + VS Code setup

Use Python 3.12. All project packages, including XGBoost and Seaborn, are listed in **requirements.txt**. This is the only installation file.

## 1. Create and activate the environment

Install Miniforge if Conda is not available. In VS Code, open the full project folder and a new terminal. Run these commands from the project root:

```bash
conda create -n llm-router python=3.12 pip
conda activate llm-router
```

If the environment already exists, run only the activation command.

## 2. Install and check

Install the OpenMP runtime required by XGBoost on macOS, then install the Python packages and check the environment:

```bash
conda install -n llm-router -c conda-forge llvm-openmp
python -m pip install -r 0_environment/requirements.txt
python 0_environment/check.py
```

Wait for the installation to finish successfully before running the check. Expected result: **Environment checks passed.** The checker verifies Python, the listed package versions, imports, and dependency conflicts. It does not train models or change artifacts.
