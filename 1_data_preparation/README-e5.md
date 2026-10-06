# E5 Python environment setup

Create a separate Python environment for E5 embedding work and verify that NumPy, PyTorch, and Sentence Transformers import successfully.

These instructions are for **macOS** and use **Python 3.12**. Run all commands in a terminal, such as VS Code's integrated terminal, rather than in a notebook cell.

## 1. Open your project folder

Open the project in VS Code, then choose **Terminal → New Terminal**. Check that the terminal is in the project's main folder:

```bash
pwd
```

For this project, the folder should be `fall-2026-right-for-less`. If necessary, use `cd` to navigate to your own copy of the project.

Check that Python 3.12 is available:

```bash
python3.12 --version
```

The output should start with `Python 3.12`. If the command is not found, install Python 3.12 for your Mac from [Python's official downloads](https://www.python.org/downloads/macos/), reopen the terminal, and check again.

## 2. Create and activate the environment

```bash
python3.12 -m venv .venv-e5-clean
source .venv-e5-clean/bin/activate
```

The first command creates a folder named `.venv-e5-clean` inside the project. This folder holds the environment's Python executable and installed packages. Keep your notebooks and data outside it.

Check which Python executable is active:

```bash
python -c "import sys; print(sys.executable)"
```

The printed path should end with `.venv-e5-clean/bin/python`.

## 4. Install and test PyTorch

With the environment activated, run:

```bash
python -m pip install --upgrade pip
python -m pip install torch
```

Test PyTorch before installing the remaining dependencies:

```bash
python -X faulthandler -c "import torch; print(torch.__version__); print(torch.rand(2, 3))"
```

Successful output contains the installed PyTorch version and a tensor with two rows and three columns. The version and random numbers may differ between installations.

At this stage, you may see a warning containing `Failed to initialize NumPy: No module named 'numpy'`. If the version and tensor still print successfully, continue: NumPy is installed in the next step.

If the process crashes, stop and save the terminal output for troubleshooting. The `-X faulthandler` option can provide details about native-library crashes.

## 5. Install the remaining dependencies

```bash
python -m pip install numpy sentence-transformers ipykernel pandas pyarrow scikit-learn
python -m pip check
```

These packages support numerical arrays, embedding models, notebook execution, and the project's data preparation work.

The dependency check should report:

```text
No broken requirements found.
```

If it reports conflicts, save the output and resolve them before continuing. This check verifies declared package dependencies; it does not guarantee that every operation will run successfully.

## 6. Verify the imports together

```bash
python -X faulthandler -c "import numpy; import torch; from sentence_transformers import SentenceTransformer; print('All imports OK'); print(torch.rand(2, 3).numpy())"
```

Successful output includes:

```text
All imports OK
```

followed by a two-by-three NumPy array of random numbers. This verifies the imports and conversion of a PyTorch tensor to a NumPy array. It does not download an E5 model or run embedding generation.

## Use the environment again

When opening a new terminal, navigate to the project's main folder and activate the environment:

```bash
source .venv-e5-clean/bin/activate
```

To leave the environment:

```bash
deactivate
```

For a VS Code notebook, select the kernel at the top right, then choose **Select Another Kernel → Python Environments** and select the environment whose path contains `.venv-e5-clean/bin/python`. Restart an already-running notebook kernel after switching environments.

## References

- [PyTorch installation and verification](https://pytorch.org/get-started/locally/)
- [Sentence Transformers installation](https://sbert.net/docs/installation.html)
- [pip dependency checks](https://pip.pypa.io/en/stable/cli/pip_check/)
- [VS Code notebook kernel selection](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management)
