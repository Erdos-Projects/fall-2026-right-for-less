import importlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys


def main():
    failures = []
    print("Python:", platform.python_version())
    print("Interpreter:", sys.executable)

    if sys.version_info[:2] != (3, 12):
        failures.append("Use Python 3.12 for this project.")

    module_names = {
        "scikit-learn": "sklearn",
        "huggingface-hub": "huggingface_hub",
    }
    requirements = Path(__file__).with_name("requirements.txt")

    for line in requirements.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        package, expected = line.split("==", 1)
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"Missing package: {package}")
            continue

        if installed != expected:
            failures.append(f"{package}: expected {expected}, found {installed}")

        try:
            importlib.import_module(module_names.get(package, package))
        except Exception as error:
            failures.append(f"Cannot import {package}: {error}")
        else:
            print(f"Imported {package}=={installed}")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        failures.append(result.stdout.strip() or result.stderr.strip())

    if failures:
        print("\nEnvironment needs attention:")
        for failure in failures:
            print(f"- {failure}")
        print("Activate the project environment, install requirements.txt, and rerun.")
        return 1

    print("\nEnvironment checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
