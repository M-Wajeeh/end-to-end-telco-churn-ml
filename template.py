import os

structure = [
    "data/raw",
    "data/processed",
    "data/external",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/utils",
    "app",
    "configs",
    "scripts",
    "tests",
    ".github/workflows",
    "docker",
    "great_expectations",
    "mlruns",
    "artifacts",
]

for folder in structure:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created: {folder}")
    else:
        print(f"Already exists: {folder}")