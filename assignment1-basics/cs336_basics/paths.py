import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
OUT_PATH = PROJECT_ROOT / "out"

# Ensure the output directory exists
OUT_PATH.mkdir(parents=True, exist_ok=True)
