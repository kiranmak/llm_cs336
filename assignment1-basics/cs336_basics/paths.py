import pathlib
import sys
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
OUT_PATH = PROJECT_ROOT / "out"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints"
EXP_PATH = PROJECT_ROOT / "logs"

# Ensure the output directory exists
OUT_PATH.mkdir(parents=True, exist_ok=True)
EXP_PATH.mkdir(parents=True, exist_ok=True)

def set_device(device:str):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device

