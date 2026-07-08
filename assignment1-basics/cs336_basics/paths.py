from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REL_ROOT = PROJECT_ROOT.relative_to(Path.cwd())
DATA_PATH = REL_ROOT / "data"
OUT_PATH = REL_ROOT / "out"
CHECKPOINT_PATH = REL_ROOT / "checkpoints"
EXP_PATH = REL_ROOT / "logs"

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

def get_amptype():
    if torch.cuda.is_available():
        amp_dtype = torch.bfloat16  # Or torch.float16 if older
        device_type = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        amp_dtype = torch.float16   # MPS loves float16, lacks good bf16 support
        device_type = "mps"
    else:
        amp_dtype = torch.bfloat16  # CPU only supports bfloat16 for autocast
        device_type = "cpu"
    return amp_dtype, device_type


