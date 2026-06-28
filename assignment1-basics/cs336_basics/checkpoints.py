
import os
import json
from pathlib import Path

from cs336_basics.paths import PROJECT_ROOT, DATA_PATH, OUT_PATH
from cs336_basics.nn_utils import save_checkpoint, load_checkpoint
from cs336_basics.configs import get_default_config, ConfigParams

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints"

def checkpoint_hyperparams(config_params, tokenfile):
    # Build a JSON-serializable dict containing only primitive fields
    params_dict = {
        "batch_size": config_params.batch_size,
        "context_length": config_params.context_length,
        "vocab_size": config_params.vocab_size,
        "d_model": config_params.d_model,
        "d_ff": config_params.d_ff,
        "num_layers": config_params.num_layers,
        "num_heads": config_params.num_heads,
        "rope_theta": config_params.rope_theta,
        "epochs": config_params.epochs,
        "checkpoint": {
            "dir": str(config_params.checkpoint.dir),
            "interval": config_params.checkpoint.interval,
            "max_keep": config_params.checkpoint.max_keep,
        },
        "tokenfile": tokenfile,
    }

    checkpt_params_path = os.path.join(
                            config_params.checkpoint.dir,
                            "hyperparams.json")

    with open(checkpt_params_path, "w") as fh:
        json.dump(params_dict, fh, indent=4)
    relative = Path(checkpt_params_path).relative_to(Path(config_params.checkpoint.dir))
    print(f"=== Hyper parameters written {relative}===")


def load_hyperparams(checkpt_params_path):

    # Read the JSON file back into a dictionary
    with open(checkpt_params_path, "r") as fh:
        params = json.load(fh)

    tokenfile = params.pop("tokenfile", None)
    checkpoint_info = params.pop("checkpoint", None)

    # Map keys to ConfigParams constructor
    config_params = ConfigParams(
        params.get("batch_size"),
        params.get("context_length"),
        params.get("vocab_size"),
        params.get("d_model"),
        params.get("d_ff"),
        params.get("num_layers"),
        params.get("num_heads"),
        params.get("rope_theta"),
        params.get("epochs", 3),
    )

    # Restore checkpoint metadata if available
    chkpt = config_params.checkpoint
    if checkpoint_info:
        try:
            config_params.checkpoint.dir = checkpoint_info.get(
                                "dir", config_params.checkpoint.dir)
            config_params.checkpoint.interval = checkpoint_info.get(
                        "interval", config_params.checkpoint.interval)
            config_params.checkpoint.max_keep = checkpoint_info.get(
                    "max_keep", config_params.checkpoint.max_keep)
        except Exception:
            # If something unexpected is present, ignore and return defaults
            pass

    return config_params, tokenfile

def checkpoint_sync(model, optimizer, global_step, chkpt):
    checkpoint_path = os.path.join(chkpt.dir, f"checkpoint_step_{global_step}.pt")

    save_checkpoint(model, optimizer, global_step, checkpoint_path)

    max_chkpts     = chkpt.max_keep

    print(f"\n[CHECKPOINT SYNC] iteration {global_step} ({checkpoint_path})")

    # Manage rotating history to prevent storage exhaustion
    chkpt.saved_paths.append(checkpoint_path)
    if len(chkpt.saved_paths) > max_chkpts:
        oldest_checkpoint = chkpt.saved_paths.pop(0)
        if os.path.exists(oldest_checkpoint):
            os.remove(oldest_checkpoint)
            print(f"[CLEANUP] Deleted old checkpoint: {oldest_checkpoint}")

def checkpoint_resume(model, optimizer, chkpt):
    checkpoint_dir = Path("./checkpoints")
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory {chkpt.dir} does not exist.")
        return 0

    # Filter for files only, then pick the one with the maximum modification time
    files = [f for f in checkpoint_dir.iterdir()
                if f.is_file() and f.name.endswith(".pt")]
    if not files:
        print(f"No checkpoints found in {chkpt.dir}.")
        return 0

    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    global_step = load_checkpoint(latest_file, model, optimizer)

    print(f"\n[CHECKPOINT-RESUME] training iteration {global_step} ({latest_file})")

    # Manage rotating history to prevent storage exhaustion
    chkpt.saved_paths.append(str(latest_file))
    return global_step

