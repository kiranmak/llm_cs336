import os
import json
from pathlib import Path

from cs336_basics.paths import PROJECT_ROOT, DATA_PATH, OUT_PATH
from cs336_basics.paths import CHECKPOINT_PATH
from cs336_basics.nn_utils import save_checkpoint, load_checkpoint

def checkpoint_sync(model, optimizer, global_step, chkpt):
    checkpoint_path = os.path.join(chkpt.dir,
                                   f"checkpoint_step_{global_step}.pt")

    save_checkpoint(model, optimizer, global_step, checkpoint_path)

    max_chkpts     = chkpt.max_keep

    relative = Path(checkpoint_path).relative_to(Path(chkpt.dir))
    print(f"\n[chkpt sync] step {global_step} ({relative})")

    # Manage rotating history to prevent storage exhaustion
    chkpt.saved_paths.append(checkpoint_path)
    if len(chkpt.saved_paths) > max_chkpts:
        oldest_checkpoint = chkpt.saved_paths.pop(0)
        if os.path.exists(oldest_checkpoint):
            os.remove(oldest_checkpoint)
            relative = Path(oldest_checkpoint).relative_to(Path(chkpt.dir))
            print(f"[chkpt-del] Deleted old checkpoint: {relative}")

def checkpoint_resume(model, optimizer, chkpt):
    if not Path(chkpt.dir).exists():
        print(f"Checkpoint directory {chkpt.dir} does not exist.")
        return 0

    # Filter for files, pick the one with the maximum modification time
    files = [f for f in chkpt.dir.iterdir()
                if f.is_file() and f.name.endswith(".pt")]
    if not files:
        print(f"No checkpoints found in {chkpt.dir}.")
        return 0

    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    print(f"\n[chkpt resume] ({latest_file})")
    global_step = load_checkpoint(latest_file, model, optimizer)

    print(f"\n[chkpt resume] step {global_step} ")

    # Manage rotating history to prevent storage exhaustion
    chkpt.saved_paths.append(str(latest_file))
    return global_step

