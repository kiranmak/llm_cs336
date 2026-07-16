import os
import json
from typing import Optional
from dataclasses import dataclass, asdict
from cs336_basics.paths import CHECKPOINT_PATH, DATA_PATH, OUT_PATH

class CheckPtConfig:
    # Other chackpointing Configuration
    def __init__(self, name):
        self.dir = CHECKPOINT_PATH / name
        os.makedirs(self.dir, exist_ok=True)
        self.interval = 5000 # Save every 5,000 steps
        self.max_keep = 3    # Keep only the 3 most recent full states
        self.saved_paths = []
        self.best_chkpt_path = self.dir / "best_validation.pt"

@dataclass
class TrainingConfig:
    # Required args
    input_src_file: str
    vocab_file: str
    merge_file: str
    dataset: str
    valid_set: str
    exp_name: str = None

    # Model architecture
    vocab_size: int = 10_000
    d_model: int = 512
    d_ff: int = 1344
    num_layers: int = 4
    num_heads: int = 16
    rope_theta: int = 10_000

    # Hyperparameters
    max_steps: int = 1000
    batch_size: int = 256
    context_length: int = 256
    lr_max: float = 1e-3
    lr_min: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_iters: int = 2000
    cosine_cycle_iters: int = 106_667

    # Runtime args
    resume: bool = False
    log_interval: int = 1000
    eval_interval: int = 1000
    eval_batches: int = 20
    np_dtype: str = "uint16"
    model_dtype: str = "float32"

    # checkpoint args
    chkpt_dir = str(CHECKPOINT_PATH) + "/" + str(exp_name)
    chkpt_interval: int = 1000
    chkpt_maxkeep: int = 3
    chkpt_saved_paths = []
    best_chkpt_file :str = "/".join([chkpt_dir, "best_validation.pt"])

    def save(self, file_path: str):
        """Saves config parameters to a JSON file."""
        fname = "/".join([self.chkpt_dir, file_path])
        with open(fname, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, file_path: str):
        """Loads a JSON file and initializes the Dataclass."""
        fname = "/".join([cls.chkpt_dir, file_path])
        with open(fname, "r") as f:
            data = json.load(f)
        return cls(**data)  # Unpacks dictionary keys as keyword arguments
    def print(self):
        for key, value in asdict(self).items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    # Initialize your config (overriding only what you need)
    cfg = TrainingConfig(
        input_src_file = str(DATA_PATH /"TinyStoriesV2-GPT4-exper.txt"),
        vocab_file = str(OUT_PATH /"TinyStoriesV2-GPT4-train_vocab.json"),
        merge_file = str(OUT_PATH /"TinyStoriesV2-GPT4-train_merges.txt"),
        dataset = str(OUT_PATH   / "TinyStoriesV2-GPT4-exper.bin"),
        valid_set = None,
        batch_size=128,  # custom override
        max_steps = 200,
        context_length= 32,
        eval_interval = 100,
        log_interval = 50,
        resume = False,
    )
    cfg.save("hyperparams.json")
    loaded_cfg = TrainingConfig.load("hyperparams.json")
    print(loaded_cfg)
