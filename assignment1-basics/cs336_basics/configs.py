import os
import json
from dataclasses import dataclass, asdict, field
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
    exp_name: str 

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
    cosine_cycle_iters: int = 1000

    # Runtime args
    resume: bool = False
    log_interval: int = 1000
    eval_interval: int = 1000
    eval_batches: int = 20
    np_dtype: str = "uint16"
    model_dtype: str = "float32"

    # checkpoint args (computed after init)
    chkpt_dir: str = field(init=False)
    chkpt_interval: int = 1000
    chkpt_maxkeep: int = 3
    chkpt_saved_paths: list = field(default_factory=list)
    best_chkpt_file: str = field(init=False)

    def __post_init__(self):
        self.chkpt_dir = str(CHECKPOINT_PATH / self.exp_name)
        os.makedirs(self.chkpt_dir, exist_ok=True)
        self.best_chkpt_file = self.chkpt_dir + "/best_validation.pt"
        self.cosine_cycle_iters = self.max_steps

    def save(self, file_path: str):
        """Saves config parameters to a JSON file."""
        fname = "/".join([self.chkpt_dir, file_path])
        with open(fname, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, file_path: str, exp_name: str):
        """Loads a JSON file and initializes the Dataclass."""
        chkpt_dir = str(CHECKPOINT_PATH / exp_name)
        fname = chkpt_dir + "/" + file_path
        with open(fname, "r") as f:
            data = json.load(f)
        # Remove fields computed by __post_init__ — they are not __init__ args
        for key in ("chkpt_dir", "best_chkpt_file", "chkpt_saved_paths"):
            data.pop(key, None)
        return cls(**data)  # Unpacks dictionary keys as keyword arguments
    def print(self):
        data_dict = asdict(self)
        items = list(data_dict.items())
        for i in range(0, 4):
            chunk = items[i:i+1]
            line = " | ".join([f"{key:>14}: {value}" for key, value in chunk])
            print(line)

        for i in range(4, len(items)-7, 4):
            chunk = items[i:i+4]
            line = ""
            #line = " | ".join([f"{key:>8}: {value}" for key, value in chunk])
            for key, value in chunk:
                substr = f"{key:>10}: {value}"
                line = line + f"{substr:<19}" + "| "
            print(line)


if __name__ == "__main__":
    # Initialize your config (overriding only what you need)
    cfg = TrainingConfig(
        input_src_file = str(DATA_PATH /"TinyStoriesV2-GPT4-exper.txt"),
        vocab_file = str(OUT_PATH /"TinyStoriesV2-GPT4-train_vocab.json"),
        merge_file = str(OUT_PATH /"TinyStoriesV2-GPT4-train_merges.txt"),
        dataset = str(OUT_PATH   / "TinyStoriesV2-GPT4-exper.bin"),
        valid_set = None,
        exp_name = "run_001",
        batch_size=32,  # custom override
        max_steps = 200,
        context_length= 32,
        eval_interval = 100,
        log_interval = 50,
        resume = False,
    )
    cfg.save("hyperparams.json")
    loaded_cfg = TrainingConfig.load("hyperparams.json", exp_name="run_001")
    loaded_cfg.print()
