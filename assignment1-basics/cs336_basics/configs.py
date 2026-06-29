import os
from typing import Optional
from dataclasses import dataclass, field
from cs336_basics.paths import CHECKPOINT_PATH

@dataclass
class OptimizerConfig:
    lr_max: float = 1e-3
    lr_min: float = 1e-4

    warmup_iters: int = 2000
    cosine_cycle_iters: int = 106_667

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.1

    grad_clip: float = 1.0

@dataclass
class TrainingConfig:
    max_steps: int = 106_667
    log_interval: int = 100

    eval_interval: int = 2000
    eval_batches: int = 20

    seed: int = 42
    np_dtype: str = "uint16"
    model_dtype: str = "float32"


@dataclass
class TrainConfig:
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

def get_preset_cfg() -> TrainConfig:
    """
    Return a default training configuration.
    """
    cfg = TrainConfig()
    return cfg

class CheckPtConfig:
    # Other chackpointing Configuration
    def __init__(self):
        self.dir = CHECKPOINT_PATH
        os.makedirs(self.dir, exist_ok=True)
        self.interval = 5000 # Save every 5,000 steps
        self.max_keep = 3    # Keep only the 3 most recent full states
        self.saved_paths = []

class ConfigParams:
    def __init__(self, batch_size, context_length, vocab_size, d_model,
                       d_ff, num_layers, num_heads, theta, resume=False):
        self.batch_size     = batch_size
        self.context_length = context_length
        self.vocab_size     = vocab_size
        self.d_model        = d_model
        self.d_ff           = d_ff
        self.num_layers     = num_layers
        self.num_heads      = num_heads
        self.rope_theta     = theta
        self.resume         = resume
        self.checkpoint     = CheckPtConfig()

    def show(self):
        print("     batch_size     =", self.batch_size)
        print("     context_length =", self.context_length)
        print("     vocab_size     =", self.vocab_size)
        print("     d_model        =", self.d_model)
        print("     d_ff           =", self.d_ff)
        print("     num_layers     =", self.num_layers)
        print("     num_heads      =", self.num_heads)
        print("     rope_theta     =", self.rope_theta)
        print("     resume         =", self.resume)
        print("     Checkpoints:");
        print("         dir      =", self.checkpoint.dir)
        print("         interval =", self.checkpoint.interval)
        print("         max_save =", self.checkpoint.max_keep)
        print("         interval =", self.checkpoint.saved_paths)


def get_default_config() -> ConfigParams:

    cfg = ConfigParams(batch_size=256,
                       context_length=256,
                       vocab_size=10000,
                       d_model=512,
                       d_ff=1344,
                       num_layers=4,
                       num_heads=16,
                       rope_theta=10000,
                       resume=False,
                       checkpoint=CheckPtConfig()
                       )
    return cfg
