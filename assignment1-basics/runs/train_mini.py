import os
import time
import math
import torch
import json
from torch import nn
import numpy as np
from tqdm import tqdm
from pathlib import Path

from runs.train import (
    open_memmap_1d,
    torch_dtype_from_string,
    training_together)
from cs336_basics.experiments_tracker import ExperimentTracker
from cs336_basics.paths import OUT_PATH, EXP_PATH, set_device
from cs336_basics.tokenizer_exp import file_encode_bin_from_vocab_merges
from cs336_basics.checkpoints import (
        checkpoint_hyperparams)
from cs336_basics.configs import ConfigParams, CheckPtConfig, get_preset_cfg
from cs336_basics.configs import TrainingConfig
from cs336_basics.experiments_tracker import ExperimentTracker

def mini_training_proc():

    tokenfile = "TinyStoriesV2-GPT4"
    hyper_params = ConfigParams(batch_size=64, context_length=128,
                       vocab_size=10000, d_model=512, d_ff=1344, num_layers=4,
                       num_heads=16, theta=10000, resume=False)

    presets = get_preset_cfg()
    presets.train = TrainingConfig(
                        max_steps= 1000, log_interval= 100,
                        eval_interval= 2000, eval_batches = 20)
    hyper_params.checkpoint.interval = 100
    presets.train.max_steps = 500
    presets.train.eval_interval = 100
    presets.train.log_interval == 200

    device = set_device(None)
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() - 2)

    torch.manual_seed(presets.train.seed)
    np.random.seed(presets.train.seed)

    exp = ExperimentTracker(
         log_dir=EXP_PATH,
         service_name=tokenfile,
         config=presets,  # dataclass will be serialized
         mode="train",
     )

    print(78 * "=")
    print("Starting Training...")
    print(78 * "=")
    start = time.time()

    #print("Encode sample file")
    #file_encode_bin_from_vocab_merges(tokenfile, "samples")

    token_train_path = OUT_PATH / f"{tokenfile}-samples.bin"
    train_mm = open_memmap_1d(token_train_path, np_dtype = "uint16")

    token_valid_path = OUT_PATH / f"{tokenfile}-samples.bin"
    valid_mm = open_memmap_1d(token_valid_path, np_dtype = "uint16")

    model_dtype = torch_dtype_from_string("float32")

    print("Train Corpus size: ", len(train_mm))
    training_together(train_mm, valid_mm,
                      hyper_params, presets,
                      exp, device, model_dtype)

    checkpoint_hyperparams(hyper_params, tokenfile)
    exp.close()

    print("took {:.2f} seconds\n".format(time.time() - start))

if __name__ == "__main__":
    mini_training_proc()
