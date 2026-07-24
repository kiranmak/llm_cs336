import os
from cs336_basics.paths import OUT_PATH, DATA_PATH
from cs336_basics.configs import TrainingConfig
from cs336_basics.tokenizer_exp import encode_file_parallel
from runs.train import main_training_loop



def mini_training_proc():

    # Initialize your config (overriding only what you need)
    config = TrainingConfig(
        input_src_file = str(DATA_PATH /"TinyStoriesV2-GPT4-exper.txt"),
        vocab_file = str(OUT_PATH /"TinyStoriesV2-GPT4-train_vocab.json"),
        merge_file = str(OUT_PATH /"TinyStoriesV2-GPT4-train_merges.txt"),
        dataset = str(OUT_PATH   / "TinyStoriesV2-GPT4-exper.bin"),
        valid_set = None,
        batch_size=128,  # custom override
        max_steps = 400,
        context_length= 32,
        eval_interval = 100,
        log_interval = 50,
        exp_name="hello",
        resume = True,
    )
    encode_file_parallel(
                txt_path   = config.input_src_file,
                vocab_path = config.vocab_file,
                merge_path = config.merge_file,
                special_tokens = ["<|endoftext|>"],
                out_path   = config.dataset,
            )
    print(f"====== Encoding to BIN Finished ======")

    main_training_loop(config)

if __name__ == "__main__":
    mini_training_proc()
