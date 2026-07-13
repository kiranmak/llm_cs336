from runs.train import main_training_loop
from runs.train_helper import parse_user_params

def mini_training_proc():

    config = parse_user_params()
    # Print each key and value on a new line
    config.print()
    main_training_loop(config)

if __name__ == "__main__":
    mini_training_proc()
