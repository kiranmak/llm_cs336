from runs.train import main_training_loop
from runs.train_helper import parse_user_params

def mini_training_proc():

    config = parse_user_params()
    print("Training config:", config)
    main_training_loop(config)

if __name__ == "__main__":
    mini_training_proc()
