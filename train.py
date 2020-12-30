import argparse


import torch


from models.definitions.GAT import GAT


def train_gat(training_config, gat_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # todo: add data loading

    gat = GAT(gat_config['num_of_layers'], gat_config['num_heads_per_layer'], gat_config['num_features_per_layer'])

    # todo: add training loop

    # todo: test components

    # todo: add visualization


if __name__ == '__main__':
    #
    # Fixed args - don't change these unless you have a good reason
    #

    #
    # Modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=1000)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging", default=True)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=10)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=1)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    gat_config = {
        "num_of_layers": 2,
        "num_heads_per_layer": [8, 1],
        "num_features_per_layer": [1433, 8, 7]
    }

    # Train the original transformer model
    train_gat(training_config, gat_config)