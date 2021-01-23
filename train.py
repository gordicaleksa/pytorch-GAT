import argparse


import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np


from models.definitions.GAT import GAT
from utils.data_loading import load_graph_data
from utils.constants import DatasetType


def train_gat(training_config, gat_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    node_features, node_labels, edge_index, train_indices, val_indices, test_indices = load_graph_data(DatasetType.CORA.name, device, should_visualize=False)

    gat = GAT(gat_config['num_of_layers'], gat_config['num_heads_per_layer'], gat_config['num_features_per_layer']).to(device)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=training_config['lr'], weight_decay=training_config['weight_decay'])

    # todo: add tranining/val loss logging and accuracy like in other projects
    # todo: use cross entropy ignore_index instead?
    # todo: see why I'm overfitting
    training_loss_log, training_acc_log = [], []
    val_loss_log, val_acc_log = [], []
    training_labels = node_labels.index_select(0, train_indices)
    val_labels = node_labels.index_select(0, val_indices)
    best_val = -np.inf
    patience_cnt = 0
    for epoch in range(training_config['num_of_epochs']):
        if epoch % 100 == 0:
            print(f'{epoch}')
        # Training loop
        gat.train()
        training_nodes_distributions = gat((node_features, edge_index))[0].index_select(0, train_indices)
        training_loss = loss_fn(training_nodes_distributions, training_labels)
        training_predictions = torch.argmax(training_nodes_distributions, dim=-1)
        training_acc = torch.sum(torch.eq(training_predictions, training_labels).long()).item() / len(training_labels)

        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

        training_loss_log.append(training_loss)
        training_acc_log.append(training_acc)

        # Validation loop
        with torch.no_grad():
            gat.eval()
            val_nodes_distributions = gat((node_features, edge_index))[0].index_select(0, val_indices)

            val_loss = loss_fn(val_nodes_distributions, val_labels)

            val_predictions = torch.argmax(val_nodes_distributions, dim=-1)
            val_acc = torch.sum(torch.eq(val_predictions, val_labels).long()).item() / len(val_labels)

            val_loss_log.append(val_loss)
            val_acc_log.append(val_acc)
            if val_acc > best_val:
                best_val = val_acc
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt >= training_config['patience']:  # don't have any more patience to wait ...
                print('Patience has run out')
                break

    plt.plot(training_loss_log)
    plt.plot(val_loss_log)
    plt.show()

    plt.plot(training_acc_log)
    plt.grid()
    plt.plot(val_acc_log)
    plt.show()

    # todo: report test accuracy
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
    parser.add_argument("--patience", type=int, help="number of epochs with no improvement before terminating", default=100)
    parser.add_argument("--lr", type=float, help="learning rate", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)

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
        "num_features_per_layer": [1433, 8, 7]  # todo: make a variable or dynamically figure out we need 7
    }

    # Train the original transformer model
    train_gat(training_config, gat_config)