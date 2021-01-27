import argparse
import os
import time


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


from models.definitions.GAT import GAT
from utils.data_loading import load_graph_data
from utils.constants import DatasetType, CHECKPOINTS_PATH, BINARIES_PATH, LayerType, CORA_NUM_CLASSES, CORA_NUM_INPUT_FEATURES
import utils.utils as utils


# Global vars used for early stopping. After some number of epochs (as defined by the patience_period var) without any
# improvement on the validation dataset (measured via accuracy metric), we'll break out from the training loop.
BEST_VAL_ACC = 0
PATIENCE_CNT = 0

writer = SummaryWriter()  # (tensorboard) writer will output to ./runs/ directory by default


# Simple decorator function so that I don't have to pass arguments that don't change from epoch to epoch
def get_train_val_loop(gat, cross_entropy_loss, optimizer, node_features, node_labels, edge_index, train_indices, val_indices, patience_period, time_start):

    train_labels = node_labels.index_select(0, train_indices)
    val_labels = node_labels.index_select(0, val_indices)
    node_dim = 0  # this will likely change as soon as I add an inductive example (Cora is transductive)
    graph_data = (node_features, edge_index)

    def train_val_loop(is_train, epoch):
        global BEST_VAL_ACC, PATIENCE_CNT, writer

        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        if is_train:
            gat.train()
        else:
            gat.eval()

        node_indices = train_indices if is_train else val_indices
        gt_node_labels = train_labels if is_train else val_labels  # gt stands for ground truth

        # The loss function applies both the softmax as well as the cross-entropy
        # Note: 0 index just extracts the node_features part of the data (index 1 contains the edge_index)
        # shape = (N, C) where N is the number of nodes in the graph and C is the number of classes
        nodes_unnormalized_scores = gat(graph_data)[0].index_select(node_dim, node_indices)

        # Example: let's take an output for a single node on Cora it's a vector of size 7 and it contains unnormalized
        # scores like: [3.4, 23.1, ..., -2.1] what the cross entropy does is for every vector it applies softmax
        # so we'll have the above vector transformed into say [0.05, 0.8, ..., 0.01] and then whatever the correct class
        # is (say it's 1) it will take the element at position 1, 0.8 in this case, and the loss is -log(0.8). You
        # can see that as the probability of the correct class approaches 1 we get to 0 loss! <3
        # todo: extract a real vector and put the numbers above
        loss = cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)

        if is_train:
            optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
            loss.backward()  # compute the gradients for every trainable weight in the computational graph
            optimizer.step()  # apply the gradients to weights

        # Finds the index of maximum (unnormalized) score for every node and that's the class prediction for that node.
        # Compare those to true (ground truth) labels and find the fraction of correct predictions -> accuracy metric.
        class_predictions = torch.argmax(nodes_unnormalized_scores, dim=-1)
        accuracy = torch.sum(torch.eq(class_predictions, gt_node_labels).long()).item() / len(gt_node_labels)

        #
        # Logging
        #

        if is_train:
            # Log metrics
            if training_config['enable_tensorboard']:
                writer.add_scalar('training_loss', loss.item(), epoch)
                writer.add_scalar('training_acc', accuracy, epoch)

            # Log to console
            if training_config['console_log_freq'] is not None and epoch % training_config['console_log_freq'] == 0:
                print(f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] | epoch={epoch + 1}')

            # Save model checkpoint
            if training_config['checkpoint_freq'] is not None and (epoch + 1) % training_config['checkpoint_freq'] == 0:
                ckpt_model_name = f"gat_ckpt_epoch_{epoch + 1}.pth"
                torch.save(utils.get_training_state(training_config, gat), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))
        else:
            # Log metrics
            if training_config['enable_tensorboard']:
                writer.add_scalar('val_loss', loss.item(), epoch)
                writer.add_scalar('val_acc', accuracy, epoch)

            # Break logic
            if accuracy > BEST_VAL_ACC:
                BEST_VAL_ACC = accuracy  # keep track of the best validation accuracy so far
                PATIENCE_CNT = 0  # reset the counter every time we encounter new best accuracy
            else:
                PATIENCE_CNT += 1  # otherwise keep counting

            if PATIENCE_CNT >= patience_period:
                raise Exception('Stopping the training, the universe has no more patience for this training.')

        return train_val_loop  # return the decorated function


# todo: see why I'm overfitting and why is IMP1 superior over IMP2/3???
# todo: report test accuracy
# todo: add visualization
def train_gat(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # Step 1: load the graph data
    node_features, node_labels, edge_index, train_indices, val_indices, test_indices = load_graph_data(training_config)

    # Step 2: prepare the model
    gat = GAT(
        training_config['num_of_layers'],
        training_config['num_heads_per_layer'],
        training_config['num_features_per_layer'],
        training_config['training_config']
    ).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=training_config['lr'], weight_decay=training_config['weight_decay'])

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    train_val_loop = get_train_val_loop(
        gat,
        loss_fn,
        optimizer,
        node_features,
        node_labels,
        edge_index,
        train_indices,
        val_indices,
        training_config['patience_period'],
        time.time())

    # Step 4: Start the training
    for epoch in range(training_config['num_of_epochs']):
        # Training loop
        train_val_loop(is_train=True, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            try:
                train_val_loop(is_train=True, epoch=epoch)
            except e:
                print(str(e))

    # Save the latest GAT in the binaries directory
    torch.save(utils.get_training_state(training_config, gat), os.path.join(BINARIES_PATH, utils.get_available_binary_name()))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=1000)
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement before terminating", default=1000)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)

    # Dataset related
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training', default=DatasetType.CORA.name)
    parser.add_argument("--should_visualize", action='store_true', help='should visualize the dataset? (no by default)')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging")
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=10)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=100)
    args = parser.parse_args()

    # Model architecture related
    gat_config = {
        "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [8, 1],
        "num_features_per_layer": [CORA_NUM_INPUT_FEATURES, 8, CORA_NUM_CLASSES],
        "layer_type": LayerType.IMP3  # fastest implementation enabled by default
    }

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    # Add additional config information
    training_config.update(gat_config)

    # Train the graph attention network
    train_gat(training_config)
