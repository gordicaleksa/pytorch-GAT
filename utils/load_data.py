"""
    Currently I only have support for Cora dataset - feel free to add your own graph data.
    You can find the details on how Cora was constructed here: http://eliassi.org/papers/ai-mag-tr08.pdf

    TL;DR: The feature vectors are 1433 features long. The authors found the most frequent words across every paper
    in the graph (they've removed the low frequency words + some additional processing) and made a vocab from those.
    Now feature "i" in the feature vector tells us whether the paper contains i-th word from the vocab (1-yes, 0-no).
    e.g. : feature vector 100...00 means that this node/paper has only 0th word of the vocab.

    Note on Cora processing:
        GAT and many other papers (GCN, etc.) used the processed version of Cora that can be found here:
        https://github.com/kimiyoung/planetoid

        I started from that same data, and after pre-processing it the same way as GAT and GCN,
        I've saved it into only 3 files so there is no need to copy-paste the same pre-processing code around anymore.

        Node features are saved in CSR sparse format, labels go from 0-6 (not one-hot) and finally the topology of the
        graph remained the same I just renamed it to adjacency_list.dict.

"""

import pickle


import numpy as np
import networkx as nx
import torch


from utils.constants import *
from utils.visualizations import plot_in_out_degree_distributions, visualize_graph


# todo: compare across 5 different repos how people handled Cora
# todo: add t-SNE visualization of trained GAT model
# todo: understand csr, csc, coo, lil, dok sparse matrix formats (understand the other 2 from scipy)


def pickle_read(path):
    with open(path, 'rb') as file:
        out = pickle.load(file)

    return out


def load_data(dataset_name, should_visualize=True):
    dataset_name = dataset_name.lower()
    if dataset_name == DatasetType.CORA.name.lower():

        node_features_csr = pickle_read(os.path.join(CORA_PATH, 'node_features.csr'))
        # todo: process features
        node_labels_npy = pickle_read(os.path.join(CORA_PATH, 'node_labels.npy'))
        adjacency_list_dict = pickle_read(os.path.join(CORA_PATH, 'adjacency_list.dict'))
        num_of_nodes = len(adjacency_list_dict)

        # Build edge index explicitly (faster than nx ~100 times and as fast as PyGeometric imp, far less complicated)
        row, col = [], []
        seen = set()
        for source_node, neighboring_nodes in adjacency_list_dict.items():
            for value in neighboring_nodes:
                if (source_node, value) not in seen:
                    row.append(source_node)
                    col.append(value)
                seen.add((source_node, value))
        # todo: be explicit about shapes throughout the code
        edge_index = np.row_stack((row, col))

        if should_visualize:
            plot_in_out_degree_distributions(edge_index, dataset_name)
            visualize_graph(edge_index, node_labels_npy, dataset_name)

        # todo: int32/64?
        # todo: add masks

        # todo: convert to PT tensors
        return node_features_csr, node_labels_npy, edge_index
    else:
        raise Exception(f'{dataset_name} not yet supported.')


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


# Not used this is another way how you can construct the edge index by using existing package
# (it's just slower than my simple implementation)
def build_edge_index_nx(adjacency_list_dict):
    nx_graph = nx.from_dict_of_lists(adjacency_list_dict)
    adj = nx.adjacency_matrix(nx_graph)
    adj = adj.tocoo()

    return np.row_stack((adj.row, adj.col))


# For data loading testing purposes feel free to ignore
if __name__ == "__main__":
    load_data(DatasetType.CORA.name, should_visualize=True)
