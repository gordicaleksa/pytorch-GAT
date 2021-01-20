import enum
import os
import pickle


import scipy.sparse as sp
import numpy as np
import networkx as nx
import torch


from utils.constants import *
# nobody except for PyTorch Geometric did coalescing of the edge index...

# todo: link how the Cora was created and what the feature vectors represent
# todo: Visualize Cora (TikZ, networkx or whatever graph package)
# todo: compare across 5 different repos how people handled Cora

# todo: understand csr, coo, lil sparse matrix formats (understand the other 2 from scipy)

# Currently I only have support for Cora - feel free to add your own data
# GAT and many other papers used the processed version of Cora that can be found here:
# https://github.com/kimiyoung/planetoid


class DatasetType(enum.Enum):
    CORA = 0


def pickle_read(path):
    with open(path, 'rb') as file:
        out = pickle.load(file)

    return out


def load_data(dataset_name):
    if dataset_name.lower() == DatasetType.CORA.name.lower():
        node_features_csr = pickle_read(os.path.join(CORA_PATH, 'node_features.csr'))
        node_labels_npy = pickle_read(os.path.join(CORA_PATH, 'node_labels.npy'))
        adjacency_list_dict = pickle_read(os.path.join(CORA_PATH, 'adjacency_list.dict'))

        nx_graph = nx.from_dict_of_lists(adjacency_list_dict)
        adj = nx.adjacency_matrix(nx_graph)
        adj = adj.tocoo()
        edge_index = np.row_stack((adj.row, adj.col))
        # todo: int32/64?
        # todo: verify results match with PyGeometric (different approaches)
        # todo: add masks

        return node_features_csr, node_labels_npy, edge_index
    else:
        raise Exception(f'{dataset_name} not yet supported.')


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


if __name__ == "__main__":
    print('yo')
    load_data('cora')