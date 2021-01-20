import enum
import os
import pickle
import pickletools


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

        dense_matrix = np.zeros((len(adjacency_list_dict), len(adjacency_list_dict)))
        for key, values in adjacency_list_dict.items():
            for v in values:
                dense_matrix[key][v] += 1

        # todo: figure out whether this line dropped some edges silently -> yes it did
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(adjacency_list_dict))
        adj = adj.todense()

        var1 = np.sum(dense_matrix)
        var2 = np.sum(adj)
        print(var1, var2, np.trace(dense_matrix), np.trace(adj))
        print('mkay')
        # no coalescing
    else:
        raise Exception(f'{dataset_name} not yet supported.')


# def edge_index_from_dict(graph_dict, num_nodes=None):
#     row, col = [], []
#     for key, value in graph_dict.items():
#         row += repeat(key, len(value))
#         col += value
#     edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
#     # NOTE: There are duplicated edges and self loops in the datasets. Other
#     # implementations do not remove them!
#     edge_index, _ = remove_self_loops(edge_index)
#     edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
#     return edge_index


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


if __name__ == "__main__":
    print('yo')
    load_data('cora')