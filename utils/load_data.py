import enum
import os
import pickle
from itertools import repeat
import time


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

        # Build edge index explicitly (faster than nx ~100 times and as fast as PyGeometric imp, far less complicated)
        row, col = [], []
        seen = set()
        for source_node, neighboring_nodes in adjacency_list_dict.items():
            if source_node == 2707:
                print(neighboring_nodes)
            for value in neighboring_nodes:
                if (source_node, value) not in seen:
                    row.append(source_node)
                    col.append(value)
                seen.add((source_node, value))
        edge_index = np.row_stack((row, col)).astype(np.int64)

        pygeo = np.load(r"C:\tmp_data_dir\YouTube\CodingProjects\GNNs_playground\pytorch_geometric\examples\edge_index.npy")
        pygeo_seen = set()
        for i in range(pygeo.shape[1]):
            pygeo_seen.add(tuple(pygeo[:, i]))
        tmp2 = pygeo_seen.difference(seen)
        tmp = np.sum(edge_index != pygeo)
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


def build_edge_index_nx(adjacency_list_dict):
    nx_graph = nx.from_dict_of_lists(adjacency_list_dict)
    adj = nx.adjacency_matrix(nx_graph)
    adj = adj.tocoo()
    return np.row_stack((adj.row, adj.col))


if __name__ == "__main__":
    print('yo')
    load_data('cora')