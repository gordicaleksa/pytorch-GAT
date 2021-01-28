"""
    Currently I only have support for Cora dataset - feel free to add your own graph data.
    You can find the details on how Cora was constructed here: http://eliassi.org/papers/ai-mag-tr08.pdf

    TL;DR:
    The feature vectors are 1433 features long. The authors found the most frequent words across every paper
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


    Note on sparse matrices:
        If you're like me you didn't have to deal with sparse matrices until you started playing with GNNs.
        You'll usually see the following formats in GNN implementations: LIL, COO, CSR and CSC.
        Occasionally, you'll also see DOK and in special settings DIA and BSR as well (so 7 in total).

        It's not nuclear physics (it's harder :P) check out these 2 links and you're good to go:
            * https://docs.scipy.org/doc/scipy/reference/sparse.html
            * https://en.wikipedia.org/wiki/Sparse_matrix

        TL;DR:
        LIL, COO and DOK are used for efficient modification of your sparse structure (add/remove edges)
        CSC and CSR are used for efficient arithmetic operations (addition, multiplication, etc.)
        DIA and BSR are used when you're dealing with special types of sparse matrices - diagonal and block matrices.

"""

import pickle


import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch


from utils.constants import *
from utils.visualizations import plot_in_out_degree_distributions, visualize_graph


# todo: after I get GAT working e2e compare across 5 different repos how people handled Cora


def load_graph_data(training_config, device):
    dataset_name = training_config['dataset_name'].lower()
    layer_type = training_config['layer_type']
    should_visualize = training_config['should_visualize']

    if dataset_name == DatasetType.CORA.name.lower():

        # shape = (N, F), where N is the number of nodes and F is the number of features
        node_features_csr = pickle_read(os.path.join(CORA_PATH, 'node_features.csr'))
        # shape = (N, 1)
        node_labels_npy = pickle_read(os.path.join(CORA_PATH, 'node_labels.npy'))
        # shape = (N, number of neighboring nodes) <- this is a dictionary not a matrix!
        adjacency_list_dict = pickle_read(os.path.join(CORA_PATH, 'adjacency_list.dict'))

        # Normalize the features
        node_features_csr = normalize_features_sparse(node_features_csr)
        num_of_nodes = len(node_labels_npy)

        # todo: refactor this once I have imps in place
        if layer_type == LayerType.IMP2 or layer_type == LayerType.IMP1:  # some implementations rely on adjacency matrix others on edge index
            connectivity_data = nx.adjacency_matrix(nx.from_dict_of_lists(adjacency_list_dict)).todense().astype(np.float)
            connectivity_data += np.identity(connectivity_data.shape[0])
            connectivity_data[connectivity_data == 0] = -np.inf
            connectivity_data[connectivity_data == 1] = 0
        elif layer_type == LayerType.IMP3:
            # Build edge index explicitly (faster than nx ~100 times and as fast as PyGeometric imp, less complicated)
            # shape = (2, E), where E is the number of edges, and 2 for source and target nodes. Basically edge index
            # contains tuples of the format S->T, e.g. 0->3 means that node with id 0 points to a node with id 3.
            connectivity_data = build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=True)
        else:
            raise Exception(f'Layer type {layer_type} not yet supported.')

        if should_visualize:  # network analysis and graph drawing
            plot_in_out_degree_distributions(connectivity_data, dataset_name)
            visualize_graph(connectivity_data, node_labels_npy, dataset_name)

        # Convert to dense PyTorch tensors

        # Needs to be long int type because later functions like PyTorch's index_select expect it
        edge_index = torch.tensor(connectivity_data, dtype=torch.long, device=device)  # todo: long for adj?
        node_labels = torch.tensor(node_labels_npy, dtype=torch.long, device=device)  # todo: do I need long? save mem!
        node_features = torch.tensor(node_features_csr.todense(), device=device)

        # Indices that help us extract nodes that belong to the train/val and test splits
        train_indices = torch.arange(CORA_TRAIN_RANGE[0], CORA_TRAIN_RANGE[1], dtype=torch.long, device=device)
        val_indices = torch.arange(CORA_VAL_RANGE[0], CORA_VAL_RANGE[1], dtype=torch.long, device=device)
        test_indices = torch.arange(CORA_TEST_RANGE[0], CORA_TEST_RANGE[1], dtype=torch.long, device=device)

        return node_features, node_labels, edge_index, train_indices, val_indices, test_indices
    else:
        raise Exception(f'{dataset_name} not yet supported.')


# All Cora data is stored as pickle
def pickle_read(path):
    with open(path, 'rb') as file:
        out = pickle.load(file)

    return out


def pickle_save(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def normalize_features_sparse(node_features_sparse):
    assert sp.issparse(node_features_sparse), f'Expected a sparse matrix, got {node_features_sparse}.'

    # Instead of dividing (like in normalize_features_dense()) we do multiplication with inverse sum of features.
    # Modern hardware (GPUs, TPUs, ASICs) is optimized for fast matrix multiplications! ^^ (* >> /)
    node_features_sum = np.array(node_features_sparse.sum(1))  # sum features for every node feature vector
    # Make an inverse (remember * by 1/x is better (faster) then / by x)
    node_features_inv_sum = np.power(node_features_sum, -1).squeeze()
    # Again certain sums will be 0 so 1/0 will give us inf so we replace those by 0 which is a neutral element for mul
    node_features_inv_sum[np.isinf(node_features_inv_sum)] = 0.
    # squeeze() was there to make dimension go from (N, 1) -> N for sp.diags
    diagonal_inv_features_sum_matrix = sp.diags(node_features_inv_sum)
    # This thing is fast, we return the normalized features
    return diagonal_inv_features_sum_matrix.dot(node_features_sparse)


# Not used -> check out playground.py where it is used in profiling functions
def normalize_features_dense(node_features_dense):
    assert isinstance(node_features_dense, np.matrix), f'Expected np matrix got {type(node_features_dense)}.'

    # The goal is to make feature vectors normalized (sum equals 1), but since some feature vectors are all 0s
    # in those cases we'd have division by 0 so I set the min value (via np.clip) to 1.
    # Note: 1 is a neutral element for division i.e. it won't modify the feature vector
    return node_features_dense / np.clip(node_features_dense.sum(1), a_min=1, a_max=None)


def build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=True):
    source_nodes_ids, target_nodes_ids = [], []
    seen_edges = set()

    for src_node, neighboring_nodes in adjacency_list_dict.items():
        for trg_node in neighboring_nodes:
            # if this edge hasn't been seen so far we add it to the edge index (coalescing - removing duplicates)
            if (src_node, trg_node) not in seen_edges:  # it's easy to explicitly remove self-edges (Cora has none)
                source_nodes_ids.append(src_node)
                target_nodes_ids.append(trg_node)
                seen_edges.add((src_node, trg_node))

    if add_self_edges:
        source_nodes_ids.extend(np.arange(num_of_nodes))
        target_nodes_ids.extend(np.arange(num_of_nodes))

    # shape = (2, E), where E is the number of edges in the graph
    edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))

    return edge_index


# Not used - this is yet another way to construct the edge index by leveraging the existing package (networkx)
# (it's just slower than my simple implementation build_edge_index())
def build_edge_index_nx(adjacency_list_dict):
    nx_graph = nx.from_dict_of_lists(adjacency_list_dict)
    adj = nx.adjacency_matrix(nx_graph)
    adj = adj.tocoo()

    return np.row_stack((adj.row, adj.col))


# For data loading testing purposes feel free to ignore
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU
    config = {
        'dataset_name': DatasetType.CORA.name,
        'layer_type': LayerType.IMP3,
        'should_visualize': False  # don't visualize the dataset
    }
    load_graph_data(config, device)
