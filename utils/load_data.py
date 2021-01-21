import enum
import os
import pickle
from itertools import repeat
import time
import igraph as ig

import scipy.sparse as sp
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt


from utils.constants import *
# nobody except for PyTorch Geometric did coalescing of the edge index...

# todo: link how the Cora was created and what the feature vectors represent
# todo: Visualize Cora (TikZ, networkx or whatever graph package)
# todo: compare across 5 different repos how people handled Cora
# todo: add t-SNE visualization of trained GAT model

# todo: understand csr, csc, coo, lil, dok sparse matrix formats (understand the other 2 from scipy)

# Currently I only have support for Cora - feel free to add your own data
# GAT and many other papers used the processed version of Cora that can be found here:
# https://github.com/kimiyoung/planetoid


class DatasetType(enum.Enum):
    CORA = 0


def pickle_read(path):
    with open(path, 'rb') as file:
        out = pickle.load(file)

    return out


def plot_in_out_degree_distributions(edge_index, num_of_nodes):
    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'

    # Store each node's input and output degree (they're the same for undirected graphs such as Cora)
    in_degrees = np.zeros(num_of_nodes)
    out_degrees = np.zeros(num_of_nodes)

    # Edge index shape = (2, num_of_edges), the first row contains the source nodes, the second one target/sink nodes
    # Note on terminology: source nodes point to target/sink nodes
    for cnt in range(edge_index.shape[1]):
        source_node_id = edge_index[0, cnt]
        target_node_id = edge_index[1, cnt]

        out_degrees[source_node_id] += 1  # source node points towards some other node -> increment it's out degree
        in_degrees[target_node_id] += 1

    # todo: add histogram as well, avg degree, max degree

    plt.figure()
    plt.subplot(211)
    plt.plot(in_degrees, color='blue')
    plt.xlabel('node id'); plt.ylabel('degree count'); plt.title('In degree')

    plt.subplot(212)
    plt.plot(out_degrees, color='red')
    plt.xlabel('node id'); plt.ylabel('degree count'); plt.title('Out degree')
    plt.show()


# todo: check out plotly
def visualize_graph(edge_index, node_labels, visualization_tool='nx'):
    """
    This family of methods are based on physical system simulation.
    Vertices are represented as charged particles, that repulse each other, and edges are treated as elastic strings.
    These methods try to model the dynamics of this system or find a minimum of energy.

    """
    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'

    edge_index_tuples = list(zip(edge_index[0, :], edge_index[1, :]))
    if visualization_tool == 'nx':
        G = nx.Graph()
        G.add_edges_from(edge_index_tuples)
        nx.draw_networkx(G)
        plt.show()
    elif visualization_tool == 'igraph':
        # it's easy to do analysis here using igraph instead of what I did with plot_in_out_degree_distributions
        # many tools from network analysis

        g = ig.Graph()
        g.add_vertices(len(node_labels))
        # g.vs["label"] = node_labels
        g.add_edges(edge_index_tuples)
        out_fig_name = "graph.eps"
        visual_style = {}
        # Define colors used for outdegree visualization
        # Set bbox and margin
        visual_style["bbox"] = (3000, 3000)
        visual_style["margin"] = 17
        # Set vertex colours
        color_dict = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}

        visual_style["vertex_color"] = [color_dict[label] for label in node_labels]
        visual_style["edge_width"] = 0.1
        # Set vertex size
        # simple heuristic size ~ degree / 2 (it gave nice results I tried log and sqrt to small)
        tmp = [deg/2 for deg in g.degree()]
        # tmp[tmp.index(max(tmp))] = 20
        visual_style["vertex_size"] = tmp

        # Set the layout (layout_drl also gave nice results for Cora)
        my_layouts = [g.layout_kamada_kawai()]  # force-directed method
        for my_layout in my_layouts:
            visual_style["layout"] = my_layout
            # Plot the graph
            ig.plot(g, **visual_style)  # , **visual_style

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ig.plot(g, target=ax)
    else:
        raise Exception(f'Visualization tool {visualization_tool} not supported.')


def load_data(dataset_name, should_visualize=True):
    if dataset_name.lower() == DatasetType.CORA.name.lower():

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
            # plot_in_out_degree_distributions(edge_index, num_of_nodes)
            visualize_graph(edge_index, node_labels_npy)

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


def build_edge_index_nx(adjacency_list_dict):
    nx_graph = nx.from_dict_of_lists(adjacency_list_dict)
    adj = nx.adjacency_matrix(nx_graph)
    adj = adj.tocoo()

    return np.row_stack((adj.row, adj.col))


if __name__ == "__main__":
    print('yo')
    load_data('cora')