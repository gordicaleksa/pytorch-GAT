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

# todo: link how the Cora was created and what the feature vectors represent
# todo: compare across 5 different repos how people handled Cora
# todo: add t-SNE visualization of trained GAT model

# todo: understand csr, csc, coo, lil, dok sparse matrix formats (understand the other 2 from scipy)

# Currently I only have support for Cora - feel free to add your own data
# GAT and many other papers used the processed version of Cora that can be found here:
# https://github.com/kimiyoung/planetoid


class DatasetType(enum.Enum):
    CORA = 0


class GraphVisualizationTool(enum.Enum):
    NETWORKX = 0,
    IGRAPH = 1


def pickle_read(path):
    with open(path, 'rb') as file:
        out = pickle.load(file)

    return out


def get_num_nodes_from_edge_index(edge_index):
    return len(set(np.unique(edge_index[0])).union(set(np.unique(edge_index[1]))))


def plot_in_out_degree_distributions(edge_index, dataset_name):
    """
        Note: It would be easy to do various kinds of powerful network analysis using igraph.
        I chose to explicitly calculate only node degree statistics here, but you can go much further if needed and
        calculate the graph diameter, number of triangles and many other concepts from the network analysis field.

    """
    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'

    num_of_nodes = get_num_nodes_from_edge_index(edge_index)
    # Store each node's input and output degree (they're the same for undirected graphs such as Cora)
    in_degrees = np.zeros(num_of_nodes, dtype=np.int)
    out_degrees = np.zeros(num_of_nodes, dtype=np.int)

    # Edge index shape = (2, num_of_edges), the first row contains the source nodes, the second one target/sink nodes
    # Note on terminology: source nodes point to target/sink nodes
    for cnt in range(edge_index.shape[1]):
        source_node_id = edge_index[0, cnt]
        target_node_id = edge_index[1, cnt]

        out_degrees[source_node_id] += 1  # source node points towards some other node -> increment it's out degree
        in_degrees[target_node_id] += 1

    hist = np.zeros(np.max(out_degrees) + 1)
    for out_degree in out_degrees:
        hist[out_degree] += 1

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.6)
    plt.subplot(311)
    plt.plot(in_degrees, color='red')
    plt.xlabel('node id'); plt.ylabel('in-degree count'); plt.title('Input degree for different node ids')

    plt.subplot(312)
    plt.plot(out_degrees, color='green')
    plt.xlabel('node id'); plt.ylabel('out-degree count'); plt.title('Out degree for different node ids')

    plt.subplot(313)
    plt.plot(hist, color='blue')
    plt.xlabel('node degree'); plt.ylabel('# nodes for the given degree'); plt.title(f'Node degree distribution for {dataset_name} dataset')
    plt.xticks(np.arange(0, len(hist), 5.0))
    plt.grid(True)
    plt.show()


def visualize_graph(edge_index, node_labels, dataset_name, visualization_tool=GraphVisualizationTool.IGRAPH):
    """
    Check out this blog for available graph visualization tools:
        https://towardsdatascience.com/large-graph-visualization-tools-and-approaches-2b8758a1cd59

    Basically depending on how big your graph is there may be better drawing tools than igraph.

    """
    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'

    num_of_nodes = get_num_nodes_from_edge_index(edge_index)
    edge_index_tuples = list(zip(edge_index[0, :], edge_index[1, :]))

    # Networkx package is primarily used for network analysis, graph visualization was an afterthought in the design
    # of the package - but nonetheless you'll see it used for graph drawing as well
    if visualization_tool == GraphVisualizationTool.NETWORKX:
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(edge_index_tuples)
        nx.draw_networkx(nx_graph)
        plt.show()

    elif visualization_tool == GraphVisualizationTool.IGRAPH:
        # Construct the igraph graph
        ig_graph = ig.Graph()
        ig_graph.add_vertices(num_of_nodes)
        ig_graph.add_edges(edge_index_tuples)

        # Prepare the visualization settings dictionary
        visual_style = {}

        # Defines the size of the plot and margins
        visual_style["bbox"] = (3000, 3000)
        visual_style["margin"] = 17

        # A simple heuristic, defines the edge thickness. I've chosen thickness so that it's proportional to the number
        # of shortest paths (geodesics) that go through a certain edge in our graph (edge_betweenness function)

        # line1: I use log otherwise some edges will be too thick and others not visible at all
        # edge_betweeness returns 0.5 for certain edges that's why I use np.clip as log will be negative for those edges
        # line2: Normalize so that the thickest edge is 1 otherwise edges appear too thick on the chart
        # line3: The idea here is to make the strongest edge stay stronger than others, 6 just worked, don't dwell on it

        edge_weights_raw = np.clip(np.log(ig_graph.edge_betweenness()), a_min=0, a_max=None)
        edge_weights_raw_normalized = edge_weights_raw / np.max(edge_weights_raw)
        edge_weights = [w**6 for w in edge_weights_raw_normalized]
        visual_style["edge_width"] = edge_weights  # using a constant here like 0.1 also gives nice visualizations

        # A simple heuristic for vertex size. Size ~ (degree / 2) (it gave nice results I tried log and sqrt as well)
        visual_style["vertex_size"] = [deg / 2 for deg in ig_graph.degree()]

        # This is the only part that's Cora specific as Cora has 7 labels
        if dataset_name.lower() == DatasetType.CORA.name.lower():
            label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}
            visual_style["vertex_color"] = [label_to_color_map[label] for label in node_labels]
        else:
            print('Feel free to add custom color scheme for your specific dataset. Using igraph default coloring.')

        # Set the layout - the way the graph is presented on a 2D chart. Graph drawing is a subfield for itself!
        # I used "Kamada Kawai" a force-directed method, this family of methods are based on physical system simulation.
        # (layout_drl also gave nice results for Cora)
        visual_style["layout"] = ig_graph.layout_kamada_kawai()

        ig.plot(ig_graph, **visual_style)
    else:
        raise Exception(f'Visualization tool {visualization_tool.name} not supported.')


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
            # plot_in_out_degree_distributions(edge_index, dataset_name)
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


def build_edge_index_nx(adjacency_list_dict):
    nx_graph = nx.from_dict_of_lists(adjacency_list_dict)
    adj = nx.adjacency_matrix(nx_graph)
    adj = adj.tocoo()

    return np.row_stack((adj.row, adj.col))


if __name__ == "__main__":
    load_data('cora')