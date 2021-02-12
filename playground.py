import time
import os
from collections import defaultdict
import enum


import torch
import scipy.sparse as sp
from scipy.stats import entropy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig


from utils.data_loading import normalize_features_sparse, normalize_features_dense, pickle_save, pickle_read, load_graph_data
from utils.constants import CORA_PATH, BINARIES_PATH, DatasetType, LayerType, DATA_DIR_PATH, cora_label_to_color_map, VisualizationType
from utils.visualizations import draw_entropy_histogram
from models.definitions.GAT import GAT
from utils.utils import print_model_metadata, convert_adj_to_edge_index, name_to_layer_type
from training_script_cora import train_gat_cora, get_training_args


def profile_sparse_matrix_formats(node_features_csr):
    """
        Shows the benefit of using CORRECT sparse formats during processing. Results:
            CSR >> LIL >> dense format, CSR is ~2x faster than LIL and ~8x faster than dense format.

        Official GAT and GCN implementations used LIL. In this particular case it doesn't matter that much,
        since it only takes a couple of milliseconds to process Cora, but it's good to be aware of advantages that
        different sparse formats bring to the table.

        Note: CSR is the fastest format for the operations used in normalize features out of all scipy sparse formats.
        Note2: There are more precise timing functions out there but I believe this one is good enough for my purpose.

    """
    assert sp.isspmatrix_csr(node_features_csr), f'Expected scipy matrix in CSR format, got {type(node_features_csr)}.'
    num_simulation_iterations = 1000

    # You can also use: tocoo(), todok(), tocsc(), todia(), tobsr(), check out data_loading.py's header for more details
    node_features_lil = node_features_csr.tolil()
    node_features_dense = node_features_csr.todense()

    ts = time.time()
    for i in range(num_simulation_iterations):  # LIL
        _ = normalize_features_sparse(node_features_lil)
    print(f'time elapsed, LIL = {(time.time() - ts) / num_simulation_iterations}')

    ts = time.time()
    for i in range(num_simulation_iterations):  # CSR
        _ = normalize_features_sparse(node_features_csr)
    print(f'time elapsed, CSR = {(time.time() - ts) / num_simulation_iterations}')

    ts = time.time()
    for i in range(num_simulation_iterations):  # dense
        _ = normalize_features_dense(node_features_dense)
    print(f'time elapsed, dense = {(time.time() - ts) / num_simulation_iterations}')


def to_GBs(memory_in_bytes):  # beautify memory output - helper function
    return f'{memory_in_bytes / 2**30:.2f} GBs'


def profile_gat_implementations(skip_if_profiling_info_cached=False, store_cache=False):
    """
    Currently for 500 epochs of GAT training the time and memory consumption are  (on my machine - RTX 2080):
        * implementation 1 (IMP1): time ~ 17 seconds, max memory allocated = 1.5 GB and reserved = 1.55 GB
        * implementation 2 (IMP2): time = 15.5 seconds, max memory allocated = 1.4 GB and reserved = 1.55 GB
        * implementation 3 (IMP3): time = 3.5 seconds, max memory allocated = 0.05 GB and reserved = 1.55 GB

    Note: Profiling is done on Cora since the training is faster and most people will have enough VRAM to handle it.

    """

    num_of_profiling_loops = 20
    mem_profiling_dump_filepath = os.path.join(DATA_DIR_PATH, 'memory.dict')
    time_profiling_dump_filepath = os.path.join(DATA_DIR_PATH, 'timing.dict')

    training_config = get_training_args()
    training_config['num_of_epochs'] = 500  # IMP1 and IMP2 take more time so better to drop this one lower
    training_config['should_test'] = False  # just measure training and validation loops
    training_config['should_visualize'] = False  # chart popup would block the simulation
    training_config['enable_tensorboard'] = False  # no need to include this one
    training_config['console_log_freq'] = None  # same here
    training_config['checkpoint_freq'] = None  # and here ^^

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        available_gpu_memory = torch.cuda.get_device_properties(device).total_memory
        print(f'Available memory on default GPU: {to_GBs(available_gpu_memory)}')
    else:
        print('GPU not available. :(')

    # Check whether we have memory and timing info already stored in data directory
    cache_exists = os.path.exists(mem_profiling_dump_filepath) and os.path.exists(time_profiling_dump_filepath)

    # Small optimization - skip the profiling if we have the info stored already and skipping is enabled
    if not (cache_exists and skip_if_profiling_info_cached):

        gat_layer_implementations = [layer_type for layer_type in LayerType]
        results_time = defaultdict(list)
        results_memory = defaultdict(list)

        # We need this loop in order to find the average time and memory consumption more robustly
        for run_id in range(num_of_profiling_loops):
            print(f'Profiling, run_id = {run_id}')

            # Iterate over all the available GAT implementations
            for gat_layer_imp in gat_layer_implementations:
                training_config['layer_type'] = gat_layer_imp  # modify the training config so as to use different imp

                ts = time.time()
                train_gat_cora(training_config)  # train and validation
                results_time[gat_layer_imp.name].append(time.time()-ts)  # collect timing information

                # These 2 methods basically query this function: torch.cuda.memory_stats() it contains much more detail.
                # Here I just care about the peak memory usage i.e. whether you can train GAT on your device.

                # The actual number of GPU bytes needed to store the GPU tensors I use (since the start of the program)
                max_memory_allocated = torch.cuda.max_memory_allocated(device)
                # The above + caching GPU memory used by PyTorch's caching allocator (since the start of the program)
                max_memory_reserved = torch.cuda.max_memory_reserved(device)

                # Reset the peaks so that we get correct results for the next GAT implementation. Otherwise, since the
                # above methods are measuring the peaks since the start of the program one of the less efficient
                # (memory-wise) implementations may eclipse the others.
                torch.cuda.reset_peak_memory_stats(device)

                results_memory[gat_layer_imp.name].append((max_memory_allocated, max_memory_reserved))  # mem info

        if store_cache:
            pickle_save(time_profiling_dump_filepath, results_time)  # dump into cache files
            pickle_save(mem_profiling_dump_filepath, results_memory)
    else:
        print('*' * 50)
        print('Using cached profiling information!')
        print('*' * 50)
        results_time = pickle_read(time_profiling_dump_filepath)
        results_memory = pickle_read(mem_profiling_dump_filepath)

    # Print the results
    for gat_layer_imp in LayerType:
        imp_name = gat_layer_imp.name
        print('*' * 20)
        print(f'{imp_name} GAT training.')
        print(f'Layer type = {gat_layer_imp.name}, training duration = {np.mean(results_time[imp_name]):.2f} [s]')

        max_memory_allocated = np.mean([mem_tuple[0] for mem_tuple in results_memory[imp_name]])
        max_memory_reserved = np.mean([mem_tuple[1] for mem_tuple in results_memory[imp_name]])
        print(f'Max mem allocated = {to_GBs(max_memory_allocated)}, max mem reserved = {to_GBs(max_memory_reserved)}.')


def visualize_graph_dataset(dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU
    config = {
        'dataset_name': dataset_name,  # Cora or PPI
        'layer_type': LayerType.IMP3,  # don't care, but it's needed for load_graph_data function to work
        'should_visualize': True,  # visualize the dataset
        'ppi_load_test_only': True  # only used for PPI, let's just load the test graphs
    }
    load_graph_data(config, device)


def visualize_gat_properties(model_name=r'gat_000000.pth', dataset_name=DatasetType.CORA.name, visualization_type=VisualizationType.ATTENTION):
    """
    Notes on t-SNE:
    Check out this one for more intuition on how to tune t-SNE: https://distill.pub/2016/misread-tsne/

    If you think it'd be useful for me to implement t-SNE as well and explain how every single detail works
    open up an issue or DM me on social media! <3

    Note: I also tried using UMAP but it doesn't provide any more insight than t-SNE.
    (additional con: it has a lot of dependencies if you want to use their plotting functionality)

    """
    # I tried visualizing PPI's 2D embeddings without any label/color information but it's not informative
    if dataset_name == DatasetType.PPI.name and visualization_type == VisualizationType.EMBEDDINGS:
        print(f"{dataset_name} is a multi-label dataset - embeddings are thus not supported.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    config = {
        'dataset_name': dataset_name,
        'layer_type': LayerType.IMP3,
        'should_visualize': False,  # don't visualize the dataset
        'batch_size': 2,  # used only for PPI
        'ppi_load_test_only': True  # used only for PPI (optimization, we're loading only test graphs)
    }

    # Step 1: Prepare the data
    if dataset_name == DatasetType.CORA.name:
        node_features, node_labels, topology, _, _, _ = load_graph_data(config, device)
    else:
        data_loader_test = load_graph_data(config, device)
        node_features, node_labels, topology = next(iter(data_loader_test))
        node_features = node_features.to(device)  # need to explicitly push them to GPU since PPI eats up a lot of VRAM
        node_labels = node_labels.to(device)
        topology = topology.to(device)

    # Step 2: Prepare the model
    model_path = os.path.join(BINARIES_PATH, model_name)
    model_state = torch.load(model_path)

    gat = GAT(
        num_of_layers=model_state['num_of_layers'],
        num_heads_per_layer=model_state['num_heads_per_layer'],
        num_features_per_layer=model_state['num_features_per_layer'],
        add_skip_connection=model_state['add_skip_connection'],
        bias=model_state['bias'],
        dropout=model_state['dropout'],
        layer_type=name_to_layer_type(model_state['layer_type']),
        log_attention_weights=True
    ).to(device)

    print_model_metadata(model_state)
    assert model_state['dataset_name'].lower() == dataset_name.lower(), \
        f"The model was trained on {model_state['dataset_name']} but you're calling it on {dataset_name}."
    gat.load_state_dict(model_state["state_dict"], strict=True)
    gat.eval()  # some layers like nn.Dropout behave differently in train vs eval mode so this part is important

    # Step 3: Calculate the things we'll need for different visualization types (attention, scores, edge_index)

    # This context manager is important (and you'll often see it), otherwise PyTorch will eat much more memory.
    # It would be saving activations for backprop but we are not going to do any model training just the prediction.
    with torch.no_grad():
        # Step 3: Run predictions and collect the high dimensional data
        all_nodes_unnormalized_scores, _ = gat((node_features, topology))  # shape = (N, num of classes)
        all_nodes_unnormalized_scores = all_nodes_unnormalized_scores.cpu().numpy()

    # We'll need the edge index in different for multiple visualization types
    if config['layer_type'] == LayerType.IMP3:  # imp 3 works with edge index while others work with adjacency info
        edge_index = topology
    else:
        edge_index = convert_adj_to_edge_index(topology)

    # Step 4: Perform a specific visualization
    if visualization_type == VisualizationType.ATTENTION:
        # The number of nodes for which we want to visualize their attention over neighboring nodes
        # (2x this actually as we add nodes with highest degree + random nodes)
        num_nodes_of_interest = 4  # 4 is an arbitrary number you can play with these numbers
        head_to_visualize = 0  # plot attention from this multi-head attention's head
        gat_layer_id = 0  # plot attention from this GAT layer

        if dataset_name == DatasetType.PPI.name and gat_layer_id != 0:
            print(f'Attention visualization for {dataset_name} is only available for the first layer.')
            return

        # Build up the complete graph
        # node_features shape = (N, FIN), where N is the number of nodes and FIN number of input features
        total_num_of_nodes = len(node_features)
        complete_graph = ig.Graph()
        complete_graph.add_vertices(total_num_of_nodes)  # igraph creates nodes with ids [0, total_num_of_nodes - 1]
        edge_index_tuples = list(zip(edge_index[0, :], edge_index[1, :]))  # igraph requires this format
        complete_graph.add_edges(edge_index_tuples)

        # Pick the target nodes to plot (nodes with highest degree + random nodes)
        # Note: there could be an overlap between random nodes and nodes with highest degree - but highly unlikely
        nodes_of_interest_ids = np.argpartition(complete_graph.degree(), -num_nodes_of_interest)[-num_nodes_of_interest:]
        random_node_ids = np.random.randint(low=0, high=total_num_of_nodes, size=num_nodes_of_interest)
        nodes_of_interest_ids = np.append(nodes_of_interest_ids, random_node_ids)
        np.random.shuffle(nodes_of_interest_ids)

        target_node_ids = edge_index[1]
        source_nodes = edge_index[0]

        for target_node_id in nodes_of_interest_ids:
            # Step 1: Find the neighboring nodes to the target node
            # Note: self edges are included so the target node is it's own neighbor (Alexandro yo soy tu madre)
            src_nodes_indices = torch.eq(target_node_ids, target_node_id)
            source_node_ids = source_nodes[src_nodes_indices].cpu().numpy()
            size_of_neighborhood = len(source_node_ids)

            # Step 2: Fetch their labels
            labels = node_labels[source_node_ids].cpu().numpy()

            # Step 3: Fetch the attention weights for edges (attention is logged during GAT's forward pass above)
            # attention shape = (N, NH, 1) -> (N, NH) - we just squeeze the last dim it's superfluous
            all_attention_weights = gat.gat_net[gat_layer_id].attention_weights.squeeze(dim=-1)
            attention_weights = all_attention_weights[src_nodes_indices, head_to_visualize].cpu().numpy()
            # This part shows that for CORA what GAT learns is pretty much constant attention weights! Like in GCN!
            # On the other hand PPI's attention pattern is much less uniform.
            print(f'Max attention weight = {np.max(attention_weights)} and min = {np.min(attention_weights)}')
            attention_weights /= np.max(attention_weights)  # rescale the biggest weight to 1 for nicer plotting

            # Build up the neighborhood graph whose attention we want to visualize
            # igraph constraint - it works with contiguous range of ids so we map e.g. node 497 to 0, 12 to 1, etc.
            id_to_igraph_id = dict(zip(source_node_ids, range(len(source_node_ids))))
            ig_graph = ig.Graph()
            ig_graph.add_vertices(size_of_neighborhood)
            ig_graph.add_edges([(id_to_igraph_id[neighbor], id_to_igraph_id[target_node_id]) for neighbor in source_node_ids])

            # Prepare the visualization settings dictionary and plot
            visual_style = {
                "edge_width": attention_weights,  # make edges as thick as the corresponding attention weight
                "layout": ig_graph.layout_reingold_tilford_circular()  # layout for tree-like graphs
            }
            # This is the only part that's Cora specific as Cora has 7 labels
            if dataset_name.lower() == DatasetType.CORA.name.lower():
                visual_style["vertex_color"] = [cora_label_to_color_map[label] for label in labels]
            else:
                print('Add custom color scheme for your specific dataset. Using igraph default coloring.')

            ig.plot(ig_graph, **visual_style)

    elif visualization_type == VisualizationType.EMBEDDINGS:  # visualize embeddings (using t-SNE)
        node_labels = node_labels.cpu().numpy()
        num_classes = len(set(node_labels))

        # Feel free to experiment with perplexity it's arguable the most important parameter of t-SNE and it basically
        # controls the standard deviation of Gaussians i.e. the size of the neighborhoods in high dim (original) space.
        # Simply put the goal of t-SNE is to minimize the KL-divergence between joint Gaussian distribution fit over
        # high dim points and between the t-Student distribution fit over low dimension points (the ones we're plotting)
        # Intuitively, by doing this, we preserve the similarities (relationships) between the high and low dim points.
        # This (probably) won't make much sense if you're not already familiar with t-SNE, God knows I've tried. :P
        t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(all_nodes_unnormalized_scores)

        for class_id in range(num_classes):
            # We extract the points whose true label equals class_id and we color them in the same way, hopefully
            # they'll be clustered together on the 2D chart - that would mean that GAT has learned good representations!
            plt.scatter(t_sne_embeddings[node_labels == class_id, 0], t_sne_embeddings[node_labels == class_id, 1], s=20, color=cora_label_to_color_map[class_id], edgecolors='black', linewidths=0.2)
        plt.show()

    # We want our local probability distributions (attention weights over the neighborhoods) to be
    # non-uniform because that means that GAT is learning a useful pattern. Entropy histograms help us visualize
    # how different those neighborhood distributions are from the uniform distribution (constant attention).
    # If the GAT is learning const attention we could well be using GCN or some even simpler models.
    elif visualization_type == VisualizationType.ENTROPY:
        num_heads_per_layer = [layer.num_of_heads for layer in gat.gat_net]
        num_layers = len(num_heads_per_layer)
        num_of_nodes = len(node_features)

        target_node_ids = edge_index[1].cpu().numpy()

        # For every GAT layer and for every GAT attention head plot the entropy histogram
        for layer_id in range(num_layers):
            # Fetch the attention weights for edges (attention is logged during GAT's forward pass above)
            # attention shape = (N, NH, 1) -> (N, NH) - we just squeeze the last dim it's superfluous
            all_attention_weights = gat.gat_net[layer_id].attention_weights.squeeze(dim=-1).cpu().numpy()

            # tmp fix for PPI there are some numerical problems and so most of attention coefficients are 0
            # and thus we can't plot entropy histograms
            if dataset_name == DatasetType.PPI.name and layer_id > 0:
                print(f'Entropy histograms for {dataset_name} are available only for the first layer.')
                break

            for head_id in range(num_heads_per_layer[layer_id]):
                uniform_dist_entropy_list = []  # save the ideal uniform histogram as the reference
                neighborhood_entropy_list = []

                # This can also be done much more efficiently via scatter_add_ (no for loops)
                # pseudo: out.scatter_add_(node_dim, -all_attention_weights * log(all_attention_weights), target_index)
                for target_node_id in range(num_of_nodes):  # find every the neighborhood for every node in the graph
                    # These attention weights sum up to 1 by GAT design so we can treat it as a probability distribution
                    neigborhood_attention = all_attention_weights[target_node_ids == target_node_id].flatten()
                    # Reference uniform distribution of the same length
                    ideal_uniform_attention = np.ones(len(neigborhood_attention))/len(neigborhood_attention)

                    # Calculate the entropy, check out this video if you're not familiar with the concept:
                    # https://www.youtube.com/watch?v=ErfnhcEV1O8 (Aurélien Géron)
                    neighborhood_entropy_list.append(entropy(neigborhood_attention, base=2))
                    uniform_dist_entropy_list.append(entropy(ideal_uniform_attention, base=2))

                title = f'{dataset_name} entropy histogram layer={layer_id}, attention head={head_id}'
                draw_entropy_histogram(uniform_dist_entropy_list, title, color='orange', uniform_distribution=True)
                draw_entropy_histogram(neighborhood_entropy_list, title, color='dodgerblue')

                fig = plt.gcf()  # get current figure
                plt.show()
                fig.savefig(os.path.join(DATA_DIR_PATH, f'layer_{layer_id}_head_{head_id}.jpg'))
                plt.close()
    else:
        raise Exception(f'Visualization type {visualization_type} not supported.')


class PLAYGROUND(enum.Enum):
    PROFILE_SPARSE = 0,
    PROFILE_GAT = 1,
    VISUALIZE_DATASET = 2,
    VISUALIZE_GAT = 3


if __name__ == '__main__':
    #
    # Pick the function you want to play with <3
    #
    playground_fn = PLAYGROUND.VISUALIZE_GAT

    if playground_fn == PLAYGROUND.PROFILE_SPARSE:
        # shape = (N, F), where N is the number of nodes and F is the number of features
        node_features_csr = pickle_read(os.path.join(CORA_PATH, 'node_features.csr'))
        profile_sparse_matrix_formats(node_features_csr)

    elif playground_fn == PLAYGROUND.PROFILE_GAT:
        # Set to True if you want to use the caching mechanism. Once you compute the profiling info it gets stored
        # in data/ dir as timing.dict and memory.dict which you can later just load instead of computing again
        profile_gat_implementations(skip_if_profiling_info_cached=True, store_cache=True)

    elif playground_fn == PLAYGROUND.VISUALIZE_DATASET:
        visualize_graph_dataset(dataset_name=DatasetType.CORA.name)  # pick between CORA and PPI

    elif playground_fn == PLAYGROUND.VISUALIZE_GAT:
        visualize_gat_properties(
            model_name=r'gat_000000.pth',  # set to PPI or CORA model, keep it in sync with the 'dataset_name'
            dataset_name=DatasetType.CORA.name,  # pick between CORA and PPI
            visualization_type=VisualizationType.EMBEDDINGS  # pick between attention, t-SNE embeddings and entropy
        )

    else:
        raise Exception(f'Woah, this playground function "{playground_fn}" does not exist.')



