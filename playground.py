import time
import os
from collections import defaultdict


import torch
import scipy.sparse as sp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig


from utils.data_loading import normalize_features_sparse, normalize_features_dense, pickle_save, pickle_read, load_graph_data
from utils.constants import CORA_PATH, BINARIES_PATH, DatasetType, LayerType, DATA_DIR_PATH, name_to_layer_type, label_to_color_map
from models.definitions.GAT import GAT
from utils.utils import print_model_metadata
from train import train_gat, get_training_args


def profile_sparse_matrix_formats(node_features_csr):
    """
        Shows the benefit of using CORRECT sparse formats during processing, results:
            CSR >> LIL >> dense format, CSR is ~2x faster from LIL and ~8x faster from dense format

        Official GAT and GCN implementations used LIL. In this particular case it doesn't matter that much,
        since it only takes a couple of ms to process Cora, but it's good to be aware of advantages that
        different sparse formats bring to the table.

        Note: CSR is the fastest format for the operations used in normalize features out of all scipy sparse formats.

    """
    assert sp.isspmatrix_csr(node_features_csr), f'Expected scipy matrix in CSR format, got {type(node_features_csr)}.'
    num_loops = 1000

    # You can also use: tocoo(), todok(), tocsc(), todia(), tobsr(), check out data_loading.py's header for more details
    node_features_lil = node_features_csr.tolil()
    node_features_dense = node_features_csr.todense()

    ts = time.time()
    for i in range(num_loops):  # LIL
        _ = normalize_features_sparse(node_features_lil)
    print(f'time elapsed, LIL = {(time.time() - ts) / num_loops}')

    ts = time.time()
    for i in range(num_loops):  # CSR
        _ = normalize_features_sparse(node_features_csr)
    print(f'time elapsed, CSR = {(time.time() - ts) / num_loops}')

    ts = time.time()
    for i in range(num_loops):  # dense
        _ = normalize_features_dense(node_features_dense)
    print(f'time elapsed, dense = {(time.time() - ts) / num_loops}')


def to_GBs(memory_in_bytes):  # beautify memory output - helper function
    return f'{memory_in_bytes / 2**30:.2f} GBs'


def profile_gat_implementations(skip_if_profiling_info_cached=False):
    """
    Currently for 1000 epochs of GAT training the time and memory consumption are  (on my machine - RTX 2080):
        * implementation 1 (IMP1): time ~ 34 seconds, max memory allocated = 1.5 GB and reserved = 1.54 GB
        * implementation 2 (IMP2): time = 20 seconds, max memory allocated = 1.4 GB and reserved = 1.54 GB
        * implementation 3 (IMP3): time = 1.91 seconds, max memory allocated = 0.05 GB and reserved = 1.54 GB

    """

    num_of_profiling_loops = 20
    mem_profiling_dump_filepath = os.path.join(DATA_DIR_PATH, 'memory.dict')
    time_profiling_dump_filepath = os.path.join(DATA_DIR_PATH, 'timing.dict')

    training_config = get_training_args()
    training_config['num_of_epochs'] = 500  # IMP1 and IMP2 take more time so better to drop this one lower
    training_config['should_test'] = False
    training_config['should_visualize'] = False
    training_config['enable_tensorboard'] = False
    training_config['console_log_freq'] = None
    training_config['checkpoint_freq'] = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        available_gpu_memory = torch.cuda.get_device_properties(device).total_memory
        print(f'Available memory on default GPU: {to_GBs(available_gpu_memory)}')
    else:
        print('GPU not available. :(')

    # Check whether we have memory and timing info already stored in data directory
    cache_exists = os.path.exists(mem_profiling_dump_filepath) and os.path.exists(time_profiling_dump_filepath)

    # Small optimization skip the profiling if we have the info stored already and skipping is enabled
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
                train_gat(training_config)
                results_time[gat_layer_imp.name].append(time.time()-ts)  # collect timing information

                # These 2 methods basically query this function: torch.cuda.memory_stats() - contains much more detail.
                # Here I just care about the peak memory usage i.e. whether you can train GAT on your device.

                # The actual number of GPU bytes needed to store the GPU tensors I use (since the start of the program)
                max_memory_allocated = torch.cuda.max_memory_allocated(device)
                # The above + caching GPU memory used by PyTorch's caching allocator (since the start of the program)
                max_memory_reserved = torch.cuda.max_memory_reserved(device)
                # Reset the peaks so that we get correct results for the next GAT implementation. Otherwise, since the
                # above methods are measuring the peaks since the start of the program one of the less efficient
                # (memory-wise) implementations may eclipse the others.
                torch.cuda.reset_peak_memory_stats(device)

                results_memory[gat_layer_imp.name].append((max_memory_allocated, max_memory_reserved))  # collect mem info

        pickle_save(time_profiling_dump_filepath, results_time)
        pickle_save(mem_profiling_dump_filepath, results_memory)
    else:
        results_time = pickle_read(time_profiling_dump_filepath)
        results_memory = pickle_read(mem_profiling_dump_filepath)

    for gat_layer_imp in LayerType:
        imp_name = gat_layer_imp.name
        print('*' * 20)
        print(f'{imp_name} GAT training.')
        print(f'Layer type = {gat_layer_imp.name}, training duration = {np.mean(results_time[imp_name]):.2f} [s]')

        max_memory_allocated = np.mean([mem_tuple[0] for mem_tuple in results_memory[imp_name]])
        max_memory_reserved = np.mean([mem_tuple[1] for mem_tuple in results_memory[imp_name]])
        print(f'Max mem allocated = {to_GBs(max_memory_allocated)}, max mem reserved = {to_GBs(max_memory_reserved)}.')


def visualize_embedding_space_or_attention(model_name=r'gat_000000.pth', dataset_name=DatasetType.CORA.name, visualize_attention=True):
    """
    Using t-SNE to visualize GAT embeddings in 2D space.
    Check out this one for more intuition on how to tune t-SNE: https://distill.pub/2016/misread-tsne/

    If you think it'd be useful for me to implement t-SNE as well and explain how every single detail works
    open up an issue or DM me on social media!

    Note: I also tried using UMAP but it doesn't provide any more insight than t-SNE.
    (con: it has a lot of dependencies if you want to use their plotting functionality)

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    config = {
        'dataset_name': dataset_name,
        'layer_type': LayerType.IMP3,
        'should_visualize': False  # don't visualize the dataset
    }

    # Step 1: Prepare the data
    node_features, node_labels, edge_index, train_indices, val_indices, test_indices = load_graph_data(config, device)

    # Step 2: Prepare the model
    model_path = os.path.join(BINARIES_PATH, model_name)
    model_state = torch.load(model_path)

    gat = GAT(
        num_of_layers=model_state['num_of_layers'],
        num_heads_per_layer=model_state['num_heads_per_layer'],
        num_features_per_layer=model_state['num_features_per_layer'],
        layer_type=name_to_layer_type(model_state['layer_type']),
        log_attention_weights=True
    ).to(device)

    print_model_metadata(model_state)
    gat.load_state_dict(model_state["state_dict"], strict=True)
    gat.eval()

    with torch.no_grad():
        # Step 3: Run predictions and collect the high dimensional data
        all_nodes_unnormalized_scores, _ = gat((node_features, edge_index))
        all_nodes_unnormalized_scores = all_nodes_unnormalized_scores.cpu().numpy()

    # todo: visualize attention
    if visualize_attention:
        # todo: pick some interesting ones with nice degree
        if config['layer_type'] == LayerType.IMP3:
            head_to_visualize = 0  # todo: do it like for the transformer multiple heads
            node_of_interest = 1358  # 1701  306 1358
            target_nodes = edge_index[1]
            source_nodes = edge_index[0]
            indices = torch.eq(target_nodes, node_of_interest)
            neighbors = list(source_nodes[indices].cpu().numpy())
            attention_layer_2 = gat.gat_net[1].attention_weights.squeeze(dim=-1)
            attention_weights = list(attention_layer_2[neighbors, head_to_visualize].cpu().numpy())

            neighbors_to_igraph_id = dict(zip(neighbors, range(len(neighbors))))

            ig_graph = ig.Graph()
            ig_graph.add_vertices(len(neighbors))
            ig_graph.add_edges([(neighbors_to_igraph_id[neighbor], neighbors_to_igraph_id[node_of_interest]) for neighbor in neighbors])

            # Prepare the visualization settings dictionary
            visual_style = {}
            visual_style["edge_width"] = attention_weights  # make edges as thick as the corresponding attention weight
            visual_style["layout"] = ig_graph.layout_reingold_tilford_circular()
            ig.plot(ig_graph, **visual_style)
            
            # This is the only part that's Cora specific as Cora has 7 labels
            # if dataset_name.lower() == DatasetType.CORA.name.lower():
            #     visual_style["vertex_color"] = [label_to_color_map[label] for label in node_labels]
            # else:
            #     print('Feel free to add custom color scheme for your specific dataset. Using igraph default coloring.')

        else:
            print(f'Attention analysis not yet implemented for layer type = {config["layer_type"]}.')

    else:  # visualize embeddings (using t-SNE)
        node_labels = node_labels.cpu().numpy()
        num_classes = len(set(node_labels))

        # Step 4: Calculate the low dim t-SNE representation

        # Feel free to experiment with perplexity it's arguable the most important parameter of t-SNE and it basically
        # controls the standard deviation of Gaussians i.e. the size of the neighborhoods in high dim (original) space.
        # Simply put the goal of t-SNE is to minimize the KL-divergence between joint Gaussian distribution fit over
        # high dim points and between the t-Student distribution fit over low dimension points (the ones we're plotting)
        # Intuitively, by doing this, we preserve the similarities (relationships) between the high and low dim points.
        # This (probably) won't make much sense if you're not already familiar with t-SNE, God knows I've tried. :P
        t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(all_nodes_unnormalized_scores)

        for class_id in range(num_classes):
            # We extract the points whose true label equals class_id and we color them in the same way, hopefully
            # they'll be clustered together on the 2D chart - that would mean that GAT learned good representations!
            plt.scatter(t_sne_embeddings[node_labels == class_id, 0], t_sne_embeddings[node_labels == class_id, 1], s=20, color=label_to_color_map[class_id])
        plt.show()


if __name__ == '__main__':
    # shape = (N, F), where N is the number of nodes and F is the number of features
    # node_features_csr = pickle_read(os.path.join(CORA_PATH, 'node_features.csr'))
    # profile_different_matrix_formats(node_features_csr)
    visualize_embedding_space_or_attention()
    # profile_gat_implementations()
