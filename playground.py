import time
import os


import torch
import torch.nn as nn
import scipy.sparse as sp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


from utils.data_loading import normalize_features_sparse, normalize_features_dense, pickle_read, load_graph_data
from utils.constants import CORA_PATH, BINARIES_PATH, DatasetType, LayerType
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


def profile_gat_implementations():
    """
    Currently for 100 epochs the time and memory are (on my machine - RTX 2080):
        * implementation 1 (IMP1): time = 37.38 seconds
        * implementation 2 (IMP2): time = 32.59 seconds
        * implementation 3 (IMP3): time = 1.91 seconds

    """

    training_config = get_training_args()
    training_config['num_of_epochs'] = 100  # IMP1 and IMP2 take more time so better to drop this one lower
    training_config['should_test'] = False
    training_config['should_visualize'] = False
    training_config['enable_tensorboard'] = False
    training_config['console_log_freq'] = None
    training_config['checkpoint_freq'] = None

    def to_GBs(memory_in_bytes):  # beautify memory output - helper function
        return f'{memory_in_bytes / 2**30:.2f} GBs'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        available_gpu_memory = torch.cuda.get_device_properties(device).total_memory
        print(f'Available memory on default GPU: {to_GBs(available_gpu_memory)}')
    else:
        print('GPU not available. :(')

    gat_layer_implementations = [layer_type for layer_type in LayerType]

    # Iterate over all the available GAT implementations
    for gat_layer_imp in gat_layer_implementations:
        training_config['layer_type'] = gat_layer_imp  # modify the training config so as to use different imp

        ts = time.time()
        print('*' * 20)
        print(f'Starting {gat_layer_imp.name} GAT training.')
        train_gat(training_config)
        print(f'Layer type = {gat_layer_imp.name}, training duration = {time.time()-ts:.2f} [s]')

        # These 2 methods basically query this function: torch.cuda.memory_stats(), which contains much more detail.
        # Here I just care about the peak memory usage i.e. whether you can train GAT on your device.

        # The actual number of GPU bytes needed to store the GPU tensors I use (since the start of the program)
        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        # The above + the caching GPU memory used by PyTorch's caching allocator (since the start of the program)
        max_memory_reserved = torch.cuda.max_memory_reserved(device)
        # Reset the peaks so that we get correct results for the next GAT implementation. Otherwise, since the above
        # methods are measuring the peaks since the start of the program one of the less efficient (memory-wise)
        # implementations may eclipse the others.
        torch.cuda.reset_peak_memory_stats(device)

        print(f'Max mem allocated = {to_GBs(max_memory_allocated)}, max mem reserved = {to_GBs(max_memory_reserved)}.')


def visualize_embedding_space(model_name = r'gat_000000.pth', dataset_name = DatasetType.CORA.name):
    """
    Using t-SNE to visualize GAT embeddings in 2D space.
    Check out this one for more intuition on how to tune t-SNE: https://distill.pub/2016/misread-tsne/

    If you think it'd be useful for me to implement t-SNE as well and explain how every single detail works
    open up an issue or DM me on social media!

    Note: I also tried using UMAP but it doesn't provide any more insight than t-SNE.
    (con: it has a lot of dependencies if you want to use their plotting functionality)

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # Step 1: Prepare the data
    node_features, node_labels, edge_index, train_indices, val_indices, test_indices = load_graph_data(dataset_name, layer_type=LayerType.IMP3, device=device, should_visualize=False)

    # Step 2: Prepare the model
    model_path = os.path.join(BINARIES_PATH, model_name)
    model_state = torch.load(model_path)

    gat = GAT(model_state['num_of_layers'], model_state['num_heads_per_layer'], model_state['num_features_per_layer']).to(device)
    print_model_metadata(model_state)
    gat.load_state_dict(model_state["state_dict"], strict=True)
    gat.eval()

    # Step 3: Run predictions
    with torch.no_grad():
        all_nodes_distributions, _ = gat((node_features, edge_index))

        all_nodes = all_nodes_distributions.cpu().numpy()
        node_labels = node_labels.cpu().numpy()
        num_classes = len(set(node_labels))

        perplexities = [30]
        t_sne_embeddings_list = [TSNE(n_components=2, perplexity=p, method='barnes_hut').fit_transform(all_nodes) for p in perplexities]

        # fig, axs = plt.subplots(3, 2)

        # plt.figure(figsize=(8, 8))

        # for cnt, t_sne_embeddings in enumerate(t_sne_embeddings_list):
        #     col = cnt % 2
        #     row = int(cnt / 2)
        for i in range(num_classes):
            plt.scatter(t_sne_embeddings_list[0][node_labels == i, 0], t_sne_embeddings_list[0][node_labels == i, 1], s=20, color=colors[i])
        # plt.axis('off')
        plt.show()


colors = [
    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
]


# todo: visualize attention
# todo: add env and README files
def visualize_attention():
    print('todo')


if __name__ == '__main__':
    # shape = (N, F), where N is the number of nodes and F is the number of features
    # node_features_csr = pickle_read(os.path.join(CORA_PATH, 'node_features.csr'))
    # profile_different_matrix_formats(node_features_csr)
    # visualize_embedding_space()
    profile_gat_implementations()
