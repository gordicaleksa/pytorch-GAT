import time
import os


import torch
import scipy.sparse as sp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utils.data_loading import normalize_features_sparse, normalize_features_dense, pickle_read, load_graph_data
from utils.constants import CORA_PATH, BINARIES_PATH, DatasetType
from models.definitions.GAT import GAT
from utils.utils import print_model_metadata


def profile_different_matrix_formats(node_features_csr):
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

    node_features_dense = node_features_csr.todense()
    node_features_lil = node_features_csr.tolil()

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


def profiling_different_gat_implementations():
    print('todo profile 3 different implementations: time and memory-wise.')


def visualize_embedding_space():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!
    gat_config = {
        "num_of_layers": 2,
        "num_heads_per_layer": [8, 1],
        "num_features_per_layer": [1433, 8, 7]  # todo: make a variable or dynamically figure out we need 7
    }
    model_name = r'gat_000000.pth'
    dataset_name = DatasetType.CORA.name

    # Step 1: Prepare the data
    node_features, node_labels, edge_index, train_indices, val_indices, test_indices = load_graph_data(dataset_name, device, should_visualize=False)

    # Step 2: Prepare the model
    gat = GAT(gat_config['num_of_layers'], gat_config['num_heads_per_layer'], gat_config['num_features_per_layer']).to(device)

    model_path = os.path.join(BINARIES_PATH, model_name)
    model_state = torch.load(model_path)
    print_model_metadata(model_state)
    gat.load_state_dict(model_state["state_dict"], strict=True)
    gat.eval()

    # Step 3: Run predictions
    with torch.no_grad():
        all_nodes_distributions, _ = gat((node_features, edge_index))

        # todo: t-SNE, UMAP for both trained and random GAT model
        t_sne_embeddings = TSNE(n_components=2).fit_transform(all_nodes_distributions.cpu().numpy())
        node_labels = node_labels.cpu().numpy()

        plt.figure(figsize=(8, 8))
        num_classes = len(set(node_labels))
        for i in range(num_classes):
            plt.scatter(t_sne_embeddings[node_labels == i, 0], t_sne_embeddings[node_labels == i, 1], s=20, color=colors[i])
        plt.axis('off')
        plt.show()


colors = [
    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
]


def visualize_attention():
    print('todo')


if __name__ == '__main__':
    # shape = (N, F), where N is the number of nodes and F is the number of features
    # node_features_csr = pickle_read(os.path.join(CORA_PATH, 'node_features.csr'))
    # profile_different_matrix_formats(node_features_csr)
    visualize_embedding_space()

