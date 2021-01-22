import time


from utils.load_data import normalize_features_sparse, normalize_features_dense


def profile_different_matrix_formats(node_features_csr):
    """
        Show the benefit of using CORRECT sparse formats during processing, result:
            CSR >> LIL >> dense format, CSR is ~2x faster from LIL and ~8x faster from dense format

        Official GAT and GCN implementations used LIL. In this particular case it doesn't matter that much,
        since it only takes a couple of ms to process Cora, but it's good to be aware of advantages that
        different sparse formats bring to the table.

    """
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


def profiling():
    print('todo profile 3 different implementations: time and memory-wise.')


if __name__ == '__main__':
    profile_different_matrix_formats()

