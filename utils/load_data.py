import enum
# nobody except for PyTorch Geometric did coalescing of the edge index...

# todo: link how the Cora was created and what the feature vectors represent
# todo: Visualize Cora (TikZ, networkx or whatever graph package)
# todo: compare across 5 different repos how people handled Cora

# todo: understand csr, coo, lil sparse matrix formats

# Currently I only have support for Cora - feel free to add your own data
# GAT and many other papers used the processed version of Cora that can be found here:
# https://github.com/kimiyoung/planetoid
class DatasetType(enum.Enum):
    CORA = 0


def load_data(dataset_name):
    if dataset_name == DatasetType.CORA:
        print('todo')

        # todo: adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)) <- test this one how does it handle duplicates
        # no coalescing
    else:
        raise Exception(f'{dataset_name} not yet supported.')
