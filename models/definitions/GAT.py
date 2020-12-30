import enum


import torch
import torch.nn as nn


# Support for 3 different GAT implementations - we'll profile each one of these
class LayerType(enum.Enum):
    IMP1 = 0,
    IMP2 = 1,
    IMP3 = 2,


def get_layer_type(layer_type):
    assert isinstance(layer_type, LayerType), f'Expected {LayerType} got {type(layer_type)}.'

    if layer_type == LayerType.IMP1:
        return GATLayerImp1
    elif layer_type == LayerType.IMP2:
        return GATLayerImp2
    elif layer_type == LayerType.IMP3:
        return GATLayerImp3
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')


class GAT(torch.nn.Module):
    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, dropout, layer_type=LayerType.IMP3):
        super().__init__()

        # Short names for readability (much shorter lines)
        nfpl = num_features_per_layer
        nhpl = num_heads_per_layer

        GATLayer = get_layer_type(layer_type)

        self.GAT = nn.Sequential(
            *[GATLayer(nfpl[i - 1], nfpl[i], nhpl[i], dropout=dropout) for i in range(1, num_of_layers - 1)],
            GATLayer(nfpl[-2], nfpl[-1], nhpl[-1], dropout=dropout, concat=False, activation=nn.Softmax)
        )

    def forward(self, graph, edge_index):
        return self.GAT(graph, edge_index)


class GATLayerImp3(torch.nn.Module):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric
    """

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU,
                 dropout_prob=0.6, add_self_loops=True, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_self_loops = add_self_loops

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper),
        # attention target/source (denoted as "a" in the paper) and bias (not mentioned in the paper)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives out a un-normalized score (e). Here we split the "a" vector - but the semantics remain the same.
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # Bias is not crucial to GAT method (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        #
        # End of trainable weights
        #

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)  # Used in 3 locations, feature matrix before/after proj and attention

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.
        """

        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        torch.nn.init.zeros_(self.bias)

    def forward(self, nodes_features_matrix, edge_index):
        # Shape = [N, FIN] where N - number of nodes in the graph, FIN number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        nodes_features_matrix = self.dropout(nodes_features_matrix)

        # Shape = [N, NH, FOUT] where NH - number of heads, FOUT number of output features per head
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_matrix_proj = self.linear_proj(nodes_features_matrix).view(-1, self.num_of_heads, self.num_out_features)

        scores_target = (nodes_features_matrix_proj * self.scoring_fn_target).sum(dim=-1)
        scores_source = (nodes_features_matrix_proj * self.scoring_fn_source).sum(dim=-1)

        if self.add_self_loops:  # Check out how Petar managed the data
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # expand this part
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)

        if self.log_attention_weights:
            print('todo')
            # self.attention_weights =

        # todo: add residual/skip connection

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return self.activation(out)


# Adapted from the official GAT implementation
class GATLayerImp2(torch.nn.Module):
    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU,
                 dropout=0.6, add_self_loops=True, bias=True, log_attention_weights=False):
        super().__init__()
        print('todo')


# Other
class GATLayerImp1(torch.nn.Module):
    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU,
                 dropout=0.6, add_self_loops=True, bias=True, log_attention_weights=False):
        super().__init__()
        print('todo')


