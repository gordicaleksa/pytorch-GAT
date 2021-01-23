import torch
import torch.nn as nn


from utils.constants import LayerType


# todo: experiment with both transductive AND inductive settings
# todo: be explicit about shapes throughout the code
class GAT(torch.nn.Module):
    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, dropout=0.6, layer_type=LayerType.IMP3):
        super().__init__()

        # Short names for readability (much shorter lines)
        nfpl = num_features_per_layer
        nhpl = num_heads_per_layer

        GATLayer = get_layer_type(layer_type)

        self.gat_net = nn.Sequential(
            *[GATLayer(nfpl[i - 1], nfpl[i], nhpl[i-1], dropout_prob=dropout) for i in range(1, num_of_layers)],
            GATLayer(nfpl[-2], nfpl[-1], nhpl[-1], dropout_prob=dropout, concat=False, activation=nn.Softmax)
        )

    # data is just a (in_nodes_features, edge_index) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        return self.gat_net(data)


class GATLayerImp3(torch.nn.Module):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    """

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU,
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
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

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()

    def forward(self, data):
        in_nodes_features, edge_index = data
        num_of_nodes = in_nodes_features.shape[0]

        # shape = (N, FIN) where N - number of nodes in the graph, FIN number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, NH, FOUT) where NH - number of heads, FOUT number of output features per head
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH)
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply repeat the scores for source/target nodes based on the edge index
        # scores_x_lifted shape = (E, NH, 1), where E is the number of edges in the graph
        # nodes_features_proj_lifted shape = (E, NH, FOUT)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # todo: implement scatter softmax
        attentions_per_edge = self.scatter_softmax(scores_per_edge, edge_index[self.trg_nodes_dim])
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # todo: self-edges needed?
        # This part adds up weighted, projected neighborhoods for every target node
        out_nodes_features = torch.zeros(num_of_nodes, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        trg_index_broadcasted = self.broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(0, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        if self.log_attention_weights:
            self.attention_weights = attentions_per_edge

        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=1)

        if self.bias is not None:
            out_nodes_features += self.bias

        return self.activation(out_nodes_features)

    #
    # Helper functions
    #
    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        nodes_dim = 0
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        this = this.expand_as(other)
        return this

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
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


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


#
# Helper functions
#
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


