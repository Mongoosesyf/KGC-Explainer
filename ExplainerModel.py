import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NodeExplainerModule(nn.Module):
    # Class inner variables
    loss_coef = {
        "g_size": 0.05,
        "feat_size": 1.0,
        "g_ent": 0.1,
        "feat_ent": 0.1
    }

    def __init__(self,
                 num_edges,
                 activation='sigmoid',
                 agg_fn='sum',
                 mask_bias=False):
        super(NodeExplainerModule, self).__init__()
        self.num_edges = num_edges
        self.activation = activation
        self.agg_fn = agg_fn
        self.mask_bias = mask_bias

        # Initialize parameters on masks
        self.edge_mask, self.edge_mask_bias = self.create_edge_mask(self.num_edges)

    def create_edge_mask(self, num_edges, init_strategy='normal', const=1.):
        mask = nn.Parameter(torch.Tensor(num_edges, 1))

        if init_strategy == 'normal':
            std = nn.init.calculate_gain("relu") * math.sqrt(
                1.0 / num_edges
            )
            print("std=", std)
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "const":
            nn.init.constant_(mask, const)

        if self.mask_bias:
            mask_bias = nn.Parameter(torch.Tensor(num_edges, 1))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias


    def forward(self):
        edge_mask = self.edge_mask.sigmoid()
        print("\nedge mask: ", edge_mask[:20])

        return edge_mask

    def _loss(self, shared_neighbor_weight, similar_neighbor_weight, KGC_score_weight):
        edge_mask = self.edge_mask
        shared_neighbor_weight = shared_neighbor_weight.reshape((edge_mask.size()[0], 1))
        shared_edge = edge_mask * shared_neighbor_weight
        shared_neighbor_loss = 1 / torch.sum(shared_edge)
        shared_neighbor_loss = -1 * torch.log(torch.sum(shared_edge))
        non_zero_idx = torch.nonzero(shared_edge, as_tuple=False)
        non_zero_idx = non_zero_idx[:, 0]
        non_zero_weight = torch.index_select(shared_edge, dim=0, index=non_zero_idx.squeeze())
        shared_neighbor_loss = -torch.log(torch.prod(non_zero_weight, 0))

        # similar_neighbor_loss
        similar_neighbor_weight = similar_neighbor_weight.reshape((edge_mask.size()[0], 1))
        similar_edge = edge_mask * similar_neighbor_weight、
        non_zero_idx = torch.nonzero(similar_edge, as_tuple=False)
        non_zero_idx = non_zero_idx[:, 0]、
        non_zero_weight = torch.index_select(similar_edge, dim=0, index=non_zero_idx.squeeze())
        similar_neighbor_loss = -torch.log(torch.prod(non_zero_weight, 0))

        KGC_score_weight = KGC_score_weight.reshape((edge_mask.size()[0], 1))
        KGC_score_edge = edge_mask * KGC_score_weight
        non_zero_idx = torch.nonzero(KGC_score_edge, as_tuple=False)
        non_zero_idx = non_zero_idx[:, 0]
        non_zero_weight = torch.index_select(KGC_score_edge, dim=0, index=non_zero_idx.squeeze())
        KGC_score_loss = -torch.log(torch.prod(non_zero_weight, 0))

        total_loss = shared_neighbor_loss + similar_neighbor_loss + KGC_score_loss
        
        return total_loss
