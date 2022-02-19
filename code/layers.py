import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable


CUDA = torch.cuda.is_available()


class ConvKB_Bert(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky, num_relation):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU(inplace = True)
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        self.num_relation = num_relation
        self.bert_dim = 1024
        self.w_relation = torch.nn.Embedding(self.num_relation, self.bert_dim, padding_idx=0)
        nn.init.xavier_normal_(self.w_relation.weight.data)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, head_entity_embs, tail_entity_embs, relation_embeddings, batch_inputs):
        relation_embs = torch.cat([relation_embeddings[batch_inputs[:, 1]], self.w_relation(batch_inputs[:, 1])], dim=1)
        conv_input = torch.cat((head_entity_embs.unsqueeze(1), relation_embs.unsqueeze(1), tail_entity_embs.unsqueeze(1)), dim=1)

        #del relation_embeddings
        #del head_entity_embs
        #del relation_embs
        #del tail_entity_embs

        #print(head_entity_embs.shape) torch.Size([5248, 1224])
        #print(tail_entity_embs.shape) torch.Size([5248, 1224])
        #print(relation_embs.shape) torch.Size([5248, 1224])
        #print(conv_input.shape) torch.Size([5248, 3, 1224])

        batch_size, _, _ = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)
        #print(conv_input.shape) torch.Size([5248, 1, 1224, 3])

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))
        #del conv_input
        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        #del out_conv
        #print(input_fc.shape) torch.Size([5248, 6120])
        output = self.fc_layer(input_fc)
        #del input_fc
        #print(output.shape) torch.Size([5248, 1])
        return output



class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output



class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features])) # 创建稀疏矩阵
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim

        self.a = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features + nrela_dim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge, edge_embed, edge_list_nhop, edge_embed_nhop):
        N = input.size()[0] #实体嵌入
        #print(N) # 78334

        # Self-attention on the nodes - Shared attention mechanism
        edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1) #该实体和邻居实体
        #print(edge.shape) # [2, 273184]
        edge_embed = torch.cat(
            (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0) #该关系和邻居关系
        #print(edge_embed.shape) [273184, 200]

        edge_h = torch.cat(
            (input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim=1).t() #
        #print(edge_h.shape) #[600, 273184]
        # edge_h: (2*in_dim + nrela_dim) x E

        edge_m = self.a.mm(edge_h) # 论文中C_ijk
        # print(edge_m.shape) # [100, 273184]
        # edge_m: D * E

        # to be checked later
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze()) # 论文中b_ijk
        # print(powers.shape) # [273184]
        #print(powers)
        edge_e = torch.exp(powers).unsqueeze(1) # # 论文中a_ijk的分子
        #print(edge_e.shape) # [273184, 1]
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.shape[0], 1) #
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum ## 论文中a_ijk的分母
        # print(e_rowsum.shape) #[78334, 1]
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # print(edge_e.shape) #[273184]
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        #print(edge_w.shape) #[273184, 100]

        # edge_w: E * D
        # attention_value = edge_e.div(e_rowsum)

        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.out_features)
        #print(h_prime.shape) # [78334, 100]

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out

        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
