import torch
# from torchviz import make_dot, make_dot_from_trace
from models import SpKBGATModified, SpKBGATConvOnly
from layers import ConvKB
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch import Corpus

import torch
import dgl
import itertools
from scipy.sparse import coo_matrix

torch.set_printoptions(profile="full")

import random
import argparse
import os
import logging
import time
import pickle


CUDA = torch.cuda.is_available()

gat_loss_func = nn.MarginRankingLoss(margin=0.5)


def GAT_Loss(train_indices, valid_invalid_ratio):
    len_pos_triples = train_indices.shape[0] // (int(valid_invalid_ratio) + 1)

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(valid_invalid_ratio), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=2, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=2, dim=1)

    y = torch.ones(int(args.valid_invalid_ratio)
                   * len_pos_triples).cuda()
    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def render_model_graph(model, Corpus_, train_indices, relation_adj, averaged_entity_vectors):
    graph = make_dot(model(Corpus_.train_adj_matrix, train_indices, relation_adj, averaged_entity_vectors),
                     params=dict(model.named_parameters()))
    graph.render()


def print_grads(model):
    print(model.relation_embed.weight.grad)
    print(model.relation_gat_1.attention_0.a.grad)
    print(model.convKB.fc_layer.weight.grad)
    for name, param in model.named_parameters():
        print(name, param.grad)


def clip_gradients(model, gradient_clip_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, "norm before clipping is -> ", param.grad.norm())
            torch.nn.utils.clip_grad_norm_(param, args.gradient_clip_norm)
            print(name, "norm beafterfore clipping is -> ", param.grad.norm())


def plot_grad_flow(named_parameters, parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in zip(named_parameters, parameters):
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="r")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="g")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="g", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('initial.png')
    plt.close()


def plot_grad_flow_low(named_parameters, parameters):
    ave_grads = []
    layers = []
    for n, p in zip(named_parameters, parameters):
        # print(n)
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('initial.png')
    plt.close()

def save_model(model, name, epoch, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(),
               (folder_name + "trained_{}.pth").format(epoch))
    print("Done saving Model")


def get_adj_and_degrees(num_nodes, num_rels, triplets):
    """ Get adjacency list and degrees of the graph
    """

    col = []
    row = []
    rel = []
    adj_list = [[] for _ in range(num_nodes)]
    for i, triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])
        row.append(triplet[0])
        col.append(triplet[2])
        rel.append(triplet[1])
        row.append(triplet[2])
        col.append(triplet[0])
        rel.append(triplet[1] + num_rels)

    sparse_adj_matrix = coo_matrix((np.ones(len(triplets) * 2), (row, col)), shape=(num_nodes, num_nodes))

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees, sparse_adj_matrix, rel


def get_adj(triplets):
    """ Get adjacency list of the graph
    """

    col = []
    row = []
    rel = []
    for i, triplet in enumerate(triplets):
        row.append(triplet[2])
        col.append(triplet[0])
        rel.append(triplet[1])

    return (row, col, rel)


def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size, sample=True, sampling_edge_ids=None):
    """ Edge neighborhood sampling to reduce training graph size
    """

    if sample:
        edges = np.zeros((sample_size), dtype=np.int32)

        # initialize
        sample_counts = np.array([d for d in degrees])
        picked = np.array([False for _ in range(n_triplets)])
        seen = np.array([False for _ in degrees])
        i = 0

        while i != sample_size:
            weights = sample_counts * seen

            if np.sum(weights) == 0:
                weights = np.ones_like(weights)
                weights[np.where(sample_counts == 0)] = 0

            probabilities = weights / np.sum(weights)
            chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                             p=probabilities)
            chosen_adj_list = adj_list[chosen_vertex]
            seen[chosen_vertex] = True

            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

            while picked[edge_number]:
                chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
                chosen_edge = chosen_adj_list[chosen_edge]
                edge_number = chosen_edge[0]

            edges[i] = edge_number
            other_vertex = chosen_edge[1]
            picked[edge_number] = True
            sample_counts[chosen_vertex] -= 1
            sample_counts[other_vertex] -= 1
            seen[other_vertex] = True
            i += 1

    else:
        if sampling_edge_ids is None:
            random_edges = random.sample(range(n_triplets), sample_size)
        else:
            random_edges = np.random.choice(sampling_edge_ids, sample_size, replace=False)
        edges = np.array(random_edges)

    return edges


def generate_sampled_graph_and_labels(triplets, sample_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate, sim_sim=False, add_sim_relations=False,
                                      sim_train_e1_to_multi_e2=None, sampling_edge_ids=None):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """

    # perform edge neighbor sampling
    edges = sample_edge_neighborhood(adj_list, degrees, len(triplets),
                                     sample_size, sample=False, sampling_edge_ids=sampling_edge_ids)
    edges = triplets[edges]

    # add sim edges
    if add_sim_relations:
        edges = densify_subgraph(edges, num_rels, sim_train_e1_to_multi_e2)

    # connect neighbors of nodes connected by sim edges (not used)
    if sim_sim:
        edges = sim_sim_connect(edges, triplets, num_rels)

    src, rel, dst = edges.transpose()

    # relabel nodes to have consecutive node ids

    # uniq_v : sorted unique nodes in subsampled graph (original node ids)
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    # node ids now lie in range(0, number of unique nodes in subsampled graph)
    src, dst = np.reshape(edges, (2, -1))
    # relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Add inverse edges to training samples
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    if negative_rate == 0:
        samples = relabeled_edges
        labels = np.ones(len(samples))
    else:
        samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                            negative_rate)

    # build DGL graph
    print("# sampled nodes: {}".format(len(uniq_v)))
    print("# sampled edges: {}".format(len(src)))
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels


def densify_subgraph(edges, num_rels, sim_train_e1_to_multi_e2):
    sim_edges = []
    no_sim_indices = np.where(edges[:, 1] != num_rels - 1)[0]
    no_sim_edges = edges[no_sim_indices]
    unique, edges = np.unique((no_sim_edges[:, 0], no_sim_edges[:, 2]), return_inverse=True)
    for pair in itertools.combinations(unique, 2):
        if (pair[0], num_rels - 1) in sim_train_e1_to_multi_e2:
            if pair[1] in sim_train_e1_to_multi_e2[(pair[0], num_rels - 1)]:
                sim_edges.append(np.array([pair[0], num_rels - 1, pair[1]]))

    return np.concatenate((no_sim_edges, np.array(sim_edges)))


def comp_deg_norm(g):
    # 返回节点的入度
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm


def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """
        Create a DGL graph.
    """
    # 创建一个没有节点和边的空图
    g = dgl.DGLGraph()
    # 添加N个节点
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    # 添加多条边
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g, rel, norm


def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    print("Test graph:")
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))


def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    # TODO: pick negative samples only with same relations
    values = np.random.randint(num_entity, size=num_to_generate)
    # values = np.random.choice(tot_entities, size=num_to_generate, replace=False)

    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    # for i, sample in enumerate(neg_samples):
    #    if any(np.array_equal(sample, x) for x in pos_samples):
    #        labels[i+size_of_batch] = 1

    return np.concatenate((pos_samples, neg_samples)), labels


def sim_sim_connect(pos_samples, all_triplets, num_rels):
    """
    connect neighbors of node with sim edge type to a candidate node
    """

    # filter sim relations

    sample_ids = np.where(pos_samples[:, 1] == num_rels - 1)[0]
    sampled_sim_edges = pos_samples[sample_ids]
    addl_samples = []

    for edge in sampled_sim_edges:
        src, rel, tgt = edge
        # find all neighboring edges of tgt in large graph
        neighbors = np.where(all_triplets[:, 0] == tgt)[0]
        no_sim_neighbors = np.where(all_triplets[neighbors][:, 1] != num_rels - 1)[0]
        new_edges = np.copy(all_triplets[neighbors][no_sim_neighbors])
        new_edges[:, 0] = src
        addl_samples.append(new_edges)

    if len(addl_samples) == 0:
        return pos_samples
    else:
        addl_samples = np.concatenate(addl_samples)
        final_samples = np.concatenate((pos_samples, addl_samples))
        unique_samples = np.unique(final_samples, axis=0)
        print("Adding %d sim-sim edges" % (unique_samples.shape[0] - pos_samples.shape[0]))
        return unique_samples
