import torch
#from models import SpKBGATModified, SpKBGATConvOnly
from model_bert import SpKBGATModified, SpKBGATConvOnly
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from copy import deepcopy
from create_batch import Corpus
import utils

import random
import argparse
from collections import Counter
from reader import ConceptNetTSVReader
import reader_utils
import time
from bert_feature_extractor import BertLayer
import sys
import logging
import pickle


# %%
# %%from torchviz import make_dot, make_dot_from_trace

def save_model(model, name, epoch, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(),
               (folder_name + "trained_{}.pth").format(epoch))
    print("Done saving Model")

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-d", "--dataset", type=str, default="conceptnet",
                        help="dataset to use")
    args.add_argument("-data", "--data",
                      default="./data/ConceptNet/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3000, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=0.00001, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=0.000, help="L2 reglarization for conv") # 0.000001
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=200, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=True)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=True)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/cn/model_sim/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=222618, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=4, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=1, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=50,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=8,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.3, help="Dropout probability for convolution layer")

    args = args.parse_args()
    return args

args = parse_args()
print(args)
# %%

def build_data(dataset, reader_cls, data_dir, sim_relations):
    # 构建关于训练集验证集测试集的知识图
    train_network = reader_cls(dataset)
    dev_network = reader_cls(dataset)
    test_network = reader_cls(dataset)

    train_network.read_network(data_dir=data_dir, split="train")
    # 输出训练图谱的信息
    train_network.print_summary()
    node_list = train_network.graph.iter_nodes()
    node_degrees = [node.get_degree() for node in node_list]
    degree_counter = Counter(node_degrees)
    avg_degree = sum([k * v for k, v in degree_counter.items()]) / sum([v for k, v in degree_counter.items()])
    print("Average Degree: ", avg_degree)

    # 将测试图和验证图上的节点边加入到训练图中；只是将节点和关系加入到了其中，并没有增加新的边
    dev_network.read_network(data_dir=data_dir, split="valid", train_network=train_network)
    test_network.read_network(data_dir=data_dir, split="test", train_network=train_network)

    # 所有的节点ID
    entity2id = train_network.graph.node2id
    # 所有的关系ID
    relation2id = train_network.graph.relation2id
    unique_entities_train = train_network.unique_entities

    bert_model = BertLayer(dataset)

    # Add sim nodes
    if sim_relations:
        print("Adding sim edges..")
        train_network.add_sim_edges_bert(bert_model)

    train_network.print_summary()
    # 输出图谱
    train_triples = reader_utils.get_triple(entity2id, train_network, train_network)
    test_triples = reader_utils.get_triple(entity2id, test_network, train_network)
    valid_triples = reader_utils.get_triple(entity2id, dev_network, train_network)

    all_tuples = train_triples + valid_triples + test_triples

    # build adj list and calculate degrees for sampling
    train_adjacency_mat = utils.get_adj(train_triples)
    valid_adjacency_mat = utils.get_adj(test_triples)
    test_adjacency_mat = utils.get_adj(valid_triples)

    train_data = (train_triples, train_adjacency_mat)
    valid_data = (valid_triples, valid_adjacency_mat)
    test_data = (test_triples, test_adjacency_mat)

    return train_data, valid_data, test_data, entity2id, relation2id, train_network, unique_entities_train, bert_model, all_tuples


def load_data(args):
    train_data, validation_data, test_data, entity2id, relation2id, train_network, unique_entities_train, bert_model, all_tuples = build_data("conceptnet",
                                                                                            ConceptNetTSVReader,
                                                                                            "data/ConceptNet/",
                                                                                            True)

    all_e1_to_multi_e2, all_e2_to_multi_e1 = reader_utils.create_entity_dicts(all_tuples, len(train_network.graph.relations), True)

    train_data_no_sim, validation_data_no_sim, test_data_no_sim, _, relation2id_no_sim, train_network_no_sim, _, _, _ = build_data(
        "conceptnet",
        ConceptNetTSVReader,
        "data/ConceptNet/",
        False)
    corpus_no_sim = Corpus(args, train_data_no_sim, validation_data_no_sim, test_data_no_sim, entity2id, relation2id_no_sim, args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)

    entity_embeddings = np.random.randn(
            len(entity2id), args.embedding_size)
    relation_embeddings = np.random.randn(
            len(relation2id), args.embedding_size)
    print("Initialised relations and entities randomly")

    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)

    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings), corpus_no_sim, bert_model, train_network, train_data, test_data, all_e1_to_multi_e2, test_data_no_sim, train_network_no_sim

# 得到语料库，实体嵌入矩阵，关系嵌入矩阵
Corpus_, entity_embeddings, relation_embeddings, Corpus_no_sim, bert_model, train_network, train_data, test_data, all_e1_to_multi_e2, test_data_no_sim, train_network_no_sim = load_data(args)
print('Initialised Successfully')


# 存储2跳邻居的信息
#if(args.get_2hop):
#    file = args.data + "/2hop.pickle"
#    with open(file, 'wb') as handle:
#        pickle.dump(Corpus_.node_neighbors_2hop, handle,
#                    protocol=pickle.HIGHEST_PROTOCOL)


if(args.use_2hop):
    print("Opening node_neighbors pickle object")
    file = args.data + "/2hop.pickle"
    with open(file, 'rb') as handle:
        node_neighbors_2hop = pickle.load(handle) # 2跳邻居的信息


print("Initial entity dimensions {} , relation dimensions {}".format(
    entity_embeddings.size(), relation_embeddings.size())) # FB：【14541，100】，【237，100】
# %%

CUDA = torch.cuda.is_available()


def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def train_gat(args):

    # Creating the gat model here.
    ####################################

    print("Defining model")

    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT)

    if CUDA:
        model_gat.cuda()

    optimizer = torch.optim.Adam(
        model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)

    # 评价相似度的损失
    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

    # 当前batch的2跳邻居的切片
    current_batch_2hop_indices = torch.tensor([])
    if(args.use_2hop):
        current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args,
                                                                          Corpus_.unique_entities_train, node_neighbors_2hop)

    if CUDA:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices)).cuda()
    else:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices))

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_gat))

    for epoch in range(args.epochs_gat):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        # 获取一个epoch里的迭代次数
        if len(Corpus_.train_indices) % args.batch_size_gat == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_gat
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_gat) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            # forward pass
            entity_embed, relation_embed = model_gat(
                Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)

            optimizer.zero_grad()

            loss = batch_gat_loss(
                gat_loss_func, train_indices, entity_embed, relation_embed)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        if (epoch + 1) % 3000 == 0:
            save_model(model_gat, args.data, epoch,
                   args.output_folder)


def train_conv(args):

    # Creating convolution model here.
    ####################################

    print("Defining model")
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT)
    print("Only Conv model trained")
    model_conv = SpKBGATConvOnly(bert_model, entity_embeddings, relation_embeddings[:-1,:], args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)

    if CUDA:
        if torch.cuda.device_count() > 1:
            model_conv = nn.DataParallel(model_conv, device_ids=[0, 1])
        else:
            model_conv.cuda()
        model_gat.cuda()

    model_gat.load_state_dict(torch.load(
        '{}/trained_{}.pth'.format(args.output_folder, args.epochs_gat - 1)), strict=False)
    model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
    model_conv.final_relation_embeddings = nn.Parameter(model_gat.final_relation_embeddings[:-1,:])

    Corpus_no_sim.batch_size = args.batch_size_conv
    Corpus_no_sim.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)

    optimizer = torch.optim.Adam(
        model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss()

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_conv))

    for epoch in range(args.epochs_conv):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_no_sim.train_triples)
        Corpus_no_sim.train_indices = np.array(
            list(Corpus_no_sim.train_triples)).astype(np.int32)

        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_no_sim.train_indices) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(
                Corpus_no_sim.train_indices) // args.batch_size_conv
        else:
            num_iters_per_epoch = (
                len(Corpus_no_sim.train_indices) // args.batch_size_conv) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_no_sim.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            preds = model_conv(
                Corpus_no_sim, Corpus_no_sim.train_adj_matrix, train_indices)

            optimizer.zero_grad()

            #print(train_values) # 1和-1
            loss = margin_loss(preds.view(-1), train_values.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        if (epoch + 1) % 20 == 0:
            save_model(model_conv, args.data, epoch,
                   args.output_folder + "conv_bert/")


def evaluate_conv(args, unique_entities, train_network):
    model_conv = SpKBGATConvOnly(bert_model, entity_embeddings, relation_embeddings[:-1, :], args.entity_out_dim,
                                 args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)
    model_conv.load_state_dict(torch.load(
        '{0}conv_bert/trained_{1}.pth'.format(args.output_folder, args.epochs_conv - 1)), strict=False)

    if torch.cuda.device_count() > 1:
        model_conv = nn.DataParallel(model_conv, device_ids=[0,1])
    else:
        model_conv.cuda()

    model_conv.eval()
    with torch.no_grad():
        Corpus_no_sim.get_validation_pred_new(args, model_conv, unique_entities, train_network)


train_gat(args)
#train_conv(args)
#evaluate_conv(args, Corpus_no_sim.unique_entities_train, train_network_no_sim)
