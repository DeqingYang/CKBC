import torch
import os
import numpy as np


def read_entity_from_id(filename='./data/ConceptNet/entity2id.txt'):
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().rsplit('\t', 1)) > 1:
                entity, entity_id = line.strip().rsplit('\t', 1)[0].strip(), line.strip().rsplit('\t', 1)[1].strip()
                #print(line.strip().rsplit('\t', 1))
                entity2id[entity] = int(entity_id)
    return entity2id


def read_relation_from_id(filename='./data/ConceptNet/relation2id.txt'):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id

# 对conceptnet的初始向量可以随机初始化
def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            #print(len(line.strip().split())) # 100维向量
            entity_emb.append([float(val) for val in line.strip().split()])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split()])

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)


def parse_line(line):
    line = line.strip().split('\t')
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2


def load_data(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    rows, cols, data = [], [], []
    unique_entities = set()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))
        if not directed:
                # Connecting source and tail entity
            rows.append(entity2id[e1])
            cols.append(entity2id[e2])
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation2id[relation])

        # Connecting tail and source entity
        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])

    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data), list(unique_entities)


def build_data(path='./data/ConceptNet/', is_unweigted=False, directed=True):
    entity2id = read_entity_from_id(path + 'new_entity2id.txt')
    relation2id = read_relation_from_id(path + 'new_relation2id.txt')

    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(
        path, 'train_1112.txt'), entity2id, relation2id, is_unweigted, directed)
    validation_triples, valid_adjacency_mat, unique_entities_validation = load_data(
        os.path.join(path, 'valid_1112.txt'), entity2id, relation2id, is_unweigted, directed)
    test_triples, test_adjacency_mat, unique_entities_test = load_data(os.path.join(
        path, 'test_1112.txt'), entity2id, relation2id, is_unweigted, directed)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    left_entity, right_entity = {}, {}

    with open(os.path.join(path, 'train_1112.txt')) as f:
        lines = f.readlines()

    for line in lines:
        e1, relation, e2 = parse_line(line)

        # Count number of occurences for each (e1, relation)
        # 计数对于每个关系的每个左实体的个数
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1

        # Count number of occurences for each (relation, e2)
        # 计数对于每个关系的每个右实体的个数
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1

    # 计算每个关系左实体出现的平均个数
    left_entity_avg = {}
    for i in range(2, len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])

    # 计算每个关系右实体出现的平均个数
    right_entity_avg = {}
    for i in range(2, len(relation2id)):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i])

    # 计算每个关系右实体出现的比例
    headTailSelector = {}
    for i in range(2, len(relation2id)):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])

    return (train_triples, train_adjacency_mat), (validation_triples, valid_adjacency_mat), (test_triples, test_adjacency_mat), \
        entity2id, relation2id, headTailSelector, unique_entities_train


#init_embeddings('./data/FB15k-237/entity2vec.txt', './data/FB15k-237/relation2vec.txt')
#read_entity_from_id('./data/ConceptNet/entity2id.txt')