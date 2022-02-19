import numpy as np


def getID(folder='data/ConceptNet/'):
    lstEnts = {}
    lstRels = {}
    with open(folder + 'train.txt') as f, open(folder + 'train_new.txt', 'w') as f2:
        count = 0
        for line in f:
            line = line.strip().split('\t')
            line = [i.strip() for i in line]
            #print(line[0])
            if line[1] not in lstEnts:
                lstEnts[line[1]] = len(lstEnts)
            if line[0] not in lstRels:
                lstRels[line[0]] = len(lstRels)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            count += 1
            f2.write(str(line[1]) + '\t' + str(line[0]) +
                     '\t' + str(line[2]) + '\n')
        #print(lstEnts) # 字典保存每个实体或关系的索引
        print("Size of train_marked set set ", count)

    with open(folder + 'valid.txt') as f, open(folder + 'valid_new.txt', 'w') as f2:
        count = 0
        for line in f:
            line = line.strip().split('\t')
            line = [i.strip() for i in line]
            # print(line[0], line[1], line[2])
            if line[1] not in lstEnts:
                lstEnts[line[1]] = len(lstEnts)
            if line[0] not in lstRels:
                lstRels[line[0]] = len(lstRels)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            count += 1
            f2.write(str(line[1]) + '\t' + str(line[0]) +
                     '\t' + str(line[2]) + '\n')
        print("Size of VALID_marked set set ", count)

    with open(folder + 'test.txt') as f, open(folder + 'test_new.txt', 'w') as f2:
        count = 0
        for line in f:
            line = line.strip().split('\t')
            line = [i.strip() for i in line]
            # print(line[0], line[1], line[2])
            if line[1] not in lstEnts:
                lstEnts[line[1]] = len(lstEnts)
            if line[0] not in lstRels:
                lstRels[line[0]] = len(lstRels)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            count += 1
            f2.write(str(line[1]) + '\t' + str(line[0]) +
                     '\t' + str(line[2]) + '\n')
        print("Size of test_marked set set ", count)

    wri = open(folder + 'entity2id.txt', 'w')
    for entity in lstEnts:
        wri.write(entity + '\t' + str(lstEnts[entity]))
        wri.write('\n')
    wri.close()

    wri = open(folder + 'relation2id.txt', 'w')
    for entity in lstRels:
        wri.write(entity + '\t' + str(lstRels[entity]))
        wri.write('\n')
    wri.close()

getID()
