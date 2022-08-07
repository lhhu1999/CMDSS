import pandas as pd
import numpy as np
import os
from math import sqrt
from config import Config
from sklearn.metrics import roc_auc_score, precision_score, recall_score
config = Config()


def load_data_affinity(task, flag):
    smiles_all_input = "./datasets/affinity/{}/{}/smiles_all.txt".format(task, flag)
    smiles_all = pd.read_csv(smiles_all_input, header=None)[:][0].to_list()
    smiles_all = np.array(smiles_all)

    smiles_input = "./datasets/affinity/{}/smiles.txt".format(task)
    smiles = pd.read_csv(smiles_input, sep=' ', header=None)[:][0].to_list()

    positions_input = "./datasets/affinity/{}/positions.npy".format(task)
    pos = np.load(positions_input, allow_pickle=True)

    adjacency_input = "./datasets/affinity/{}/adjacency.npy".format(task)
    adj = np.load(adjacency_input, allow_pickle=True)

    positions = []
    adjacency = []
    for smile in smiles_all:
        index = smiles.index(smile)
        adjacency.append(adj[index])
        positions.append(pos[index])
    adjacency = np.array(adjacency)
    positions = np.array(positions)

    dir_input = "./datasets/affinity/{}/{}/".format(task, flag)
    skeletons = np.load(dir_input + 'skeletons.npy', allow_pickle=True)
    marks = np.load(dir_input + 'marks.npy', allow_pickle=True)
    residues = np.load(dir_input + 'residues.npy', allow_pickle=True)
    interactions = np.load(dir_input + 'interactions.npy', allow_pickle=True)
    data_pack = [skeletons, adjacency, marks, positions, residues, interactions]

    return data_pack


def load_data_interaction(task):
    smiles_all_input = "./datasets/interaction/{}/smiles_all.txt".format(task)
    smiles_all = pd.read_csv(smiles_all_input, header=None)[:][0].to_list()
    smiles_all = np.array(smiles_all)

    smiles_input = "./datasets/interaction/{}/smiles.txt".format(task)
    smiles = pd.read_csv(smiles_input, sep=' ', header=None)[:][0].to_list()

    positions_input = "./datasets/interaction/{}/positions.npy".format(task)
    pos = np.load(positions_input, allow_pickle=True)

    adjacency_input = "./datasets/interaction/{}/adjacency.npy".format(task)
    adj = np.load(adjacency_input, allow_pickle=True)

    positions = []
    adjacency = []
    for smile in smiles_all:
        index = smiles.index(smile)
        adjacency.append(adj[index])
        positions.append(pos[index])
    adjacency = np.array(adjacency)
    positions = np.array(positions)

    dir_input = "./datasets/interaction/{}/".format(task)
    skeletons = np.load(dir_input + 'skeletons.npy', allow_pickle=True)
    marks = np.load(dir_input + 'marks.npy', allow_pickle=True)
    residues = np.load(dir_input + 'residues.npy', allow_pickle=True)
    interactions = np.load(dir_input + 'interactions.npy', allow_pickle=True)
    data_pack = [skeletons, adjacency, marks, positions, residues, interactions]

    return data_pack


def split_data(train_data, ratio):
    idx = np.arange(len(train_data[0]))
    num = int(len(train_data[0])/config.batch_size * (1-ratio))
    num_train = num * config.batch_size
    idx_train, idx_dev = idx[:num_train], idx[num_train:]
    data_train = [train_data[di][idx_train] for di in range(len(train_data))]
    data_dev = [train_data[di][idx_dev] for di in range(len(train_data))]
    return data_train, data_dev


def result_affinity(total_loss, label, pred):
    LOSS = np.mean(total_loss)
    LOSS = float(format(LOSS, '.4f'))
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    rmse = sqrt(((label - pred)**2).mean(axis=0))
    pearson = np.corrcoef(label, pred)[0, 1]
    return round(rmse, 4), round(pearson, 4), LOSS


def save_evaluation_affinity(task, item, res):
    dir_output = "./output/affinity/{}".format(task)
    os.makedirs(dir_output, exist_ok=True)
    file = "./output/affinity/{}/{}.txt".format(task, item)
    with open(file, 'a') as f:
        f.write('\t'.join(map(str, res)) + '\n')


def get_kfold_data(i_fold, datasets, K_FOLD):
    trainsets = []
    validsets = []
    i_fold = i_fold - 1
    fold_size = len(datasets[0]) // K_FOLD
    val_start = i_fold * fold_size
    for i in range(len((datasets))):
        dataset = datasets[i]
        if i_fold == 0:
            val_end = fold_size
            validset = dataset[val_start:val_end]
            trainset = dataset[val_end:]
        elif i_fold != K_FOLD-1 and i_fold != 0:
            val_end = (i_fold + 1) * fold_size
            validset = dataset[val_start:val_end]
            if i == 5:
                trainset = np.vstack((dataset[0:val_start], dataset[val_end:]))
            else:
                trainset = np.hstack((dataset[0:val_start],  dataset[val_end:]))
        else:
            validset = dataset[val_start:]
            trainset = dataset[0:val_start]
        trainsets.append(trainset)
        validsets.append(validset)
    return trainsets, validsets


def result_interaction(total_loss, pred_labels, predictions, labels):
    LOSS = np.mean(total_loss)
    predictions = np.array(predictions)
    labels = np.array(labels)

    AUC = roc_auc_score(labels, predictions)
    Pre = precision_score(labels, pred_labels)
    Recall = recall_score(labels, pred_labels)


    AUC = float(format(AUC, '.4f'))
    Pre = float(format(Pre, '.4f'))
    Recall = float(format(Recall, '.4f'))
    LOSS = float(format(LOSS, '.4f'))
    return AUC, Pre, Recall, LOSS


def save_evaluation_interaction(task, item, k, res):
    dir_output = "./output/interaction/{}".format(task)
    os.makedirs(dir_output, exist_ok=True)
    file = "./output/interaction/{}/{}_fold_{}.txt".format(task, k, item)
    with open(file, 'a') as f:
        f.write('\t'.join(map(str, res)) + '\n')
