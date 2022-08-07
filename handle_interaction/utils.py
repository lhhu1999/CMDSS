from rdkit import Chem
import numpy as np


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    adjacency = np.array(adjacency)
    adjacency += np.eye(adjacency.shape[0], dtype=int)    # 对角线都变成1 (diagonal set to 1)
    return adjacency


def save_adjacency(smiles, dataset):
    adjacency = []
    for smile in smiles:
        mol = Chem.AddHs(Chem.MolFromSmiles(smile))
        adjacency.append(create_adjacency(mol))
    file_output = "../datasets/interaction/{}/adjacency".format(dataset)
    np.save(file_output, adjacency)


def shuffle_dataset(dataset):
    np.random.seed(1234)
    np.random.shuffle(dataset)
    return dataset
