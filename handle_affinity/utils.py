import numpy as np
from rdkit import Chem
import warnings
warnings.filterwarnings("ignore")


def shuffle_dataset(dataset):
    np.random.seed(1234)
    np.random.shuffle(dataset)
    return dataset


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    adjacency = np.array(adjacency)
    adjacency += np.eye(adjacency.shape[0], dtype=int)    # change the diagonal to 1
    return adjacency


def save_adjacency(smiles, dataset):
    adjacency = []
    for smile in smiles:
        mol = Chem.AddHs(Chem.MolFromSmiles(smile))
        adjacency.append(create_adjacency(mol))

    dir_output = "../datasets/affinity/{}/adjacency".format(dataset)
    np.save(dir_output, adjacency)

