from rdkit import Chem
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")


atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
word_dict = defaultdict(lambda: len(word_dict))


def create_atoms(mol):     # 把不同类型原子改为数字序列替代
    atom = [a.GetSymbol() for a in mol.GetAtoms()]
    mark = np.zeros(len(atom))                # 记录非氢原子位置
    j = 0
    for c in atom:
        if c != 'H':
            mark[j] = 1
        j = j + 1
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atom[i] = (atom[i], 'aromatic')
    atom = [atom_dict[a] for a in atom]
    return np.array(atom), mark


def create_ijbonddict(mol):     # 生成每个点其对应点和化学键类型数字(j,bond)组合的字典 eg:{0:[(1,0),(35,0),(46,0)],1:[(3,0),[5,0]],2:[..]}
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))

    atoms_set = set(range(mol.GetNumAtoms()))       # 孤立原子边设为nan
    isolate_atoms = atoms_set - set(i_jbond_dict.keys())
    bond = bond_dict['nan']
    for a in isolate_atoms:
        i_jbond_dict[a].append((a, bond))
    return i_jbond_dict


def atom_features(atoms, i_jbond_dict):   # return the subgraph of each atom
    if len(atoms) == 1:
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(2):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])

            nodes = fingerprints
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict
    return np.array(fingerprints)


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i + ngram]]
             for i in range(len(sequence) - ngram + 1)]
    return np.array(words)


if __name__ == '__main__':
    # dataset = 'EC50', 'Kd', 'Ki', 'IC50'
    dataset = 'EC50'
    os.makedirs('../datasets/affinity/' + dataset + '/train', exist_ok=True)
    os.makedirs('../datasets/affinity/' + dataset + '/test', exist_ok=True)

    print(">>> Encode in " + dataset)
    for task in ['train', 'test']:
        smiles_all, skeletons, marks, residues, interactions = [], [], [], [], []
        file_input = "../RawData/affinity/{}/{}_shuffle.txt".format(dataset, task)
        path_output = "../datasets/affinity/{}/{}".format(dataset, task)
        data = pd.read_csv(file_input, sep=',', header=None)
        for i in tqdm(range(len(data[0]))):
            smile = data[0][i]
            sequence = data[1][i]
            interaction = data[2][i]

            mol = Chem.AddHs(Chem.MolFromSmiles(smile))
            atoms, mark = create_atoms(mol)
            i_jbond_dict = create_ijbonddict(mol)

            smiles_all.append(smile)
            marks.append(mark)
            skeletons.append(atom_features(atoms, i_jbond_dict))  # 返回每个点r范围内子图类型组成的序列 eg:[24,25,26,27,...]
            residues.append(split_sequence(sequence, 3))
            interactions.append(np.array([float(interaction)]))
        with open(path_output + '/smiles_all.txt', 'w') as f:
            for smile in smiles_all:
                f.write(str(smile) + '\n')
            f.close()
        np.save(os.path.join(path_output, 'skeletons'), skeletons)
        np.save(os.path.join(path_output, 'marks'), marks)
        np.save(os.path.join(path_output, 'residues'), residues)
        np.save(os.path.join(path_output, 'interactions'), interactions)

    with open('../datasets/affinity/' + dataset + '/drugs_dict', 'wb') as f:       # 保存子图字典
        pickle.dump(dict(fingerprint_dict), f)
    with open('../datasets/affinity/' + dataset + '/proteins_dict', 'wb') as f:       # 保存子图字典
        pickle.dump(dict(word_dict), f)
