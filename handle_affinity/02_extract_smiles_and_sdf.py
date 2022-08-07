import os
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    datasets = ['EC50', 'Kd', 'Ki', 'IC50']

    for dataset in datasets:
        os.makedirs('../datasets/affinity/' + dataset, exist_ok=True)

        smiles = []
        mols = []
        file_input1 = "../RawData/affinity/{}/test_shuffle.txt".format(dataset)
        file_input2 = "../RawData/affinity/{}/train_shuffle.txt".format(dataset)
        with open(file_input1, 'r') as f:
            data1 = f.read().strip().split('\n')
            f.close()
        with open(file_input2, 'r') as f:
            data2 = f.read().strip().split('\n')
            f.close()
        data = data1 + data2
        for item in tqdm(data):
            smile = str(item).split(',')[0]
            if smile not in smiles:
                smiles.append(smile)
                mol = AllChem.AddHs(Chem.MolFromSmiles(smile))
                AllChem.EmbedMolecule(mol, randomSeed=1234)
                mols.append(mol)
        with open('../datasets/affinity/'+ dataset +'/smiles.txt', 'w') as f:
            for smile in smiles:
                f.write(str(smile) + '\n')
            f.close()

        # the mols.sdf contains atomic space coordinates
        w = Chem.SDWriter('../datasets/affinity/' + dataset + '/mols.sdf')
        for m in mols:
            w.write(m)
    print(">>> succeed !!!")
