from rdkit import Chem
import numpy as np
import warnings
warnings.filterwarnings("ignore")


datasets = ['human', 'celegans', 'Davis']
for dataset in datasets:
    positions = []
    sdf_input = "../datasets/interaction/{}/mols.sdf".format(dataset)
    mols = Chem.SDMolSupplier(sdf_input)
    for mol in mols:
        position = np.array(mol.GetConformer().GetPositions())
        positions.append(position)
    dir_output = "../datasets/interaction/{}/positions".format(dataset)
    np.save(dir_output, positions)
    print("succeeded !!!")
