from rdkit import Chem
import numpy as np
import warnings
warnings.filterwarnings("ignore")


datasets = ['EC50', 'Kd', 'Ki', 'IC50']
for dataset in datasets:
    positions = []
    sdf_input = "../datasets/affinity/{}/mols.sdf".format(dataset)
    mols = Chem.SDMolSupplier(sdf_input)
    for mol in mols:
        position = np.array(mol.GetConformer().GetPositions())
        positions.append(position)
    dir_output = "../datasets/affinity/{}/positions".format(dataset)
    np.save(dir_output, positions)
    print("succeeded !!!")
