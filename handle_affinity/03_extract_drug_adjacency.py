from utils import save_adjacency
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    datasets = ['EC50', 'Kd', 'Ki', 'IC50']

    for data in datasets:
        print(">>> extract in " + data)
        dir_input = "../datasets/affinity/{}/smiles.txt".format(data)
        with open(dir_input, "r") as f:
            data_list = f.read().strip().split('\n')

        smiles = []
        for j in tqdm(data_list):
            smile = j.strip().split(' ')[0]
            smiles.append(smile)

        save_adjacency(smiles, data)
    print(">>>>>> extract succeeded !!!")
