from utils import shuffle_dataset
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    datasets = ['EC50', 'Kd']

    for dataset in datasets:
        for item in ['train', 'test']:
            fpath = '../RawData/affinity/{}/{}.txt'.format(dataset, item)

            with open(fpath, "r") as f:
                data_list = f.read().strip().split('\n')
            f.close()

            data_shuffle = shuffle_dataset(data_list)

            output = "../RawData/affinity/{}/{}_shuffle.txt".format(dataset, item)
            with open(output, 'w') as f:
                for i in data_shuffle:
                    f.write(str(i) + '\n')
                f.close()
    print("shuffle succeeded !!!")
