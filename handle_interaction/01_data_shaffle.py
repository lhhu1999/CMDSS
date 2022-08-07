from utils import shuffle_dataset
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    datasets = ['human', 'celegans']

    for dataset in datasets:
        fpath = '../RawData/interaction/{}/data.txt'.format(dataset)

        with open(fpath, "r") as f:
            data_list = f.read().strip().split('\n')
        f.close()

        data_shuffle = shuffle_dataset(data_list)

        output = "../RawData/interaction/{}/data_shuffle.txt".format(dataset)
        with open(output, 'a') as f:
            for i in data_shuffle:
                f.write(str(i) + '\n')
            f.close()
    print("shuffle succeeded !!!")
