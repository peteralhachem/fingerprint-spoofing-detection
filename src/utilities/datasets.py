import numpy as np
from src.utilities.arrays import int_range, vcol


def load_fingerprint_spoofing_detection_train_set(filepath="data/train.txt"):
    """
    Load the Fingerprint Spoofing Detection **train set** from its text file
    :param filepath: (optional) relative path of the file containing the dataset, starting from /src.
     If not provided, the default path is used. The file must contain one sample per row,
     each one with space-separated features values, and the label at the end (0/1)
    :return: a 2-D numpy array having one column for each sample and
     a 1-D numpy array containing the associated labels
    """
    return internal_load_binary_dataset("../../../%s" % filepath)

def load_fingerprint_spoofing_detection_test_set(filepath="data/test.txt"):
    """
    Load the Fingerprint Spoofing Detection **test set** from its text file
    :param filepath: (optional) relative path of the file containing the dataset, starting from /src.
     If not provided, the default path is used. The file must contain one sample per row,
     each one with space-separated features values, and the label at the end (0/1)
    :return: a 2-D numpy array having one column for each sample and
     a 1-D numpy array containing the associated labels
    """
    return internal_load_binary_dataset("../../../%s" % filepath)

def internal_load_binary_dataset(filename, num_dimensions=10):
    """
    Load a dataset from its text file
    :param filename: path of the file containing the dataset. It must contain one sample per row, each one with
     space-separated features values, and the label at the end (0/1)
    :param num_dimensions: (default: 10) number of dimensions of each sample in the file
    :return: a 2-D numpy array having one column for each sample and a 1-D numpy array containing the associated labels
    """
    data = []
    labels = []

    with open(filename, 'r') as file:
        for line in file:
            sample_fields = line.strip().split(",")

            sample_features = map(lambda feature: float(feature), sample_fields[0:num_dimensions])

            data.append(vcol(np.fromiter(sample_features, dtype=np.float64)))
            labels.append(int(sample_fields[num_dimensions]))

    data_matrix = np.hstack(data)
    labels = np.array(labels)

    return data_matrix, labels


def split_dataset(data_matrix, labels, first_set_ratio=2.0/3.0, shuffle=True, seed=0):
    """
    Split a dataset into 2 sets (ex: training set and cross_validation set),
    optionally shuffling first the dataset
    :param data_matrix: 2-D numpy array containing the dataset (1 sample per column)
    :param labels: 1-D numpy array containing the labels of the samples (1 label for each sample)
    :param first_set_ratio: (default: 2/3) the percentage of elements that will be in the first returned subset
    :param shuffle: (default: True) indicate if shuffling the dataset before the split
    :param seed: (default: 0) the integer seed to randomly shuffle data
    :return: two pairs of values, one per each split:
    (1) the first set split (2-D array) and the corresponding labels (1-D array);
        it will contain the specified percentage (first_set_ratio) of elements
    (2) the second set split (2-D array) and the corresponding labels (1-D array)
    """
    first_set_size = int(first_set_ratio * data_matrix.shape[1])
    # create an array of indexes
    indexes = np.arange(data_matrix.shape[1])

    if shuffle:
        np.random.seed(seed)
        indexes = np.random.permutation(indexes)    # shuffle indexes

    first_set_indexes = indexes[:first_set_size]
    second_set_indexes = indexes[first_set_size:]

    # extract first split
    first_set_data = data_matrix[:, first_set_indexes]
    first_set_labels = labels[first_set_indexes]

    # extract second split
    second_set_data = data_matrix[:, second_set_indexes]
    second_set_labels = labels[second_set_indexes]

    first_split = (first_set_data, first_set_labels)
    second_split = (second_set_data, second_set_labels)
    return first_split, second_split


def equal_partitions_of(dataset, labels, k, shuffle=False, seed=None):
    """
    Compute k equal partitions of a dataset, each containing an integer number of (ordered) elements
    :param dataset: 2-D Numpy array to be partitioned, containing one sample per column
    :param labels: 1-D Numpy array to be partitioned, containing one label per sample
    :param k: integer number of partitions
    :param shuffle: (default: False) specify if shuffling dataset before the split
    :param seed: (default: None) seed used to shuffle data; if None, data are shuffled by default with
         a different seed at each call
    :return: an array with the partitions (one 2-D Numpy array for each partition)
    and an array of the corresponding partitioned labels
    """
    num_samples = dataset.shape[1]
    partitions_sizes = int_range(num_samples, k)    # array

    dataset_partitions = []
    labels_partitions = []
    indexes = range(num_samples)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        indexes = np.random.permutation(indexes)

    start = 0

    for size in partitions_sizes:
        # extract a partition
        indexes_partition = indexes[start:start+size]

        dataset_partition = dataset[:, indexes_partition]
        labels_partition = labels[indexes_partition]

        dataset_partitions.append(dataset_partition)
        labels_partitions.append(labels_partition)

        start += size

    return dataset_partitions, labels_partitions
