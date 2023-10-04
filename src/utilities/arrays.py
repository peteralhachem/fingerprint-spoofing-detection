
def vcol(_1D_array):
    """
    Reshape a 1-D array into a 2-D **column** array
    :param _1D_array: 1-D Numpy array to be reshaped
    :return: the 2-D Numpy **column** array
    """
    return _1D_array.reshape((_1D_array.size, 1))

def vrow(_1D_array):
    """
    Reshape a 1-D array into a 2-D **row** array
    :param _1D_array: 1-D Numpy array to be reshaped
    :return: the 2-D Numpy **row** array
    """
    return _1D_array.reshape((1, _1D_array.size))

def int_range(num, k):
    """
    Divide an integer number into k integer addends of the same size
    :param num: integer number to be split
    :param k: number of splits
    :return: array of integers being x=num//k, or x+1 if num is not exactly divisible by k

    Examples:

    >>> int_range(10, 5)
    >>> [2, 2, 2, 2, 2]
    >>> int_range(10, 4)
    >>> [3, 3, 2, 2]
    """
    equal_num_elements_per_partition = num // k
    partitions_sizes = [equal_num_elements_per_partition for _ in range(k)]

    num_partitions_with_one_more_element = num % k
    for i in range(num_partitions_with_one_more_element):
        partitions_sizes[i] += 1

    return partitions_sizes

def flat_map(func, list_of_lists):
    flat_list = []
    for x in list_of_lists:
        flat_list.extend(func(x))
    return flat_list
