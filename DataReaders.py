from typing import List, Any
import numpy as np
from functools import partial
import examples.tflowtools as tft
from mnist import mnist_basics
from pprint import pprint


# Type Aliases
Case = List[Any]


def read_numeric_file_with_class_in_final_column(path: str,
                                                 separator: str = ",",
                                                 normalize_parameters: bool = False,
                                                 one_hot_vector_target: bool = True) -> List[Any]:

    def process_line(line: str) -> List[float]:
        return list(map(float, line.split(separator)))

    file = open(path)
    data = np.array(list(map(process_line, file)))


    parameters = data[:, :data.shape[1]-1]
    classes = data[:, [-1]]

    means = np.mean(parameters, axis=0)
    std = np.std(parameters, axis=0)

    for i in range(len(parameters)):
        for j in range(len(parameters[i])):
            parameters[i][j] = (parameters[i][j] - means[j]) / std[j]

    if one_hot_vector_target:
        cases = []
        class_indices = np.unique(classes)
        target_vector_length = class_indices
        for i in range(len(parameters)):
            target = [0] * target_vector_length
            target[np.where(class_indices == classes[i])] = 1
            cases.append([parameters[i].tolist(), target])

        return cases

    assert False, "Only one hot vector support"


def parity():
    return tft.gen_all_parity_cases(num_bits=10)


def wine():
    return read_numeric_file_with_class_in_final_column(
                   path="data/winequality_red.txt",
                   separator=";",
                   normalize_parameters=True)


def yeast():
    return read_numeric_file_with_class_in_final_column(
                   path="data/yeast.txt",
                   separator=",",
                   normalize_parameters=False)


def glass():
    return read_numeric_file_with_class_in_final_column(
                   path="data/glass.txt",
                   separator=",",
                   normalize_parameters=True)


def bit_count():
    return tft.gen_vector_count_cases(num=500, size=15)


def seg_count():
    return tft.gen_segmented_vector_cases(25, 1000, 0, 8)


def auto():
    return tft.gen_all_one_hot_cases(2**3)


def mnist():
    cases = mnist_basics.load_all_flat_cases()
    features = np.array(cases[0])
    features = features / 255
    classes = cases[1]
    cases = []
    for i in range(len(classes)):
        one_hot = [0] * 10
        one_hot[classes[i]] = 1
        cases.append([features[i].tolist(), one_hot])
    return cases


def iris():
    return read_numeric_file_with_class_in_final_column(path="data/iris.txt")