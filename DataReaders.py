from typing import List, Any
import numpy as np
from functools import partial
import examples.tflowtools as tft

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

    if normalize_parameters:
        parameters = parameters / parameters.max(axis=0)

    if one_hot_vector_target:
        cases = []
        class_indices = np.unique(classes)
        target_vector_length = class_indices
        for i in range(len(parameters)):
            target = [0] * target_vector_length
            target[np.where(class_indices == classes[i])] = 1
            cases.append([parameters[i].tolist(), target])

        return cases

    return list(np.concatenate((parameters, classes), axis=1).tolist())


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


def count():
    return tft.gen_vector_count_cases(num=500, size=15)

