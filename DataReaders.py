from typing import List, Any
import numpy as np

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
        target_vector_length = int(classes.max())+1
        for i in range(len(parameters)):
            target = [0] * target_vector_length
            target[int(classes[i])] = 1
            cases.append([parameters[i].tolist(), target])

        return cases

    # TODO: split into inputs and targets
    return list(np.concatenate((parameters, classes), axis=1).tolist())


