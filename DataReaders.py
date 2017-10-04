from typing import List, Any
import numpy as np

# Type Aliases
Case = List[Any]


def read_numeric_file_with_class_in_final_column(path: str,
                                                 separator: str = ",",
                                                 normalize_parameters: bool = False) -> List[Any]:

    def process_line(line: str) -> List[float]:
        return list(map(float, line.split(separator)))

    file = open(path)
    data = np.array(list(map(process_line, file)))

    parameters = data[:, :data.shape[1]-1]
    classes = data[:, [-1]]

    if normalize_parameters:
        parameters = parameters / parameters.max(axis=0)

    return list(np.concatenate((parameters, classes), axis=1).tolist())


