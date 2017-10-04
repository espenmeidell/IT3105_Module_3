from typing import Callable, List, Any
import numpy as np

Case = List[Any]


class CaseManager:
    def __init__(self
                 , case_function: Callable[[], List[Case]]
                 , validation_fraction: float = 0
                 , testing_fraction: float = 0):
        self.validation_fraction = validation_fraction
        self.testing_fraction = testing_fraction
        self.training_fraction = 1 - (validation_fraction + testing_fraction)

        self.cases = np.array(case_function())
        np.random.shuffle(self.cases)
        

