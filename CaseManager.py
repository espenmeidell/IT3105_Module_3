from typing import Callable, List, Any
import numpy as np

# Type Aliases
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
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases) * self.validation_fraction)
        self.training_cases = self.cases[0:separator1]
        self.validation_cases = self.cases[separator1:separator2]
        self.testing_cases = self.cases[separator2:]

    def get_training_cases(self): return self.training_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.testing_cases


