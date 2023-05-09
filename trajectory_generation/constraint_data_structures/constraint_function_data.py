import numpy as np
from dataclasses import dataclass

        
@dataclass
class ConstraintFunctionData:
    constraint_function: callable
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    key: np.ndarray = None
