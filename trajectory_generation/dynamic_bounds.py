import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
        
@dataclass
class TurningBound:
    max_turning_bound: float = None
    bound_type: str = None # "angular_rate", "curvature", "centripetal_acceleration"

    def checkIfTurningBoundActive(self):
        if self.max_turning_bound is not None:
            return True
        else:
            return False
        

@dataclass
class DerivativeBounds:
    max_velocity: float = None
    max_acceleration: float = None
    gravity: float = None
    max_upward_velocity: float = None
    max_downward_velocity: float = None

    def __post_init__(self):
        if self.max_upward_velocity is not None:
            if self.max_velocity is None:
                raise Exception("To set max upward velocity you need a general max velocity")
            if self.max_velocity < self.max_upward_velocity:
                raise Exception("Max upward velocity should be less than general max velocity")



    def checkIfDerivativesActive(self):
        if self.max_velocity is not None or self.max_acceleration is not None:
            return True
        else:
            return False