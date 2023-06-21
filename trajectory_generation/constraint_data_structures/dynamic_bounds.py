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
    
    def __post_init__(self):
        print("self.bound_type: " , self.bound_type)
        print("max_turning_bound: " , self.max_turning_bound)
        if self.bound_type == "centripetal_acceleration" or \
           self.bound_type == "curvature" or \
           self.bound_type == "angular_rate":
            pass
        else:
            raise Exception("Bound type must be either [angular_rate, curvature, centripetal_acceleration]")
        

@dataclass
class DerivativeBounds:
    max_velocity: float = None
    max_acceleration: float = None
    gravity: float = None
    max_upward_velocity: float = None
    max_horizontal_velocity: float = None

    def __post_init__(self):
        if self.max_upward_velocity is not None:
            if self.max_velocity is None:
                raise Exception("To set max upward velocity you need a general max velocity")
            if self.max_velocity < self.max_upward_velocity:
                raise Exception("Max upward velocity should be less than or equal to general max velocity")
        if self.max_horizontal_velocity is not None:
            if self.max_velocity is None:
                raise Exception("To set max horizontal velocity you need a general max velocity")
            if self.max_velocity < self.max_horizontal_velocity:
                raise Exception("Max horizontal velocity should be less than or equal to general max velocity")

    def checkIfDerivativesActive(self):
        if self.max_velocity is not None or self.max_acceleration is not None:
            return True
        else:
            return False