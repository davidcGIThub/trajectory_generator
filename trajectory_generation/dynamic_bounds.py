import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
        
@dataclass
class TurningBound:
    max_curvature: float = None
    max_angular_rate: float = None
    max_centripetal_acceleration: float = None

    def checkIfTurningBoundActive(self):
        if self.max_curvature is not None or self.max_angular_rate is not None or \
            self.max_centripetal_acceleration is not None:
            return True
        else:
            return False
        

@dataclass
class DynamicBounds:
    turning_bound: TurningBound = None
    max_velocity: float = None
    max_acceleration: float = None

    def checkIfDerivativesActive(self):
        if self.max_velocity is not None or self.max_acceleration is not None:
            return True
        else:
            return False