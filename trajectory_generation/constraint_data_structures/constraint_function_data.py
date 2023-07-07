import numpy as np
from dataclasses import dataclass
import numpy.typing as npt

@dataclass
class ConstraintFunctionData:
    constraint_function: callable
    lower_bound: npt.NDArray[np.float64]
    upper_bound: npt.NDArray[np.float64]
    key: npt.NDArray[np.dtype('U1')] = None
    constraint_class: str = None
    constraint_tolerance: float = 10e-6
# class ConstraintFunctionData:
#     constraint_function: callable
#     lower_bound: np.ndarray
#     upper_bound: np.ndarray
#     key: np.ndarray = None
#     constraint_class: str = None
#     constraint_tolerance: float = 10e-6

    def __post_init__(self):
        if self.constraint_class == "Derivative" or \
           self.constraint_class == "Obstacle" or \
           self.constraint_class == "Safe_Flight_Corridor" or \
           self.constraint_class == "Turning" or \
           self.constraint_class == "Start_Waypoint_Location" or \
           self.constraint_class == "End_Waypoint_Location" or \
           self.constraint_class == "Start_Waypoint_Derivatives" or \
           self.constraint_class == "End_Waypoint_Derivatives" or \
           self.constraint_class == "Start_Waypoint_Direction" or \
           self.constraint_class == "End_Waypoint_Direction" or \
           self.constraint_class == "Intermediate_Waypoint_Locations" or \
           self.constraint_class == "Intermediate_Waypoint_Velocities" or \
           self.constraint_class == "Zero_Velocity_End_Waypoint_Location" or \
           self.constraint_class == "Zero_Velocity_Start_Waypoint_Location":
            pass
        else:
            raise Exception("Constraint class [", self.constraint_class ,"] invalid")
        
    def get_output(self, optimized_result):
        return self.constraint_function(optimized_result)
        
    def get_violations(self, output):
        violations = np.logical_or(output>(self.upper_bound + self.constraint_tolerance),
                                       output < (self.lower_bound - self.constraint_tolerance))
        return violations
    
    def get_error(self, output):
        if self.constraint_class == "Derivative":
            return output
        if self.constraint_class == "Turning":
            return output
        if self.constraint_class == "Obstacle":
            return -output
        if self.constraint_class == "Safe_Flight_Corridor":
            return np.max((self.lower_bound - output, output - self.upper_bound),0)
        if self.constraint_class == "Start_Waypoint_Location" or \
           self.constraint_class == "End_Waypoint_Location":
            return np.abs(output - self.lower_bound)
        if self.constraint_class == "Start_Waypoint_Derivatives" or \
           self.constraint_class == "End_Waypoint_Derivatives":
            return np.abs(output - self.lower_bound)
        if self.constraint_class == "Start_Waypoint_Direction" or \
           self.constraint_class == "End_Waypoint_Direction":
            return np.abs(output - self.lower_bound)
        if self.constraint_class == "Intermediate_Waypoint_Locations":
            return np.abs(output - self.lower_bound)
        if self.constraint_class == "Intermediate_Waypoint_Velocities":
            return np.abs(output - self.lower_bound)
        if self.constraint_class == "Zero_Velocity_End_Waypoint_Location" or \
                                    "Zero_Velocity_Start_Waypoint_Location":
            return np.abs(output - self.lower_bound)

