from trajectory_generation.constraint_data_structures.dynamic_bounds import DerivativeBounds, TurningBound
from trajectory_generation.constraint_data_structures.obstacle import ObstacleList
from trajectory_generation.constraint_data_structures.obstacle import Obstacle
from trajectory_generation.constraint_data_structures.safe_flight_corridor import SFC_Data
from trajectory_generation.constraint_data_structures.waypoint_data import WaypointData
from dataclasses import dataclass


@dataclass
class ConstraintsContainer:
    # @typechecked
    # def __init__(self, waypoint_constraints: WaypointData, 
    #              derivative_constraints: DerivativeBounds = None, 
    #              turning_constraint: TurningBound = None,
    #              sfc_constraints: SFC_Data = None,
    #              obstacle_constraints: 'list[Obstacle]' = None):
    #     self.waypoint_constraints = waypoint_constraints
    #     self.derivative_constraints: derivative_constraints
    #     self.turning_constraint: turning_constraint
    #     self.sfc_constraints: sfc_constraints
    #     self.obstacle_constraints = obstacle_constraints
    waypoint_constraints: WaypointData
    derivative_constraints: DerivativeBounds = None
    turning_constraint: TurningBound = None
    sfc_constraints: SFC_Data = None
    obstacle_constraints: 'list[Obstacle]' = None

