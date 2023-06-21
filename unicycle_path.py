import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from bsplinegenerator.bsplines import BsplineEvaluation
from trajectory_generation.trajectory_generator import TrajectoryGenerator
from trajectory_generation.constraint_data_structures.safe_flight_corridor import SFC_Data, get3DRotationAndTranslationFromPoints
from trajectory_generation.path_plotter import set_axes_equal
from trajectory_generation.constraint_data_structures.waypoint_data import Waypoint, WaypointData, plot2D_waypoints
from trajectory_generation.constraint_data_structures.dynamic_bounds import DerivativeBounds, TurningBound
from trajectory_generation.constraint_data_structures.obstacle import Obstacle, plot_2D_obstacles
from trajectory_generation.constraint_data_structures.constraints_container import ConstraintsContainer
from trajectory_generation.spline_data_concatenater import SplineDataConcatenater

import time

#### Bicycle Properties ####
L = 1
l_r = 0.5
R = 0.2 # animation property
max_velocity = 10 #m/s
max_acceleration = 2.5 #m/s^2
max_longitudinal_acceleration = max_acceleration
max_angular_rate = 2
max_curvature = max_angular_rate/max_velocity
# max_centripetal_acceleration = max_curvature*ave_velocity**2

#### Path properties ####
dimension = 2
order = 3
start_time = 0

#### Path Objective ####
traj_objective_type = "minimal_acceleration_path" 
# traj_objective_type =  "minimal_velocity_path"

#### Trajectory Generator Object ####
traj_gen = TrajectoryGenerator(dimension)

#### Path Constraints ####
turning_bound = None
# turning_bound = TurningBound(max_centripetal_acceleration,"centripetal_acceleration")
turning_bound = TurningBound(max_angular_rate,"angular_rate")
# turning_bound = TurningBound(max_curvature,"curvature")

derivative_bounds = DerivativeBounds(max_velocity, max_acceleration)

# Path 1 generation
start_point_1 = Waypoint(location=np.array([[-7],[-7]]),velocity=np.array([[0],[0.01]]))
end_point_1 = Waypoint(location=np.array([[-7],[0]]),velocity=np.array([[0],[3]]))
waypoint_data_1 = WaypointData((start_point_1, end_point_1))
constraints_container_1 = ConstraintsContainer(waypoint_data_1, derivative_bounds,turning_bound)
gen_start_time = time.time()
control_points_1, scale_factor_1 = traj_gen.generate_trajectory(constraints_container_1, traj_objective_type)
gen_end_time = time.time()
print("Trajectory 1 generation time: " , gen_end_time - gen_start_time)
print("control_points_1: " , control_points_1)
print("scale_factor_1: " , scale_factor_1)