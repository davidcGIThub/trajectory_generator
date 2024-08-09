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
from vehicle_simulator.vehicle_models.bicycle_model import BicycleModel
from vehicle_simulator.vehicle_controllers.bicycle_trajectory_tracker import BicycleTrajectoryTracker
from vehicle_simulator.vehicle_simulators.vehicle_trajectory_tracking_simulator import VehicleTrajectoryTrackingSimulator, TrajectoryData

import time

#### Bicycle Properties ####

gravity = 9.8
max_velocity = 28 #m/s
min_velocity = 16 #m/s
max_roll = np.radians(25)
max_centripetal_acceleration = gravity*np.tan(max_roll)

max_curvature = max_centripetal_acceleration/max_velocity**2
max_angular_rate = max_centripetal_acceleration/max_velocity
max_tang_acceleration = 5
min_tang_acceleration = -1.5

# print("centr accel bound: " , max_centripetal_acceleration)
# print("curvature bound: " , max_curvature)

#### Path Properties ####
dimension = 2
order = 3
start_time = 0

#### Path Objective ####
traj_objective_type = "minimal_acceleration_and_time_path" 
# traj_objective_type =  "minimal_velocity_and_time_path"
# traj_objective_type = "minimal_distance_and_time_path"
# traj_objective_type = "minimal_time_path"
# traj_objective_type = "minimal_time_path_velocity_penalty"

#### Trajectory Generator Object ####
traj_gen = TrajectoryGenerator(dimension)

#### Path Constraints ####
# turn_type = "curvature"
# turn_type = "angular_rate"
turn_type = "centripetal_acceleration"
if turn_type == "curvature": max_turn_value = max_curvature
elif turn_type == "angular_rate": max_turn_value = max_angular_rate
elif turn_type == "centripetal_acceleration": max_turn_value = max_centripetal_acceleration
else: turn_type = None
print("max " , turn_type , ": ", max_turn_value)
turning_bound = TurningBound(max_turn_value, turn_type)
# turning_bound = None
derivative_bounds = DerivativeBounds(
                                    #  max_velocity=max_velocity, \
                                    #  max_tangential_acceleration=max_tang_acceleration, \
                                    #  min_tangential_acceleration=min_tang_acceleration, \
                                        )
# Path generation
# start_point = Waypoint(location = np.array([600,-100,300])[:,None],
#                          velocity = np.array([-(max_velocity-0.5),  0, 0])[:,None])
# end_point = Waypoint(location=np.array([600,  400,  300])[:,None],
#                       velocity=np.array([-(max_velocity-0.5), 0,  0.])[:,None])
start_point = Waypoint(location = np.array([600,-100])[:,None],
                         velocity = np.array([-300,  0])[:,None])
end_point = Waypoint(location=np.array([600,  400])[:,None],
                      velocity=np.array([-300, 0])[:,None])
waypoint_data = WaypointData((start_point, end_point))
constraints_container = ConstraintsContainer(waypoint_data, derivative_bounds, turning_bound)
num_intervals_free_space = 8
gen_start_time = time.time()
control_points, scale_factor, is_violation = traj_gen.generate_trajectory(constraints_container, traj_objective_type, num_intervals_free_space)
gen_end_time = time.time()
print("control_points: " , control_points)
print("scale_factor: " , scale_factor)
print("Trajectory generation time: " , gen_end_time - gen_start_time)

bspline = BsplineEvaluation(control_points, order, 0, scale_factor)
num_points_per_interval = 500
location_data, time_data = bspline.get_spline_data(num_points_per_interval)
velocity_data, time_data = bspline.get_spline_derivative_data(num_points_per_interval, 1)
velocity_magnitude_data = np.max(np.linalg.norm(velocity_data,2,0))
acceleration_data, time_data = bspline.get_spline_derivative_data(num_points_per_interval, 2)
jerk_data, time_data = bspline.get_spline_derivative_data(num_points_per_interval, 3)
centripetal_acceleration_data, time_data = bspline.get_centripetal_acceleration_data(num_points_per_interval)
angular_rate_data, time_data = bspline.get_angular_rate_data(num_points_per_interval)
curvature_data, time_data = bspline.get_spline_curvature_data(num_points_per_interval)
unit_velocity_data = velocity_data/velocity_magnitude_data
tangential_acceleration_magnitude_data = np.diag(np.dot(acceleration_data.T, unit_velocity_data))

print("max velocity: " , np.max(np.linalg.norm(velocity_data,2,0)))
print("min velocity: " , np.min(np.linalg.norm(velocity_data,2,0)))
# print("max_acceleration: " , np.max(np.linalg.norm(acceleration_data,2,0)))
print("max centr accel: " , np.max(centripetal_acceleration_data))
print("max ang rate: " , np.max(angular_rate_data))
print("max_curvature: " , np.max(curvature_data))
print("max tangential acceleration: " , np.max(tangential_acceleration_magnitude_data))
print("min tangential acceleration: " , np.min(tangential_acceleration_magnitude_data))

bspline.plot_spline(1000)

