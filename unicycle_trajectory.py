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
from vehicle_simulator.vehicle_models.unicycle_model import UnicycleModel
from vehicle_simulator.vehicle_controllers.unicycle_trajectory_tracker import UnicycleTrajectoryTracker
from vehicle_simulator.vehicle_simulators.vehicle_trajectory_tracking_simulator import VehicleTrajectoryTrackingSimulator, TrajectoryData

import time

#### Bicycle Properties ####
L = 1
l_r = 0.5
R = 0.2 # animation property
max_velocity = 30 #m/s
max_acceleration = 100 #m/s^2
max_longitudinal_acceleration = max_acceleration
expected_min_vel = 20
max_angular_rate = 7
max_curvature = max_angular_rate / expected_min_vel
max_centripetal_acceleration = max_angular_rate * max_velocity

#### Path Properties ####
dimension = 2
order = 3
start_time = 0

#### Path Objective ####
# traj_objective_type = "minimal_acceleration_and_time_path" 
traj_objective_type =  "minimal_velocity_and_time_path"

#### Trajectory Generator Object ####
traj_gen = TrajectoryGenerator(dimension)

#### Path Constraints ####
turn_type = "curvature"
# turn_type = "angular_rate"
# turn_type = "centripetal_acceleration"
if turn_type == "curvature": max_turn_value = max_curvature
elif turn_type == "angular_rate": max_turn_value = max_angular_rate
elif turn_type == "centripetal_acceleration": max_turn_value = max_centripetal_acceleration
else: turn_type = None
print("max " , turn_type , ": ", max_turn_value)
turning_bound = TurningBound(max_turn_value, turn_type)
# turning_bound = None
derivative_bounds = DerivativeBounds(max_velocity, max_acceleration)


# Path generation
start_point = Waypoint(location=np.array([[-5],[0]]),velocity=np.array([[0],[28]]))
end_point = Waypoint(location=np.array([[5],[0]]),velocity=np.array([[0],[20]]))
waypoint_data = WaypointData((start_point, end_point))
constraints_container = ConstraintsContainer(waypoint_data, derivative_bounds, turning_bound)
num_intervals_free_space = 5
gen_start_time = time.time()
control_points, scale_factor = traj_gen.generate_trajectory(constraints_container, traj_objective_type, num_intervals_free_space)
print("control_points: " , control_points)
print("scale_factor: " , scale_factor)
gen_end_time = time.time()
print("Trajectory generation time: " , gen_end_time - gen_start_time)


bspline = BsplineEvaluation(control_points, order, 0, scale_factor)
num_points_per_interval = 500
location_data, time_data = bspline.get_spline_data(num_points_per_interval)
velocity_data, time_data = bspline.get_spline_derivative_data(num_points_per_interval, 1)
acceleration_data, time_data = bspline.get_spline_derivative_data(num_points_per_interval, 2)
jerk_data, time_data = bspline.get_spline_derivative_data(num_points_per_interval, 3)

start_vel = velocity_data[:,0]
start_direction = start_vel/np.linalg.norm(start_vel,2,0)
start_point = location_data[:,0]
start_heading = np.arctan2(start_direction[1], start_direction[0])


# Unicycle Model
unicycle = UnicycleModel(
                    x = start_point[0], 
                    y = start_point[1],
                    theta = start_heading,
                    x_dot = start_vel[0], 
                    y_dot = start_vel[1], 
                    theta_dot = 0, 
                    # alpha = np.array([0.1,0.01,0.1,0.01]),
                    alpha = np.array([0,0,0,0]),
                    max_theta_dot = max_angular_rate,
                    max_vel = max_velocity,
                    max_vel_dot = max_acceleration)

controller = UnicycleTrajectoryTracker(k_pos = 8, 
                                        k_vel = 5,
                                        k_theta = 7,
                                        max_vel_dot = max_acceleration,
                                        max_vel = max_velocity,
                                        max_theta_dot= max_angular_rate)


unicycle_traj_sim = VehicleTrajectoryTrackingSimulator(unicycle, controller)
des_traj_data = TrajectoryData(location_data, velocity_data, acceleration_data, 
                           jerk_data, time_data)
vehicle_traj_data, vehicle_motion_data = unicycle_traj_sim.run_simulation(des_traj_data)
unicycle_traj_sim.plot_simulation_dynamics(vehicle_motion_data, des_traj_data, vehicle_traj_data, max_velocity,
                                       max_acceleration, max_turn_value, turn_type, "unicyle")

