import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from bsplinegenerator.bsplines import BsplineEvaluation
# from trajectory_generation.trajectory_generator import TrajectoryGenerator
from trajectory_generation.trajectory_generator_2 import TrajectoryGenerator
from trajectory_generation.safe_flight_corridor import SFC_Data, get3DRotationAndTranslationFromPoints
from trajectory_generation.path_plotter import set_axes_equal
from trajectory_generation.waypoint_data import Waypoint, WaypointData, plot3D_waypoints
from trajectory_generation.dynamic_bounds import DerivativeBounds, TurningBound
from trajectory_generation.safe_flight_corridor import SFC, plot_sfcs
import time

#note incline constraints work much better when have a start and an end direction

dimension = 3
# max_curvature = 1
order = 3
# traj_objective_type = "minimal_acceleration_path"
# traj_objective_type = "minimal_velocity_path"
traj_objective_type = "minimal_distance_path"


point_1 = np.array([[3],[4],[0]])
point_2 = np.array([[7],[10],[3]])
point_3 = np.array([[14],[7],[7]])
point_4 = np.array([[20],[31],[20]])
point_sequence = np.concatenate((point_1,point_2,point_3,point_4),1)
dimension = np.shape(point_1)[0]
R1, T1, min_len_1 = get3DRotationAndTranslationFromPoints(point_1, point_2)
R2, T2, min_len_2 = get3DRotationAndTranslationFromPoints(point_2, point_3)
R3, T3, min_len_3 = get3DRotationAndTranslationFromPoints(point_3, point_4)
sfc_1 = SFC(np.array([[min_len_1+3],[2],[3]]), T1, R1)
sfc_2 = SFC(np.array([[min_len_2 + 2],[3],[4]]), T2, R2)
sfc_3 = SFC(np.array([[min_len_3+3],[2],[2]]), T3, R3)
sfcs = (sfc_1, sfc_2, sfc_3)
sfc_data = SFC_Data(sfcs, point_sequence)
# sfc_data = None
obstacles = None
max_turning_bound = .5
turning_bound = TurningBound(max_turning_bound,"angular_rate")
# turning_bound = TurningBound(max_turning_bound,"centripetal_acceleration")
# turning_bound = TurningBound(max_turning_bound,"curvature")
turning_bound = None

max_velocity = 2
# max_velocity = None
max_acceleration = .8
gravity = 0.1
# gravity = None
max_upward_velocity = 1
derivative_bounds = DerivativeBounds(max_velocity, max_acceleration, gravity, max_upward_velocity)
# derivative_bounds = None

### 1st path
waypoint_1_location = point_1
waypoint_4_location = point_4
waypoint_1 = Waypoint(location=waypoint_1_location)
waypoint_4 = Waypoint(location=waypoint_4_location)
waypoint_data = WaypointData(start_waypoint=waypoint_1,end_waypoint=waypoint_4)
traj_gen = TrajectoryGenerator(dimension)
start_time_1 = time.time()

control_points, scale_factor = traj_gen.generate_trajectory(waypoint_data, derivative_bounds, 
    turning_bound, sfc_data, obstacles, traj_objective_type)
end_time_1 = time.time()
spline_start_time_1 = 0
bspline = BsplineEvaluation(control_points, order, spline_start_time_1, scale_factor, False)
end_time_spline = bspline.get_end_time()

## spline 1 data
number_data_points = 10000
spline_data, time_data = bspline.get_spline_data(number_data_points)
curvature_data, time_data = bspline.get_spline_curvature_data(number_data_points)
velocity_data, time_data = bspline.get_derivative_magnitude_data(number_data_points,1)
acceleration_data, time_data = bspline.get_derivative_magnitude_data(number_data_points,2)
acceleration_spline_data, time_data = bspline.get_spline_derivative_data(number_data_points, 2)
angular_rate_data, time_data = bspline.get_angular_rate_data(number_data_points)
centripetal_acceleration_data, time_data = bspline.get_centripetal_acceleration_data(number_data_points)
path_length = bspline.get_arc_length(number_data_points)
start_velocity = bspline.get_derivative_at_time_t(0,1)
start_acceleration = bspline.get_derivative_at_time_t(0,2)
print("start_velocity: " , start_velocity)
print("start_acceleraiton: " , start_acceleration)
print("path_length: " , path_length)
print("computation time: " , end_time_1 - start_time_1)


velocity_matrix, time_data = bspline.get_spline_derivative_data(number_data_points,1)
acceleration_matrix, time_data = bspline.get_spline_derivative_data(number_data_points,2)

plt.figure()
ax = plt.axes(projection='3d')
# ax.scatter(control_points[0,:], control_points[1,:], color="tab:orange")
ax.plot(spline_data[0,:], spline_data[1,:], spline_data[2,:], color = "tab:blue")
plot3D_waypoints(waypoint_data, ax)
plot_sfcs(sfc_data._sfc_list, ax)
set_axes_equal(ax,dimension)
plt.title("Optimized Path")
plt.show()

if max_velocity is not None:
    plt.figure()
    plt.plot(time_data, velocity_data,color = "b")
    plt.plot(time_data, max_velocity + velocity_data*0)
    plt.title("velocity")
    plt.show()

if max_acceleration is not None:
    plt.figure()
    plt.plot(time_data, acceleration_data,color = "b")
    plt.plot(time_data, max_acceleration + acceleration_data*0)
    plt.title("acceleration")
    plt.show()

if gravity is not None:
    plt.figure()
    plt.plot(time_data, acceleration_spline_data[2,:],color = "b")
    plt.plot(time_data, max_acceleration + gravity + acceleration_data*0)
    plt.plot(time_data, -max_acceleration + gravity + acceleration_data*0)
    plt.title("acceleration z dir ")
    plt.show()



if turning_bound is not None:
    turn_data = []
    if turning_bound.bound_type == "angular_rate":
        turn_data = angular_rate_data
    elif turning_bound.bound_type == "curvature":
        turn_data = curvature_data
    elif turning_bound.bound_type == "centripetal_acceleration":
        turn_data = centripetal_acceleration_data
    turn_title = turning_bound.bound_type
    plt.figure()
    plt.plot(time_data, turn_data,color = "b")
    # plt.plot(time_data, acceleration_data,color = "g")
    plt.plot(time_data, max_turning_bound + turn_data*0)
    plt.title(turn_title)
    plt.show()