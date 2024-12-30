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
import time

dimension = 2
# max_curvature = 1
order = 3
traj_objective_type = "minimal_acceleration_path"
# traj_objective_type = "minimal_velocity_path"
# traj_objective_type = "minimal_distance_path"
sfc_data = None
# obstacles = [Obstacle(center=np.array([[5.5],[7]]), radius=1)]
obstacles = None

max_turning_bound = 2 #angular rate
turning_bound = TurningBound(max_turning_bound,"angular_rate")
# turning_bound = None
obstacle_list = None

max_velocity = 5
# max_velocity = None
# max_acceleration = 5
max_acceleration = None
derivative_bounds = DerivativeBounds(max_velocity, max_acceleration)
# derivative_bounds = None

### 1st path

waypoint_1 = Waypoint(location=np.array([[3  ],[4]]),velocity = np.array([[1],[0]]))
waypoint_2 = Waypoint(location=np.array([[1.5],[6]]),velocity = np.array([[0],[1]]))
waypoint_3 = Waypoint(location=np.array([[3.4],[8]]),velocity = np.array([[4],[0]]))
waypoint_4 = Waypoint(location=np.array([[2  ],[10]]),velocity = np.array([[0],[1]]))

waypoint_sequence = (waypoint_1, waypoint_2, waypoint_3, waypoint_4)
waypoint_data = WaypointData(waypoint_sequence)
traj_gen = TrajectoryGenerator(dimension)
start_time_1 = time.time()

initial_control_points = None
initial_scale_factor = None

constraints_container = ConstraintsContainer(waypoint_constraints = waypoint_data, 
    derivative_constraints=derivative_bounds, turning_constraint=turning_bound, 
    sfc_constraints=sfc_data, obstacle_constraints=obstacle_list)

control_points, scale_factor, is_violation = traj_gen.generate_trajectory(constraints_container, num_intervals_free_space=14)

print("control_points: " , control_points)
print("scale_factor: ", scale_factor)
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
angular_rate_data, time_data = bspline.get_angular_rate_data(number_data_points)
centripetal_acceleration_data, time_data = bspline.get_centripetal_acceleration_data(number_data_points)
path_length = bspline.get_arc_length(number_data_points)
start_velocity = bspline.get_derivative_at_time_t(0,1)
start_acceleration = bspline.get_derivative_at_time_t(0,2)

print("path_length: " , path_length)
print("computation time: " , end_time_1 - start_time_1)

velocity_matrix, time_data = bspline.get_spline_derivative_data(number_data_points,1)
acceleration_matrix, time_data = bspline.get_spline_derivative_data(number_data_points,2)

plt.figure()
ax = plt.axes()
# ax.scatter(control_points[0,:], control_points[1,:], color="tab:orange")
ax.plot(spline_data[0,:], spline_data[1,:], color = "tab:blue")
plot2D_waypoints(waypoint_data, ax)
set_axes_equal(ax,dimension)
plt.title("Optimized Path")
plt.show()


plt.figure(2)
plt.plot(time_data,velocity_matrix[0,:],label="x")
plt.plot(time_data,velocity_matrix[1,:],label="y")
plt.plot(time_data,np.linalg.norm(velocity_matrix,2,0),label="mag")
plt.legend()
plt.title("Velocity")
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



