import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from bsplinegenerator.bsplines import BsplineEvaluation
from trajectory_generation.path_generator import PathGenerator
from trajectory_generation.path_plotter import set_axes_equal
from trajectory_generation.constraint_data_structures.waypoint_data import Waypoint, WaypointData, plot2D_waypoints
from trajectory_generation.constraint_data_structures.dynamic_bounds import TurningBound
from trajectory_generation.constraint_data_structures.constraints_container import ConstraintsContainer
import time

#note incline constraints work much better when have a start and an end direction.

dimension = 2
# max_curvature = 1
order = 3
# traj_objective_type = "minimal_acceleration_path"
path_objective_type = "minimal_velocity_path"
# traj_objective_type = "minimal_distance_path"
sfc_data = None
obstacle_list = None

max_turning_bound = 1 #angular rate
turning_bound = TurningBound(max_turning_bound,"angular_rate")
turning_bound = None
# max_turning_bound = 0.5 #cent accel
# turning_bound = TurningBound(max_turning_bound,"centripetal_acceleration")

# max_turning_bound = 1 #curv
# turning_bound = TurningBound(max_turning_bound,"curvature")

# turning_bound = None

max_velocity = 1.5
# max_velocity = None
# max_acceleration = 0.1
max_acceleration = 3
derivative_bounds = DerivativeBounds(max_velocity, max_acceleration)
# derivative_bounds = None



### 1st path
waypoint_1 = Waypoint(location=np.array([[3],[4]]),velocity=np.array([[1],[0]]))
waypoint_2 = Waypoint(location=np.array([[2],[10]]),velocity=np.array([[0],[0]]))
# waypoint_1 = Waypoint(location=np.array([[3],[4]]),direction=np.array([[-max_velocity],[0]]))
# waypoint_2 = Waypoint(location=np.array([[2],[10]]),direction=np.array([[0],[-max_velocity]]))

waypoint_sequence = (waypoint_1, waypoint_2)
waypoint_data = WaypointData(waypoint_sequence)
path_gen = PathGenerator(dimension)
constraints_container = ConstraintsContainer(waypoint_constraints = waypoint_data, 
                                             turning_constraint=turning_bound, 
                                             sfc_constraints=sfc_data, 
                                             obstacle_constraints=obstacle_list)
start_time_1 = time.time()
control_points = path_gen.generate_path(constraints_container, 
                                        objective_function_type=path_objective_type,
                                        isIndirect = False)
end_time_1 = time.time()
print("shape control points: " , np.shape(control_points))
scale_factor = 1
spline_start_time_1 = 0
bspline = BsplineEvaluation(control_points, order, spline_start_time_1, scale_factor, False)
end_time_spline = bspline.get_end_time()
bspline.plot_spline(1000)

## spline 1 data
number_data_points = 10000
spline_data, time_data = bspline.get_spline_data(number_data_points)
print("spline data shape: " , np.shape(spline_data))
curvature_data, time_data = bspline.get_spline_curvature_data(number_data_points)
# velocity_data, time_data = bspline.get_derivative_magnitude_data(number_data_points,1)
# acceleration_data, time_data = bspline.get_derivative_magnitude_data(number_data_points,2)
path_length = bspline.get_arc_length(number_data_points)
print("path_length: " , path_length)
print("computation time: " , end_time_1 - start_time_1)


plt.figure()
ax = plt.axes()
# ax.scatter(control_points[0,:], control_points[1,:], color="tab:orange")
ax.plot(spline_data[0,:], spline_data[1,:], color = "tab:blue")
plot2D_waypoints(waypoint_data, ax)
set_axes_equal(ax,dimension)
plt.title("Optimized Path")
plt.show()


if turning_bound is not None:
    turn_title = turning_bound.bound_type
    plt.figure()
    plt.plot(time_data, curvature_data,color = "b")
    # plt.plot(time_data, acceleration_data,color = "g")
    plt.plot(time_data, max_curvature + curvature_data*0)
    plt.title(turn_title)
    plt.show()
