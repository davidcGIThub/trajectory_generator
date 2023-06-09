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
import time

#note incline constraints work much better when have a start and an end direction.

dimension = 2
order = 3
# traj_objective_type = "minimal_acceleration_path"
traj_objective_type = "minimal_velocity_path"
# traj_objective_type = "minimal_distance_path"
sfc_data = None
obstacles = None

max_curvature = 0.5773502691896257
max_turning_bound = max_curvature
turning_bound = TurningBound(max_turning_bound,"curvature")
turning_bound = None

max_velocity = 3
# max_velocity = None
max_acceleration = None
derivative_bounds = DerivativeBounds(max_velocity, max_acceleration)
# derivative_bounds = None

### 1st path
waypoint_1 = Waypoint(location=np.array([[-6],[0]]),velocity=np.array([[0],[2.5]]))
waypoint_2 = Waypoint(location=np.array([[6],[0]]),velocity=np.array([[0],[2.5]]))
# waypoint_1 = Waypoint(location=np.array([[3],[4]]),direction=np.array([[-max_velocity],[0]]))
# waypoint_2 = Waypoint(location=np.array([[2],[10]]),direction=np.array([[0],[-max_velocity]]))

waypoint_data = WaypointData((waypoint_1, waypoint_2))
traj_gen = TrajectoryGenerator(dimension)
start_time = time.time()
control_points, scale_factor = traj_gen.generate_trajectory(waypoint_data, derivative_bounds, 
    turning_bound, sfc_data, obstacles, traj_objective_type)
end_time = time.time()
print("control_points: ", control_points)
print("scale_factor: " , scale_factor)

spline_start_time_1 = 0
bspline = BsplineEvaluation(control_points, order, spline_start_time_1, scale_factor, False)
end_time_spline = bspline.get_end_time()

## spline 1 data
number_data_points = 10000
spline_data, time_data = bspline.get_spline_data(number_data_points)
curvature_data, time_data = bspline.get_spline_curvature_data(number_data_points)
velocity_data, time_data = bspline.get_derivative_magnitude_data(number_data_points,1)
path_length = bspline.get_arc_length(number_data_points)
print("path_length: " , path_length)
print("computation time: " , end_time - start_time)

plt.figure()
ax = plt.axes()
# ax.scatter(control_points[0,:], control_points[1,:], color="tab:orange")
ax.plot(spline_data[0,:], spline_data[1,:], color = "tab:blue")
plot2D_waypoints(waypoint_data, ax)
set_axes_equal(ax,dimension)
plt.title("Optimized Path")
plt.show()

if max_velocity is not None:
    plt.figure()
    plt.plot(time_data, velocity_data,color = "b")
    plt.plot(time_data, max_velocity + velocity_data*0)
    plt.title("velocity")
    plt.show()

plt.figure()
plt.plot(time_data, curvature_data,color = "b")
# plt.plot(time_data, acceleration_data,color = "g")
plt.plot(time_data, max_curvature + curvature_data*0)
plt.title('curvature')
plt.show()
