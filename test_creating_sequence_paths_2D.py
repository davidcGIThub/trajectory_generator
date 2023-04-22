import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from bsplinegenerator.bsplines import BsplineEvaluation
from path_generation.path_generator import PathGenerator
from path_generation.safe_flight_corridor import SFC_Data, get3DRotationAndTranslationFromPoints
from path_generation.path_plotter import set_axes_equal
from path_generation.waypoint_data import Waypoint, WaypointData
import time

#note incline constraints work much better when have a start and an end direction

dimension = 2
max_curvature = 1
order = 3
scale_factor = 1
# path_objective_type = "minimal_acceleration_path"
path_objective_type = "minimal_velocity_path"

### 1st path
waypoint_1 = Waypoint(location=np.array([[3],[4]]))
waypoint_2 = Waypoint(location=np.array([[7],[10]]))
waypoint_1.velocity = np.array([[3],[0.5]])
waypoint_2.velocity = np.array([[1],[1]])
waypoint_data = WaypointData(start_waypoint=waypoint_1,end_waypoint=waypoint_2)
path_gen = PathGenerator(dimension)
start_time_1 = time.time()
control_points = path_gen.generate_path(waypoint_data=waypoint_data, max_curvature=max_curvature,
    max_incline=None, sfc_data=None, obstacles=None, objective_function_type=path_objective_type)
end_time_1 = time.time()
print("computation time path 1: " , end_time_1 - start_time_1)
spline_start_time_1 = 0
bspline = BsplineEvaluation(control_points, order, spline_start_time_1, scale_factor, False)
end_time_spline = bspline.get_end_time()


### second path

waypoint_1_two = Waypoint(location=np.array([[7],[10]]))
waypoint_2_two = Waypoint(location=np.array([[15],[2]]))
waypoint_1_two.velocity = bspline.get_derivative_at_time_t(end_time_spline,1)
waypoint_1_two.acceleration = bspline.get_derivative_at_time_t(end_time_spline,2)
waypoint_2_two.velocity = np.array([[1],[1]])
waypoint_data_2 = WaypointData(start_waypoint=waypoint_1_two,end_waypoint=waypoint_2_two)
spline_start_time_2 = end_time_spline
scale_factor_2 = 1
start_time_2 = time.time()
control_points_2 = path_gen.generate_path(waypoint_data=waypoint_data_2, max_curvature=max_curvature,
    max_incline=None, sfc_data=None, obstacles=None)
end_time_2 = time.time()

## spline 1 data
number_data_points = 10000
spline_data_1, time_data_1 = bspline.get_spline_data(number_data_points)
curvature_data_1, time_data_1 = bspline.get_spline_curvature_data(number_data_points)
velocity_data_1, time_data_1 = bspline.get_spline_derivative_data(number_data_points,1)
path_length_1 = bspline.get_arc_length(number_data_points)

## spline 2 data
bspline_2 = BsplineEvaluation(control_points_2, order, spline_start_time_2, scale_factor, False)
number_data_points = 10000
spline_data_2, time_data_2 = bspline_2.get_spline_data(number_data_points)
curvature_data_2, time_data_2 = bspline_2.get_spline_curvature_data(number_data_points)
velocity_data_2, time_data_2 = bspline_2.get_spline_derivative_data(number_data_points,1)
path_length_2 = bspline.get_arc_length(number_data_points)

print("computation time path 2: " , end_time_2 - start_time_2)

print("max curvature" , np.max((np.max(curvature_data_1), np.max(curvature_data_2)) ) )

plt.figure()
ax = plt.axes()
ax.plot(spline_data_1[0,:], spline_data_1[1,:], color = "b")
ax.scatter(control_points[0,:], control_points[1,:], color="b")
ax.plot(spline_data_2[0,:], spline_data_2[1,:], color = "r")
ax.scatter(control_points_2[0,:], control_points_2[1,:], color="r")
set_axes_equal(ax,dimension)
plt.title("Optimized Path")
plt.show()

plt.figure()
plt.plot(time_data_1, curvature_data_1,color = "b")
plt.plot(time_data_2, curvature_data_2, color = "r")
plt.title("curvature")
plt.show()
