import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from bsplinegenerator.bsplines import BsplineEvaluation
# from path_generation.path_generator import PathGenerator
from path_generation.path_generator import PathGenerator
from path_generation.safe_flight_corridor import SFC_Data, get3DRotationAndTranslationFromPoints
from path_generation.path_plotter import set_axes_equal
from path_generation.waypoint_data import Waypoint, WaypointData, plot3D_waypoints
from path_generation.obstacle import Obstacle, plot_3D_obstacles
import time

#note incline constraints work much better when have a start and an end direction

dimension = 3
max_curvature = 1
max_incline = 2
# max_incline = None
order = 3
scale_factor = 1
path_objective_type = "minimal_acceleration_path"
# path_objective_type = "minimal_velocity_path"

##### Initial Settings
# path 1
waypoint_1 = Waypoint(location=np.array([[3],[4],[0]]))
waypoint_2 = Waypoint(location=np.array([[7],[10],[13]]))
waypoint_1.velocity = waypoint_2.location - waypoint_1.location
waypoint_2.velocity = np.array([[1],[-1],[0]])
waypoint_data_1 = WaypointData(start_waypoint=waypoint_1,end_waypoint=waypoint_2)
obstacles_1 = [Obstacle(np.array([[4.5],[5.5],[4]]), 2.0)]
# path 2
waypoint_1_two = Waypoint(location=waypoint_2.location)
waypoint_2_two = Waypoint(location=np.array([[8],[-10],[7]]))
waypoint_1_two.velocity = waypoint_2.velocity
waypoint_2_two.velocity = waypoint_2_two.velocity = np.array([[-1],[0],[0]])
waypoint_data_2 = WaypointData(start_waypoint=waypoint_1_two,end_waypoint=waypoint_2_two)
plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')
plot_3D_obstacles(obstacles_1, ax)
plot3D_waypoints(waypoint_data_1, ax)
plot3D_waypoints(waypoint_data_2, ax)
set_axes_equal(ax,dimension)
plt.show()
number_data_points = 10000

### 1st path
path_gen = PathGenerator(dimension)
start_time_1 = time.time()
control_points = path_gen.generate_path(waypoint_data=waypoint_data_1, max_curvature=max_curvature,
    max_incline=max_incline, sfc_data=None, obstacles=obstacles_1, objective_function_type=path_objective_type)
end_time_1 = time.time()
print(" ")
print("PATH 1 DATA:")
print("computation time: " , end_time_1 - start_time_1)
spline_start_time_1 = 0
bspline = BsplineEvaluation(control_points, order, spline_start_time_1, scale_factor, False)
end_time_spline_1 = bspline.get_end_time()
#### path 1 data
spline_data_1, time_data_1 = bspline.get_spline_data(number_data_points)
curvature_data_1, time_data_1 = bspline.get_spline_curvature_data(number_data_points)
velocity_data_1, time_data_1 = bspline.get_spline_derivative_data(number_data_points,1)
z_vel_mag_1 = velocity_data_1[2,:]
horiz_vel_mag_1 = np.linalg.norm(velocity_data_1[0:2,:],2,0)
incline_data_1 = z_vel_mag_1/horiz_vel_mag_1

print("max incline: " , np.max(abs(incline_data_1)))
print("max curvature" , np.max(curvature_data_1))
print(" ")
### plot 1st path
plt.figure()
ax = plt.axes(projection='3d')
ax.plot(spline_data_1[0,:], spline_data_1[1,:],spline_data_1[2,:], color = "b")
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')
plot_3D_obstacles(obstacles_1, ax)
plot3D_waypoints(waypoint_data_1, ax)
set_axes_equal(ax,dimension)
plt.show()

### second path
waypoint_1_two.velocity = bspline.get_derivative_at_time_t(end_time_spline_1,1)
waypoint_1_two.acceleration = bspline.get_derivative_at_time_t(end_time_spline_1,2)
waypoint_data_2 = WaypointData(start_waypoint=waypoint_1_two,end_waypoint=waypoint_2_two)
spline_start_time_2 = end_time_spline_1
start_time_2 = time.time()
control_points_2 = path_gen.generate_path(waypoint_data=waypoint_data_2, max_curvature=max_curvature,
    max_incline=max_incline, sfc_data=None, obstacles=None)
end_time_2 = time.time()
## spline 2 data
bspline_2 = BsplineEvaluation(control_points_2, order, spline_start_time_2, scale_factor, False)
spline_data_2, time_data_2 = bspline_2.get_spline_data(number_data_points)
curvature_data_2, time_data_2 = bspline_2.get_spline_curvature_data(number_data_points)
velocity_data_2, time_data_2 = bspline_2.get_spline_derivative_data(number_data_points,1)
z_vel_mag_2 = velocity_data_2[2,:]
horiz_vel_mag_2 = np.linalg.norm(velocity_data_2[0:2,:],2,0)
# print(" z_vel_mag_2 ", z_vel_mag_2)
# print(" horiz_vel_mag_2 ", np.max(horiz_vel_mag_2))
incline_data_2 = z_vel_mag_2/horiz_vel_mag_2

print(" ")
print("PATH 2 DATA:")
print("computation time path 2: " , end_time_2 - start_time_2)
print("max incline: " , np.max(abs(incline_data_2)))
print("max curvature" , np.max(curvature_data_2))
print(" ")
plt.figure()
ax = plt.axes(projection='3d')
ax.plot(spline_data_1[0,:], spline_data_1[1,:],spline_data_1[2,:], color = "b")
# ax.scatter(control_points[0,:], control_points[1,:],control_points[2,:])
ax.plot(spline_data_2[0,:], spline_data_2[1,:],spline_data_2[2,:], color = "r")
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')
plot_3D_obstacles(obstacles_1, ax)
plot3D_waypoints(waypoint_data_1, ax)
plot3D_waypoints(waypoint_data_2, ax)
set_axes_equal(ax,dimension)
plt.show()

###Unforseen obstacle
obstacles_3 = [Obstacle(np.array([[10],[-5],[8]]), 1.5)]
plt.figure()
ax = plt.axes(projection='3d')
ax.plot(spline_data_1[0,:], spline_data_1[1,:],spline_data_1[2,:], color = "b")
# ax.scatter(control_points[0,:], control_points[1,:],control_points[2,:])
ax.plot(spline_data_2[0,:], spline_data_2[1,:],spline_data_2[2,:], color = "r")
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')
plot_3D_obstacles(obstacles_1, ax)
plot_3D_obstacles(obstacles_3, ax)
plot3D_waypoints(waypoint_data_1, ax)
plot3D_waypoints(waypoint_data_2, ax)
set_axes_equal(ax,dimension)
plt.show()
spline_2_end_time = bspline_2.get_end_time()
spline_2_mid_time = (spline_2_end_time - end_time_spline_1)/2.5 + end_time_spline_1
mid_point_path_2 = bspline_2.get_spline_at_time_t(spline_2_mid_time)

#### third path
# print("mid_point_path_2: " , mid_point_path_2)
waypoint_1_three = Waypoint(location=mid_point_path_2)
waypoint_2_three = Waypoint(location=waypoint_2_two.location)
waypoint_1_three.velocity = bspline_2.get_derivative_at_time_t(spline_2_mid_time,1)
waypoint_1_three.acceleration = bspline_2.get_derivative_at_time_t(spline_2_mid_time,2)
waypoint_2_three.velocity = waypoint_2_two.velocity
waypoint_data_3 = WaypointData(start_waypoint=waypoint_1_three,end_waypoint=waypoint_2_three)
spline_start_time_3 = spline_2_mid_time
start_time_3 = time.time()
control_points_3 = path_gen.generate_path(waypoint_data=waypoint_data_3, max_curvature=max_curvature,
    max_incline=max_incline, sfc_data=None, obstacles=obstacles_3)
end_time_3 = time.time()
## spline 2 data
bspline_3 = BsplineEvaluation(control_points_3, order, spline_start_time_3, scale_factor, False)
spline_data_3, time_data_3 = bspline_3.get_spline_data(number_data_points)
curvature_data_3, time_data_3 = bspline_3.get_spline_curvature_data(number_data_points)
velocity_data_3, time_data_3 = bspline_3.get_spline_derivative_data(number_data_points,1)
z_vel_mag_3 = velocity_data_3[2,:]
horiz_vel_mag_3 = np.linalg.norm(velocity_data_3[0:2,:],2,0)
incline_data_3 = z_vel_mag_3/horiz_vel_mag_3

print(" ")
print("PATH 3 DATA:")
print("computation time path 2: " , end_time_3 - start_time_3)
print("max incline: " , np.max(abs(incline_data_3)))
print("max curvature" , np.max(curvature_data_3))
print(" ")
plt.figure()
ax = plt.axes(projection='3d')
ax.plot(spline_data_1[0,:], spline_data_1[1,:],spline_data_1[2,:], color = "b")
ax.plot(spline_data_2[0,:], spline_data_2[1,:],spline_data_2[2,:], color = "r")
ax.plot(spline_data_3[0,:], spline_data_3[1,:],spline_data_3[2,:], color = "g")
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')
plot_3D_obstacles(obstacles_1, ax)
plot_3D_obstacles(obstacles_3, ax)
plot3D_waypoints(waypoint_data_1, ax)
# plot3D_waypoints(waypoint_data_2, ax)
plot3D_waypoints(waypoint_data_3, ax)
set_axes_equal(ax,dimension)
plt.show()

plt.figure()
plt.title("Curvature")
plt.plot(time_data_1, curvature_data_1,color = "b")
plt.plot(time_data_2, curvature_data_2, color = "r")
plt.plot(time_data_3, curvature_data_3, color = "g")
plt.show()
