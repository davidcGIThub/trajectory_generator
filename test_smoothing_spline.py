import numpy as np
import matplotlib.pyplot as plt
from trajectory_generation.matrix_evaluation import matrix_bspline_evaluation_for_dataset, matrix_bspline_derivative_evaluation_for_dataset
from bsplinegenerator.bsplines import BsplineEvaluation
from trajectory_generation.smoothing_spline import SmoothingSpline
import time

num_data_points_per_interval = 1000
order = 3
control_points = np.array([[-3,  -4, -2, -.5, 1  ,   0,  2, 3.5, 3],
                            [.5, 3.5,  6, 5.5, 3.7,   2, -1,   2, 5]])
dimension = 2
num_control_points = order + 4
# control_points = control_points = np.random.randint(10, size=(dimension,num_control_points)) # random
num_points = 1000
scale_factor = 1
bspline = BsplineEvaluation(control_points, order, 0, scale_factor)
spline_data, time_data = bspline.get_spline_data(num_data_points_per_interval)
spline_velocity_data, time_data = bspline.get_derivative_magnitude_data(num_data_points_per_interval,1)
spline_acceleration_data, time_data = bspline.get_derivative_magnitude_data(num_data_points_per_interval,2)
curvature_data, time_data = bspline.get_spline_curvature_data(num_data_points_per_interval)
ang_rate_data, time_data = bspline.get_angular_rate_data(num_data_points_per_interval)
centripetal_acceleration_data, time_data = bspline.get_centripetal_acceleration_data(num_data_points_per_interval)
max_velocity = np.max(spline_velocity_data)
max_acceleration = np.max(spline_acceleration_data)

new_order = 4
dimension = 2
start_time = time.time()
smoother = SmoothingSpline(new_order, dimension, 100)
end_time = time.time()
print("elapsed time: " , end_time - start_time)
new_control_points, new_scale_factor = smoother.generate_new_control_points(control_points,scale_factor,order, max_velocity = max_velocity)

new_bspline = BsplineEvaluation(new_control_points, new_order, 0, new_scale_factor)
new_spline_data, new_time_data = new_bspline.get_spline_data(num_data_points_per_interval)
new_spline_velocity_data, new_time_data = new_bspline.get_derivative_magnitude_data(num_data_points_per_interval,1)
new_spline_acceleration_data, new_time_data = new_bspline.get_derivative_magnitude_data(num_data_points_per_interval,2)
new_curvature_data, new_time_data = new_bspline.get_spline_curvature_data(num_data_points_per_interval)
new_ang_rate_data, new_time_data = new_bspline.get_angular_rate_data(num_data_points_per_interval)
new_centripetal_acceleration_data, new_time_data = new_bspline.get_centripetal_acceleration_data(num_data_points_per_interval)
new_initial_control_points = smoother.create_initial_control_points(control_points, order, np.shape(new_control_points)[1])

plt.figure()
plt.plot(spline_data[0,:],spline_data[1,:],label="3rd order spline")
plt.scatter(control_points[0,:],control_points[1,:])
# plt.plot(control_points[0,:],control_points[1,:],color="tab:blue")
plt.plot(new_spline_data[0,:],new_spline_data[1,:],'--',label="4th order spline")
plt.scatter(new_control_points[0,:],new_control_points[1,:])
# plt.scatter(new_initial_control_points[0,:],new_initial_control_points[1,:],facecolors='none', edgecolors='tab:orange')
# plt.plot(new_initial_control_points[0,:],new_initial_control_points[1,:],color='tab:orange')
plt.title("spline")
plt.legend()
plt.show()


plt.figure()
plt.plot(time_data, spline_velocity_data)
plt.plot(new_time_data, new_spline_velocity_data,'--')
plt.title("velocity")
plt.show()

plt.figure()
plt.plot(time_data, spline_acceleration_data)
plt.plot(new_time_data, new_spline_acceleration_data,'--')
plt.title("acceleration")
plt.show()

plt.figure()
plt.plot(time_data, curvature_data)
plt.plot(new_time_data, new_curvature_data,'--')
plt.title("curvature")
plt.show()

plt.figure()
plt.plot(time_data, ang_rate_data)
plt.plot(new_time_data, new_ang_rate_data,'--')
plt.title("angular_rate")
plt.show()

plt.figure()
plt.plot(time_data, centripetal_acceleration_data)
plt.plot(new_time_data, new_centripetal_acceleration_data,'--')
plt.title("cent_accel")
plt.show()