import numpy as np
from bsplinegenerator.bsplines import BsplineEvaluation
import matplotlib.pyplot as plt
from trajectory_generation.spline_data_concatenater import SplineDataConcatenater


dt = 1.2
start_time = 2.3

control_points_1 = np.array([[4,3,7,4,8,9,5 ,2 ,7],
                           [1,2,4,6,8,9,10,13,15]])
scale_factor_1 = 2
order_1 = 3
spline_1 = BsplineEvaluation(control_points_1, order_1, start_time, scale_factor_1)
spline_1_data, time_1_data = spline_1.get_spline_data(100)
spline_1_derivative_data, time_1_data = spline_1.get_spline_derivative_data(100,1)
spline_1_end_time = spline_1.get_end_time()

control_points_2 = np.array([[5 ,2 ,7 , 4, 8,    5],
                           [10,13,15, 15, 11, 10]])
scale_factor_2 = 1
order_2 = 3
spline_2 = BsplineEvaluation(control_points_2, order_2, spline_1_end_time, scale_factor_2)
spline_2_data, time_2_data = spline_2.get_spline_data(100)
spline_2_derivative_data, time_2_data = spline_2.get_spline_derivative_data(100,1)
spline_2_end_time = spline_2.get_end_time()

control_points_3 = np.array([[4,   8,  5, 9, 2, 8, 13],
                           [15, 11, 10, 8, 4, 2, 1 ]])
scale_factor_3 = 3
order_3 = 3
spline_3 = BsplineEvaluation(control_points_3, order_3, spline_2_end_time, scale_factor_3)
spline_3_data, time_3_data = spline_3.get_spline_data(100)
spline_3_derivative_data, time_3_data = spline_3.get_spline_derivative_data(100,1)

control_point_array_list = [control_points_1, control_points_2, control_points_3]
scale_factor_list = [scale_factor_1, scale_factor_2, scale_factor_3]
order_list = [order_1, order_2, order_3]

spline_conc = SplineDataConcatenater(2)
location_data, time_data = spline_conc.concatenate_spline_data(dt, start_time, order_list, control_point_array_list, scale_factor_list)
velocity_data, time_data_d = spline_conc.concatenate_spline_data(dt, start_time, order_list, control_point_array_list, scale_factor_list, derivative_order=1)

plt.figure()
plt.plot(spline_1_data[0,:], spline_1_data[1,:])
plt.plot(spline_2_data[0,:], spline_2_data[1,:])
plt.plot(spline_3_data[0,:], spline_3_data[1,:])
plt.scatter(location_data[0,:], location_data[1,:])
plt.show()

plt.figure()
plt.plot(time_1_data, spline_1_data[0,:])
plt.plot(time_2_data, spline_2_data[0,:])
plt.plot(time_3_data, spline_3_data[0,:])
plt.scatter(time_data, location_data[0,:])
plt.show()

plt.figure()
plt.plot(time_1_data, spline_1_derivative_data[0,:])
plt.plot(time_2_data, spline_2_derivative_data[0,:])
plt.plot(time_3_data, spline_3_derivative_data[0,:])
plt.scatter(time_data_d, velocity_data[0,:])
plt.show()

indices = np.linspace(1,len(time_data_d),len(time_data_d))

plt.figure()
plt.scatter(indices, time_data_d)
plt.show()