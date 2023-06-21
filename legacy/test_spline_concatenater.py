import numpy as np
from bsplinegenerator.bsplines import BsplineEvaluation
import matplotlib.pyplot as plt
from trajectory_generation.spline_concatenater import SplineConcatenater
import time

control_points_1 = np.array([[4,3,7,4,8,9,5 ,2 ,7],
                           [1,2,4,6,8,9,10,13,15]])
scale_factor_1 = 2
order_1 = 3
spline_1 = BsplineEvaluation(control_points_1, order_1, 0, scale_factor_1)
spline_1_data, time_1_data = spline_1.get_spline_data(100)
spline_1_derivative_data, time_1_data = spline_1.get_spline_derivative_data(100,1)
spline_1_end_time = spline_1.get_end_time()

control_points_2 = np.array([[5 ,2 ,7 , 4, 8,    5],
                           [10,13,15, 15, 11, 10]])
scale_factor_2 = 2
order_2 = 3
spline_2 = BsplineEvaluation(control_points_2, order_2, spline_1_end_time, scale_factor_2)
spline_2_data, time_2_data = spline_2.get_spline_data(100)
spline_2_derivative_data, time_2_data = spline_2.get_spline_derivative_data(100,1)
spline_2_end_time = spline_2.get_end_time()

control_points_3 = np.array([[4,   8,  5, 9, 2, 8, 13],
                           [15, 11, 10, 8, 4, 2, 1 ]])
scale_factor_3 = 2
order_3 = 3
spline_3 = BsplineEvaluation(control_points_3, order_3, spline_2_end_time, scale_factor_3)
spline_3_data, time_3_data = spline_3.get_spline_data(100)
spline_3_derivative_data, time_3_data = spline_3.get_spline_derivative_data(100,1)

control_point_array_list = [control_points_1, control_points_2, control_points_3]
scale_factor_list = [scale_factor_1, scale_factor_2, scale_factor_3]
order_list = [order_1, order_2, order_3]

# spline_conc = SplineDataConcatenater(100,2)
# location_data, time_data = spline_conc.concatenate_spline_data(order_list, control_point_array_list, scale_factor_list)

order = 3
dimension = 2
resolution = 100
print("optimizing")
spline_conc = SplineConcatenater(order, dimension, resolution)
start_time = time.time()
optimized_control_points, scale_factor = spline_conc.concatenate_splines(order_list, control_point_array_list, scale_factor_list)
end_time = time.time()
print("elapsed time: " , end_time - start_time)
optimized_spline = BsplineEvaluation(optimized_control_points, order, 0, scale_factor)
optimized_spline_data, optimized_time_data = optimized_spline.get_spline_data(100)
optimized_derivative_data, optimized_time_data = optimized_spline.get_spline_derivative_data(100,1)

plt.figure(1)
plt.plot(spline_1_data[0,:], spline_1_data[1,:])
plt.plot(spline_2_data[0,:], spline_2_data[1,:])
plt.plot(spline_3_data[0,:], spline_3_data[1,:])
plt.plot(optimized_spline_data[0,:], optimized_spline_data[1,:])
plt.show()

plt.figure(2)
plt.plot(spline_1_derivative_data[0,:], spline_1_derivative_data[1,:])
plt.plot(spline_2_derivative_data[0,:], spline_2_derivative_data[1,:])
plt.plot(spline_3_derivative_data[0,:], spline_3_derivative_data[1,:])
plt.plot(optimized_derivative_data[0,:], optimized_derivative_data[1,:])
plt.show()



