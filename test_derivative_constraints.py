from trajectory_generation.constraint_functions.derivative_constraints import DerivativeConstraints
import numpy as np
from bsplinegenerator.bsplines import BsplineEvaluation
from trajectory_generation.constraint_data_structures.dynamic_bounds import DerivativeBounds
import matplotlib.pyplot as plt

dimension = 3
order = 3
num_ctrl_pts = order + 1

derivative_const = DerivativeConstraints(dimension)


a = 6
b = -8
c = 1
roots = derivative_const.solve_quadratic(a,b,c)
print("roots: " , roots)

scale_factor = 2.3
t = np.random.random()*scale_factor
ctrl_pts = np.random.randint(10, size=(dimension,num_ctrl_pts))

bspline = BsplineEvaluation(ctrl_pts, order, 0, scale_factor)

tang_accel = derivative_const.calculate_dot_term_at_point_on_interval(ctrl_pts, scale_factor, t)

acceleration             = bspline.get_derivative_magnitude_at_time_t(t,2)
centripetal_acceleration = bspline.get_centripetal_acceleration_at_time_t(t)
tangential_acceleration  = np.sqrt(acceleration**2 - centripetal_acceleration**2)

print("tangential_acceleration: " , tangential_acceleration)
print("             tang accel: " , tang_accel)


bounds = derivative_const.calculate_tangential_acceleration_bounds_for_interval(ctrl_pts, scale_factor)
print("bounds: " , bounds)

velocity_magnitude, time_data = bspline.get_derivative_magnitude_data(100000,1)
minimum_velocity = np.min(velocity_magnitude)
min_vel_bound = 5
derivative_bounds = DerivativeBounds(min_velocity=min_vel_bound)
const = derivative_const.calculate_min_velocity_constraint(ctrl_pts, scale_factor, derivative_bounds)

print("         min vel: " , min_vel_bound - const)
print("minimum velocity: " , minimum_velocity)
    # def create_tangential_acceleration_constraint(self, derivative_bounds: DerivativeBounds, num_cont_pts, dimension, order):
    #     return tang_accel_constraint, constraint_function_data
        
    # def calculate_tangential_acceleration_bounds_for_interval(self, ctrl_pts, scale_factor):
    #     return max_tang_accel, min_tang_accel

control_points = np.array([[ 599.35973791,  647.09965675 , 680.50572195,  660.17545495 ],
 [ 152.32620391 , 237.35122757 , 317.46805259 , 384.12641315],
 [ 299.93785668 , 299.99056375 , 300.04673791,  299.99649911 ]])
scale_factor = 3.4807198963018147

for i in range(1):
    ctrl_pts = control_points[:,i:i+order+1]
    bounds = derivative_const.calculate_tangential_acceleration_bounds_for_interval(ctrl_pts, scale_factor)
    print("     bounds: " , bounds)
    temp_bspline = BsplineEvaluation(ctrl_pts, order,0,scale_factor)
    velocity,time_dat = temp_bspline.get_spline_derivative_data(10000,1)
    acceleration,time_dat = temp_bspline.get_spline_derivative_data(10000,2)
    velocity_unit = velocity / np.linalg.norm(velocity,2,0)
    acceleration_mag = np.linalg.norm(acceleration,2,0)
    # print("np.dot(acceleration.T, velocity_unit):" , np.dot(acceleration.T, velocity_unit))
    tang_accel = np.diag(np.dot(acceleration.T, velocity_unit))
    print("bounds true: " , np.max(tang_accel) , " " , np.min(tang_accel))



    plt.figure()
    plt.plot(time_dat, acceleration_mag, color="b")
    plt.plot(time_dat, tang_accel, color="r")
    plt.show()
