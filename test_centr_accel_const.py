

import numpy as np
import matplotlib.pyplot as plt
from trajectory_generation.constraint_functions.turning_constraints import TurningConstraints
from bsplinegenerator.bsplines import BsplineEvaluation

# control_points = np.array([[-4.83421641e+00, -5.08289179e+00, -4.83421641e+00, -4.54704421e+00,
#   -2.68488265e+00, -3.06836818e-03,  2.69674342e+00,  4.52514537e+00,
#    4.81542044e+00,  5.09228978e+00,  4.81542044e+00],
#  [-5.97971340e+00,  5.14374272e-01,  3.92221631e+00,  4.20371352e+00,
#    2.86687036e+00, -6.21545367e-03, -2.89622805e+00, -4.21499103e+00,
#   -3.92631472e+00, -5.12325069e-01,  5.97561499e+00]])


control_points = np.array([[-5.08289179, -4.83421641, -4.54704421, -2.68488265],
 [ 0.51437427,  3.92221631,  4.20371352,  2.86687036]])
scale_factor =  0.17682017335415903
dimension = 2
order = 3


turn_const = TurningConstraints(dimension)
# max_curvature = 3.1
# max_angular_rate = 0
max_centripetal_acceleration = 47.12388980384689
# curvature_constraint = turn_const.get_spline_curvature_constraint(control_points, max_curvature)
# angular_rate_constraint = turn_const.get_spline_angular_rate_constraint(control_points, max_angular_rate,  scale_factor)
num_intervals = 1
for i in range(num_intervals):
    ctrl_pts = control_points[:,i:i+order+1]
    print("ctrl_pts: " , ctrl_pts)
    constraint = turn_const.get_spline_centripetal_acceleration_constraint(ctrl_pts,max_centripetal_acceleration, scale_factor)
    centr_bound = constraint + max_centripetal_acceleration
    print("centr_bound: ", centr_bound)
    bspline = BsplineEvaluation(ctrl_pts, order, 0, scale_factor)
    centr_data, time_data = bspline.get_centripetal_acceleration_data(10000)
    accel_data, time_data = bspline.get_derivative_magnitude_data(10000,2)
    vel_data, time_data = bspline.get_derivative_magnitude_data(10000,1)
    cross_data, time_data = bspline.get_cross_term_data(10000)

    print("true bound: " , np.max(centr_data))
    print("min vel: " , np.min(vel_data))
    print("max accel: " , np.max(accel_data))
    print("max cross: " , np.max(cross_data))
    # plt.figure()
    # plt.plot(time_data, centr_data)
    # plt.show()

    plt.figure()
    plt.title("cross term")
    plt.plot(time_data, cross_data)
    plt.show()


    plt.figure()
    plt.title("velocity")
    plt.plot(time_data, vel_data)
    plt.show()

# print("curvature_constraint: " , curvature_constraint)
# print("angular_rate_constraint: " , angular_rate_constraint)
# print("centripetal_acceleration_constraint: " , centripetal_acceleration_constraint)