import numpy as np
import matplotlib.pyplot as plt
from pose_to_numpy import get_start_and_end_condition_data
import numpy as np
from math import factorial


### Collect Flight Info
bag_name = '../bag_parser/rosbag_defend_point.bag'
isTarget = False
# bag_name = '../bag_parser/rosbag_defend_point_2.bag'
# isTarget = False
# bag_name = '../bag_parser/rosbag_attack_target.bag' 
# isTarget = True
# bag_name = '../bag_parser/rosbag_attack_target_2.bag'
# isTarget = True
bag_name = '../bag_parser/rosbag_to_waypoint.bag' 
isTarget = False
# bag_name = '../bag_parser/rosbag_pursue_target.bag'
# isTarget = True
# bag_name = '../bag_parser/rosbag_pursue_target_2.bag'
# isTarget = True

num_samples = 1000
start_pos_data, start_vel_data, end_pos_data, end_vel_data = get_start_and_end_condition_data(bag_name, isTarget, num_samples)
time_data = start_vel_data[3,:]
vel_dt = (time_data[1:]-time_data[0:-1])
acceleration_data = (start_vel_data[0:3,1:] - start_vel_data[0:3,0:-1]) / vel_dt
# print(np.shape(acceleration_data))
# accel_dt = 
# jerk_data = (acceleration_data[:,1:] - acceleration_data[:,0:-1]) / time_data[2:]
velocity_mag_data = np.linalg.norm(start_vel_data[0:3,:],2,0)
acceleration_mag_data = np.linalg.norm(acceleration_data,2,0)
# print(np.shape(acceleration_mag_data))
# jerk_mag_data = np.linalg.norm(jerk_data,2,0)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

smoothed_acceleration_mag = savitzky_golay(acceleration_mag_data, 51, 3)
smoothed_acceleration_mag = smoothed_acceleration_mag[~np.isnan(smoothed_acceleration_mag)]
accel_time = time_data[0:len(smoothed_acceleration_mag)]

print("smoothed: " , np.shape(smoothed_acceleration_mag))
print("not smoothed: " , np.shape(acceleration_mag_data))

jerk_mag_data = np.abs((smoothed_acceleration_mag[1:] - smoothed_acceleration_mag[0:-1]))/ (accel_time[1:] - accel_time[0:-1])
smoothed_jerk_mag = savitzky_golay(jerk_mag_data, 51, 3)
print("max jerk: ", np.max(smoothed_jerk_mag))
plt.figure()
plt.plot(time_data, velocity_mag_data, label = "velocity")
plt.plot(accel_time, smoothed_acceleration_mag, label = "acceleration filtered")


plt.plot(accel_time[1:], smoothed_jerk_mag, label = "jerk")
plt.legend()
plt.show()