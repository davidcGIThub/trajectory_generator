import bagpy
from bagpy import bagreader
import pandas as pd
# import seaborn as sea
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import copy

# b = bagreader('rosbag_defend_point.bag')

def trim_data_set(y_data_list, time_data, start_time, end_time, zero_time = True):
    start_index = np.argmin(np.abs(time_data - start_time))
    end_index = np.argmin(np.abs(time_data - end_time))
    new_time_data = time_data[start_index:end_index] 
    if zero_time:
        new_time_data = new_time_data - time_data[start_index]
    new_y_data_list = []
    for i in range(len(y_data_list)):
        new_y_data_list.append(y_data_list[i][start_index:end_index])
    return new_y_data_list, new_time_data

# def thin_data_set(data_list, num_samples):
#     ''' Samples the data set, assumes that each data set
#     in the list is the same length'''
#     new_data_list = []
#     sample_step = int(len(data_list[0])/num_samples)
#     for i in range(len(data_list)):
#         new_data = data_list[i][0::sample_step]
#         new_data_list.append(new_data)
#     return new_data_list

def thin_time_data(time_data, num_samples):
    new_time_data = np.linspace(time_data[0], time_data[-1], num_samples)
    return new_time_data

def get_data_at_times(data_array, times):
    time_data = data_array[3,:]
    num_times = len(times)
    new_data_array = np.zeros((4,num_times))
    for i in range(num_times):
        index = np.argmin(np.abs(time_data - times[i]))
        new_data_array[:,i] = data_array[:,index]
    return new_data_array

def zero_time_data(time_data):
    new_time_data = time_data - time_data[0]
    return new_time_data

def get_start_and_end_condition_data(bag_name: str, isTarget: bool, num_samples: int):
    b = bagreader(bag_name)
    desired_position_csv = b.message_by_topic('anansi/flight_commander/desired_position')
    position_csv = b.message_by_topic('anansi/vehicle_local_position')
    velocity_csv = b.message_by_topic('anansi/vehicle_velocity')
    df_desired_position = pd.read_csv(desired_position_csv)
    des_pos_time = df_desired_position['Time'].values
    des_x_pos = df_desired_position["pose.position.x"].values
    des_y_pos = df_desired_position["pose.position.y"].values
    des_z_pos = df_desired_position["pose.position.z"].values
    df_position = pd.read_csv(position_csv)
    pos_time = df_position["Time"].values
    x_pos = df_position["pose.position.x"].values
    y_pos = df_position["pose.position.y"].values
    z_pos = df_position["pose.position.z"].values
    df_velocity = pd.read_csv(velocity_csv)
    vel_time = df_velocity["Time"].values
    x_vel = df_velocity["twist.linear.x"].values
    y_vel = df_velocity["twist.linear.y"].values
    z_vel = df_velocity["twist.linear.z"].values
    if isTarget:
        target_track_csv = b.message_by_topic('anansi/target_track')
        df_target_track = pd.read_csv(target_track_csv)
        target_time = df_target_track['Time'].values
        # target_x_pos = df_target_track["x"].values
        # target_y_pos = df_target_track["y"].values
        # target_z_pos = df_target_track["z"].values
        des_x_vel = df_target_track["vx"].values
        des_y_vel = df_target_track["vy"].values
        des_z_vel = df_target_track["vz"].values
        start_time = target_time[0]
        end_time = target_time[-1]
        #trim the values
        [des_x_pos, des_y_pos, des_z_pos], des_pos_time = trim_data_set([des_x_pos, des_y_pos, des_z_pos], des_pos_time, start_time, end_time, zero_time = True)
        des_vel_time = zero_time_data(target_time)
        time_data = thin_time_data(des_vel_time, num_samples)
    else:
        des_x_vel = x_vel*0
        des_y_vel = y_vel*0
        des_z_vel = z_vel*0
        des_vel_time = copy.deepcopy(vel_time)
        start_time = des_pos_time[0]
        end_time = des_pos_time[-1]
        #trim the values
        [des_x_vel, des_y_vel, des_z_vel], des_vel_time = trim_data_set([des_x_vel, des_y_vel, des_z_vel], des_vel_time, start_time, end_time, zero_time = True)
        des_pos_time = zero_time_data(des_pos_time)
        time_data = thin_time_data(des_pos_time, num_samples)
    #trim the values
    [x_pos, y_pos, z_pos], pos_time = trim_data_set([x_pos, y_pos, z_pos], pos_time, start_time, end_time, zero_time = True)
    [x_vel, y_vel, z_vel], vel_time = trim_data_set([x_vel, y_vel, z_vel], vel_time, start_time, end_time, zero_time = True)
    # thin values
    start_pos_data = np.vstack((x_pos, y_pos, z_pos, pos_time))
    start_vel_data = np.vstack((x_vel, y_vel, z_vel, vel_time))
    end_pos_data = np.vstack((des_x_pos, des_y_pos, des_z_pos, des_pos_time))
    end_vel_data = np.vstack((des_x_vel, des_y_vel, des_z_vel, des_vel_time))
    # print("time_data: " , time_data)
    start_pos_data = get_data_at_times(start_pos_data, time_data)
    start_vel_data = get_data_at_times(start_vel_data, time_data)
    end_pos_data = get_data_at_times(end_pos_data, time_data)
    end_vel_data = get_data_at_times(end_vel_data, time_data)
    return start_pos_data, start_vel_data, end_pos_data, end_vel_data 

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
# bag_name = 'rosbag_defend_point.bag'
# isTarget = False
# num_samples = 11
# start_pos_data, start_vel_data, end_pos_data, end_vel_data = get_start_and_end_condition_data(bag_name, isTarget, num_samples)


# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(start_pos_data[0,:], start_pos_data[1,:], start_pos_data[2,:], color="tab:blue",label ="start")
# ax.plot(end_pos_data[0,:], end_pos_data[1,:], end_pos_data[2,:], color="tab:red",label ="end")
# ax.scatter(start_pos_data[0,:], start_pos_data[1,:], start_pos_data[2,:],facecolors="tab:blue", edgecolors="tab:blue")
# ax.scatter(end_pos_data[0,:], end_pos_data[1,:], end_pos_data[2,:], facecolors="tab:red", edgecolors="tab:red")
# set_axes_equal(ax)
# plt.title("Position")
# plt.legend()
# plt.show()

# # #calculate velocity magnitudes
# start_vel_mag_data = np.linalg.norm(start_vel_data[0:3,:],2,0)
# end_vel_mag_data = np.linalg.norm(end_vel_data[0:3,:],2,0)

# ax2 = plt.figure().add_subplot()
# ax2.plot(start_vel_data[3,:],start_vel_mag_data, color="tab:blue", label ="start")
# ax2.plot(end_vel_data[3,:], end_vel_mag_data, color="tab:red",label ="end")
# plt.title("Velocity")
# plt.legend()
# plt.show()