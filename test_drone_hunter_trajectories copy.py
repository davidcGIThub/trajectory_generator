from trajectory_generation.trajectory_generator import TrajectoryGenerator
from pose_to_numpy import get_start_and_end_condition_data

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from bsplinegenerator.bsplines import BsplineEvaluation
from trajectory_generation.trajectory_generator import TrajectoryGenerator
from trajectory_generation.constraint_data_structures.safe_flight_corridor import SFC_Data, get3DRotationAndTranslationFromPoints
from trajectory_generation.path_plotter import set_axes_equal
from trajectory_generation.constraint_data_structures.waypoint_data import Waypoint, WaypointData, plot3D_waypoints
from trajectory_generation.constraint_data_structures.dynamic_bounds import DerivativeBounds, TurningBound
from trajectory_generation.constraint_data_structures.safe_flight_corridor import SFC, plot_sfcs
from trajectory_generation.constraint_data_structures.constraints_container import ConstraintsContainer
import time

dimension = 3
order = 3
traj_objective_type = "minimal_acceleration_and_time_path"
# traj_objective_type = "minimal_velocity_and_time_path"
# traj_objective_type = "minimal_distance_and_time_path"
max_velocity = 20
max_acceleration = 3
# max_acceleration = 12.8
gravity = 9.8
max_upward_velocity = 6
gravity = None
max_upward_velocity = None
derivative_bounds = DerivativeBounds(max_velocity=max_velocity, max_acceleration=max_acceleration,
                                     gravity=gravity, max_upward_velocity=max_upward_velocity)

### Collect Flight Info
# bag_name = '../bag_parser/rosbag_defend_point.bag'
# isTarget = False
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


num_samples = 20
start_pos_data, start_vel_data, end_pos_data, end_vel_data = get_start_and_end_condition_data(bag_name, isTarget, num_samples)


# plt.figure()
# ax = plt.axes(projection='3d')
# fig = plt.figure(figsize=plt.figaspect(2.))
fig = plt.figure()
fig.suptitle('Trajectories')
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
for i in range(np.shape(start_pos_data)[1]-1):
    # extract conditions
    start_pos = start_pos_data[0:3,i][:,None]
    start_vel = start_vel_data[0:3,i][:,None]
    end_pos = end_pos_data[0:3,i][:,None]
    end_vel = end_vel_data[0:3,i][:,None]
    waypoint_1 = Waypoint(location=start_pos, velocity=start_vel)
    waypoint_2 = Waypoint(location=end_pos, velocity=end_vel, is_target=isTarget)
    waypoint_sequence = (waypoint_1, waypoint_2)
    waypoint_data = WaypointData(waypoint_sequence)
    traj_gen = TrajectoryGenerator(dimension)
    constraints_container = ConstraintsContainer(waypoint_constraints = waypoint_data, 
        derivative_constraints=derivative_bounds, turning_constraint=None, sfc_constraints=None, 
        obstacle_constraints=None)
    start_time = time.time()
    control_points, scale_factor, is_violation = traj_gen.generate_trajectory(constraints_container, traj_objective_type)
    end_time = time.time()
    evaluation_time = end_time - start_time
    if is_violation:
        pass
    else:
        # spline data
        spline_start_time_1 = 0
        bspline = BsplineEvaluation(control_points, order, spline_start_time_1, scale_factor, False)
        end_time_spline = bspline.get_end_time()
        number_data_points = 10000
        spline_data, time_data = bspline.get_spline_data(number_data_points)
        curvature_data, time_data = bspline.get_spline_curvature_data(number_data_points)
        velocity_mag_data, time_data = bspline.get_derivative_magnitude_data(number_data_points,1)
        velocity_data, time_data = bspline.get_spline_derivative_data(number_data_points, 1)
        acceleration_data, time_data = bspline.get_derivative_magnitude_data(number_data_points,2)
        end_time_spline = bspline.get_end_time()
        print("evaluation time: " , evaluation_time)
        if isTarget:
            end_point_target = end_pos + end_vel*end_time_spline
            predicted_target_track = np.vstack((end_pos.flatten(),end_point_target.flatten()))
        # ax.scatter(control_points[0,:], control_points[1,:], color="tab:orange")
        eval_time = np.round(evaluation_time,3)
        ax1.plot(spline_data[0,:], spline_data[1,:], spline_data[2,:], label =str(eval_time))
        ax2.plot(spline_data[0,:], spline_data[1,:], label =str(eval_time))
        if isTarget:
            ax1.plot(predicted_target_track[:,0],predicted_target_track[:,1],predicted_target_track[:,2],color="tab:brown",linestyle="dashed")
            ax2.plot(predicted_target_track[:,0],predicted_target_track[:,1],color="tab:brown",linestyle="dashed")
        ax3.plot(time_data, velocity_mag_data, label="velocity")
        ax3.plot(time_data, velocity_mag_data*0 + max_velocity, label="max_velocity", color="k")
        ax4.plot(time_data, acceleration_data, label="acceleration")
        ax4.plot(time_data, acceleration_data*0 + max_acceleration, label="max_acceleration", color="k")
        ax5.plot(time_data, -velocity_data[2,:])
        if max_upward_velocity is not None:
            ax5.plot(time_data, velocity_data[2,:]*0 + max_upward_velocity, color ="k")
ax1.plot(start_pos_data[0,:], start_pos_data[1,:], start_pos_data[2,:], color="tab:gray", alpha=0.4)
ax1.plot(end_pos_data[0,:], end_pos_data[1,:], end_pos_data[2,:], color="tab:brown", alpha=0.4)
ax1.scatter(start_pos_data[0,:], start_pos_data[1,:], start_pos_data[2,:],facecolors="tab:gray", edgecolors="tab:gray")
ax1.scatter(end_pos_data[0,:], end_pos_data[1,:], end_pos_data[2,:], facecolors="tab:brown", edgecolors="tab:brown")
ax2.plot(start_pos_data[0,:], start_pos_data[1,:], color="k", alpha=0.2)
ax2.plot(end_pos_data[0,:], end_pos_data[1,:], color="tab:brown", alpha=0.2)
ax2.scatter(start_pos_data[0,:], start_pos_data[1,:],facecolors="none", edgecolors="tab:gray",label ="start")
ax2.scatter(end_pos_data[0,:], end_pos_data[1,:], facecolors="tab:brown", edgecolors="tab:brown",label ="end")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax3.set_xlabel("time (sec)")
ax3.set_ylabel("velocity (m/s)")
ax4.set_xlabel("time (sec)")
ax4.set_ylabel("acceleration (m/s^2)")
# ax2.xla
set_axes_equal(ax1,dimension)
ax2.legend()
plt.show()


