import numpy as np
from scipy.optimize import Bounds
from trajectory_generation.constraint_data_structures.waypoint_data import WaypointData

def get_control_points(variables, num_cont_pts, dimension):
    control_points = np.reshape(variables[0:num_cont_pts*dimension], \
                (dimension,num_cont_pts))
    return control_points

def get_scale_factor(variables, num_cont_pts, dimension):
    scale_factor = variables[num_cont_pts*dimension]
    return scale_factor
    
def get_intermediate_waypoint_scale_times(variables, num_middle_waypoints):
    intermediate_waypoint_times = variables[-num_middle_waypoints:]
    return intermediate_waypoint_times

def get_waypoint_scalars(variables, num_waypoint_scalars, num_cont_pts, dimension):
    start_index = num_cont_pts*dimension+1
    waypoint_scalars = variables[start_index:start_index+num_waypoint_scalars]
    return waypoint_scalars

def create_initial_objective_variables(num_cont_pts: int, point_sequence, waypoint_data: WaypointData, dimension: int, order: int):
    waypoint_sequence = waypoint_data.get_waypoint_locations()
    num_intermediate_waypoints = waypoint_data.get_num_intermediate_waypoints()
    control_points = create_initial_control_points(num_cont_pts, point_sequence, dimension)
    scale_factor = 1
    waypoint_scalar = 1
    variables = np.concatenate((control_points.flatten(),[scale_factor]))
    if waypoint_data.start_waypoint.direction is not None:
        variables = np.concatenate((variables,[waypoint_scalar]))
    if waypoint_data.end_waypoint.direction is not None:
        variables = np.concatenate((variables,[waypoint_scalar]))
    if (num_intermediate_waypoints > 0):
        intermediate_waypoint_time_scales = create_initial_intermediate_waypoint_time_scales(waypoint_sequence, num_cont_pts, order)
        variables = np.concatenate((variables, intermediate_waypoint_time_scales))
    return variables

def create_objective_variable_bounds(num_cont_pts, waypoint_data: WaypointData, dimension, order):
    num_intermediate_waypoints = waypoint_data.get_num_intermediate_waypoints()
    num_waypoint_scalars = waypoint_data.get_num_waypoint_scalars()
    lower_bounds = np.zeros(num_cont_pts*dimension + 1 + num_intermediate_waypoints + num_waypoint_scalars) - np.inf
    upper_bounds = np.zeros(num_cont_pts*dimension + 1 + num_intermediate_waypoints + num_waypoint_scalars) + np.inf
    start_index = num_cont_pts*dimension
    lower_bounds[start_index:start_index+num_waypoint_scalars] = 10e-8
    if num_intermediate_waypoints > 0:
        num_intervals = num_cont_pts - order
        upper_bounds[-num_intermediate_waypoints:] = num_intervals
        lower_bounds[-num_intermediate_waypoints:] = 0
    return Bounds(lb=lower_bounds, ub=upper_bounds)

def create_initial_control_points(total_num_cont_pts, point_sequence, dimension):
    num_segments = np.shape(point_sequence)[1] - 1
    if num_segments < 2:
        start_point = point_sequence[:,0]
        end_point = point_sequence[:,1]
        control_points = np.linspace(start_point,end_point,total_num_cont_pts).T
    else:
        control_points = np.empty(shape=(dimension,total_num_cont_pts))
        distances = np.linalg.norm(point_sequence[:,1:] - point_sequence[:,0:-1],2,0)
        for i in range(num_segments-1):
            distances[i+1] = distances[i+1] + distances[i]
        distance_between_cont_pts = distances[num_segments-1] / (total_num_cont_pts-1)
        segment_num = 0
        current_distance = 0
        prev_point_location = point_sequence[:,0]
        step_distance = 0
        for i in range(total_num_cont_pts-1):
            interval_start_point = point_sequence[:,segment_num]
            interval_end_point = point_sequence[:,segment_num+1]
            vector_to_point = interval_end_point - interval_start_point
            unit_vector_to_point = vector_to_point / (np.linalg.norm(vector_to_point))
            control_points[:,i] = prev_point_location + unit_vector_to_point*step_distance
            prev_point_location = control_points[:,i]
            step_distance = distance_between_cont_pts
            current_distance = current_distance + step_distance
            if distances[segment_num] < current_distance:
                step_distance = current_distance - distances[segment_num]
                segment_num += 1
                prev_point_location = point_sequence[:,segment_num]
        control_points[:,-1] = point_sequence[:,-1]
    return control_points

def create_initial_intermediate_waypoint_time_scales(point_sequence, num_cont_pts, order):
    num_intervals = num_cont_pts - order
    num_segments = np.shape(point_sequence)[1] - 1
    intermediate_waypoint_times = np.array([0.5])
    if num_segments > 2:
        distances = np.linalg.norm(point_sequence[:,1:] - point_sequence[:,0:-1],2,0)
        for i in range(num_segments-1):
            distances[i+1] = distances[i+1] + distances[i]
        norm_distances = distances/distances[num_segments-1]
        intermediate_waypoint_times = norm_distances[0:-1]*num_intervals
    return intermediate_waypoint_times