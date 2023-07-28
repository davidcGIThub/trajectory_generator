import numpy as np
from trajectory_generation.objectives.objective_variables import get_control_points, get_scale_factor 
    
def minimize_jerk_control_points_and_time_objective_function(variables, num_cont_pts, dimension):
    control_points = get_control_points(variables, num_cont_pts, dimension)
    scale_factor = get_scale_factor(variables, num_cont_pts, dimension)
    jerk_cps = control_points[:,3:] - 3*control_points[:,2:-1] + 3*control_points[:,1:-2] - control_points[:,0:-3]
    square_jerk_control_points = np.sum(jerk_cps**2,0)
    objective = np.sum(square_jerk_control_points) * scale_factor
    return objective

def minimize_velocity_control_points_and_time_objective_function(variables, num_cont_pts, dimension):
    control_points = get_control_points(variables, num_cont_pts, dimension)
    scale_factor = get_scale_factor(variables, num_cont_pts, dimension)
    velocity_cps =  control_points[:,0:-1] - control_points[:,1:]
    velocity_control_points_squared_sum = np.sum(velocity_cps**2,0)
    objective = np.sum(velocity_control_points_squared_sum) + scale_factor
    return objective

def minimize_acceleration_control_points_and_time_objective_function(variables, num_cont_pts, dimension):
    control_points = get_control_points(variables, num_cont_pts, dimension)
    scale_factor = get_scale_factor(variables, num_cont_pts, dimension)
    acceleration_cps =  control_points[:,2:] - 2*control_points[:,1:-1] + control_points[:,0:-2]
    accel_control_points_squared_sum = np.sum(acceleration_cps**2,0)
    objective = np.sum(accel_control_points_squared_sum) + scale_factor
    return objective

def minimize_jerk_control_points_objective_function(variables, num_cont_pts, dimension):
    control_points = get_control_points(variables, num_cont_pts, dimension)
    jerk_cps = control_points[:,3:] - 3*control_points[:,2:-1] + 3*control_points[:,1:-2] - control_points[:,0:-3]
    square_jerk_control_points = np.sum(jerk_cps**2,0)
    objective = np.sum(square_jerk_control_points)
    return objective

def minimize_velocity_control_points_objective_function(variables, num_cont_pts, dimension):
    control_points = get_control_points(variables, num_cont_pts, dimension)
    velocity_cps =  control_points[:,0:-1] - control_points[:,1:]
    velocity_control_points_squared_sum = np.sum(velocity_cps**2,0)
    objective = np.sum(velocity_control_points_squared_sum)
    return objective

def minimize_acceleration_control_points_objective_function(variables, num_cont_pts, dimension):
    control_points = get_control_points(variables, num_cont_pts, dimension)
    acceleration_cps =  control_points[:,2:] - 2*control_points[:,1:-1] + control_points[:,0:-2]
    accel_control_points_squared_sum = np.sum(acceleration_cps**2,0)
    objective = np.sum(accel_control_points_squared_sum)
    return objective