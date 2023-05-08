import numpy as np

def get_objective_function(objective_function_type):
    if objective_function_type == "minimal_distance_path":
        return minimize_velocity_control_points_objective_function
    elif objective_function_type == "minimal_velocity_path":
        return minimize_acceleration_control_points_objective_function
    elif objective_function_type == "minimal_acceleration_path":
        return minimize_jerk_control_points_objective_function
    else:
        raise Exception("Error, Invalid objective function type")
    
def minimize_jerk_control_points_objective_function(variables, num_cont_pts):
    control_points, scale_factor = get_objective_variables(variables, num_cont_pts)
    jerk_cps = control_points[:,3:] - 3*control_points[:,2:-1] + 3*control_points[:,1:-2] - control_points[:,0:-3]
    square_jerk_control_points = np.sum(jerk_cps**2,0)
    objective = np.sum(square_jerk_control_points) + scale_factor
    return objective

def minimize_velocity_control_points_objective_function(variables, num_cont_pts):
    control_points, scale_factor = get_objective_variables(variables, num_cont_pts)
    velocity_cps =  control_points[:,0:-1] - control_points[:,1:]
    velocity_control_points_squared_sum = np.sum(velocity_cps**2,0)
    objective = np.sum(velocity_control_points_squared_sum) + scale_factor
    return objective

def minimize_acceleration_control_points_objective_function(variables, num_cont_pts):
    control_points, scale_factor = get_objective_variables(variables, num_cont_pts)
    acceleration_cps =  control_points[:,2:] - 2*control_points[:,1:-1] + control_points[:,0:-2]
    accel_control_points_squared_sum = np.sum(acceleration_cps**2,0)
    objective = np.sum(accel_control_points_squared_sum) + scale_factor
    return objective