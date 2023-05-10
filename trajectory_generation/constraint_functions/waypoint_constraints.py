import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint
from trajectory_generation.constraint_data_structures.waypoint_data import Waypoint
from trajectory_generation.matrix_evaluation import get_M_matrix, evaluate_point_on_interval
from trajectory_generation.objectives.objective_variables import get_objective_variables, \
    get_intermediate_waypoint_scale_times
from trajectory_generation.constraint_data_structures.constraint_function_data import ConstraintFunctionData

def create_terminal_waypoint_location_constraint(waypoint: Waypoint, num_cont_pts, num_intermediate_waypoints, order):
    num_extra_spaces = 1 + num_intermediate_waypoints
    n = num_cont_pts
    k = order
    d = waypoint.dimension
    constraint_matrix = np.zeros((d,n*d+num_extra_spaces))
    M_ = get_M_matrix(order)
    if waypoint.side == "start":
        Gamma_0 = np.zeros((order+1,1))
        Gamma_0[order,0] = 1
        M_Gamma_0_T = np.dot(M_,Gamma_0).T
        for i in range(d):
            constraint_matrix[i,  i*n        : i*n+k+1] = M_Gamma_0_T
    elif waypoint.side == "end":
        Gamma_f = np.ones((order+1,1))
        M_Gamma_f_T = np.dot(M_,Gamma_f).T
        for i in range(d):
            constraint_matrix[i, (i+1)*n-k-1 : (i+1)*n] = M_Gamma_f_T
    else:
        raise Exception("Error: Not a start or end Waypoint")
    constraint = LinearConstraint(constraint_matrix, lb=waypoint.location.flatten(), ub=waypoint.location.flatten())
    return constraint

def create_intermediate_waypoint_location_constraints(intermediate_locations, num_cont_pts, num_intermediate_waypoints, order):
    lower_bound = 0
    upper_bound = 0
    dimension = np.shape(intermediate_locations)[0]
    def intermediate_waypoint_constraint_function(variables):
        control_points, scale_factor = get_objective_variables(variables, num_cont_pts, dimension)
        intermediate_waypoint_scale_times = get_intermediate_waypoint_scale_times(variables, num_intermediate_waypoints)
        constraints = np.zeros((dimension, num_intermediate_waypoints))
        for i in range(num_intermediate_waypoints):
            desired_location = intermediate_locations[:,i]
            scale_time = intermediate_waypoint_scale_times[i]
            interval = int(scale_time)
            interval_cont_pts = control_points[:,interval:interval+order+1]
            location = evaluate_point_on_interval(interval_cont_pts, scale_time, interval, 1)
            constraints[:,i] = location.flatten() - desired_location
        return constraints.flatten()
    intermediate_waypoint_constraint = NonlinearConstraint(intermediate_waypoint_constraint_function, lb= lower_bound, ub=upper_bound)
    return intermediate_waypoint_constraint

def create_intermediate_waypoint_time_scale_constraint(num_cont_pts, num_intermediate_waypoints, dimension):
    #ensures that waypoints are reached in thier proper order
    num_extra_spaces = 1 + num_intermediate_waypoints
    m = num_intermediate_waypoints
    n = num_cont_pts
    d = dimension
    constraint_matrix = np.zeros((m-1,n*d+num_extra_spaces))
    print("shape const matrix: " , np.shape(constraint_matrix))
    for i in range(m-1):
        constraint_matrix[i,-i-1] = -1
        constraint_matrix[i,-i-2] = 1
    constraint = LinearConstraint(constraint_matrix, lb=-np.inf, ub=0)
    return constraint

def create_terminal_waypoint_derivative_constraints(waypoint: Waypoint, num_cont_pts: int):
    lower_bound = 0
    upper_bound = 0
    if waypoint.checkIfVelocityActive():
        velocity_desired = waypoint.velocity.flatten()
    if waypoint.checkIfAccelerationActive():
        acceleration_desired = waypoint.acceleration.flatten()
    velocityIsActive = waypoint.checkIfVelocityActive()
    accelerationIsActive = waypoint.checkIfAccelerationActive()
    constraints = initialize_derivative_constraints(waypoint)
    side = waypoint.side
    def waypoint_derivative_constraint_function(variables):
        control_points, scale_factor = get_objective_variables(variables, num_cont_pts, waypoint.dimension)
        marker = 0
        if velocityIsActive:
            velocity = get_terminal_velocity(side, control_points, scale_factor)
            constraints[marker:waypoint.dimension] = (velocity - velocity_desired).flatten()
            marker += waypoint.dimension
        if accelerationIsActive:
            acceleration = get_terminal_acceleration(side, control_points, scale_factor)
            constraints[marker:marker+waypoint.dimension] = (acceleration - acceleration_desired).flatten()
        return constraints
    waypoint_derivative_constraint = NonlinearConstraint(waypoint_derivative_constraint_function, lb= lower_bound, ub=upper_bound)
    return waypoint_derivative_constraint

def initialize_derivative_constraints(waypoint: Waypoint):
    length = 0
    if waypoint.checkIfVelocityActive():
        length += waypoint.dimension
    if waypoint.checkIfAccelerationActive():
        length += waypoint.dimension
    return np.zeros(length)

def get_terminal_velocity(side, control_points, scale_factor):
    if side == "start":
        velocity = (control_points[:,2] - control_points[:,0])/(2*scale_factor)
    if side == "end":
        velocity = (control_points[:,-1] - control_points[:,-3])/(2*scale_factor)
    return velocity

def get_terminal_acceleration(side, control_points, scale_factor):
    if side == "start":
        acceleration = (control_points[:,0] - 2*control_points[:,1] + control_points[:,2])/(scale_factor*scale_factor)
    if side == "end":
        acceleration = (control_points[:,-3] - 2*control_points[:,-2] + control_points[:,-1])/(scale_factor*scale_factor)
    return acceleration