import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint
from trajectory_generation.constraint_data_structures.waypoint_data import Waypoint
from trajectory_generation.matrix_evaluation import get_M_matrix, evaluate_point_on_interval, \
    evaluate_point_derivative_on_interval
from trajectory_generation.objectives.objective_variables_orbit import get_control_points, get_scale_factor, \
    get_waypoint_scalars, get_intermediate_waypoint_scale_times, get_orbit_radius
from trajectory_generation.constraint_data_structures.constraint_function_data import ConstraintFunctionData

def create_terminal_waypoint_location_constraint(waypoint: Waypoint, num_cont_pts, num_intermediate_waypoints, num_waypoint_scalars, order):
    num_extra_spaces = 1 + num_intermediate_waypoints + num_waypoint_scalars
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
    def terminal_waypoint_location_constraint(variables):
        constraints = np.dot(constraint_matrix, variables).flatten()
        return constraints
    lower_bound = waypoint.location.flatten()
    upper_bound = waypoint.location.flatten()
    print("constraint_matrix: " , np.shape(constraint_matrix))
    print("lower bound: " , np.shape(lower_bound))
    print("upper bound: " , np.shape(upper_bound))
    if d ==2: constraints_key = np.array(["x","y"])
    else: constraints_key = np.array(["x","y","z"])
    if waypoint.side == "start": constraint_class = "Start_Waypoint_Location"
    else: constraint_class = "End_Waypoint_Location"
    constraint_function_data = ConstraintFunctionData(terminal_waypoint_location_constraint, lower_bound, upper_bound,constraints_key,constraint_class)
    constraint = LinearConstraint(constraint_matrix, lb=lower_bound, ub=upper_bound)
    return constraint, constraint_function_data

def create_zero_velocity_terminal_waypoint_constraint(waypoint: Waypoint, num_cont_pts, num_intermediate_waypoints, num_waypoint_scalars, order):
    num_extra_spaces = 1 + num_intermediate_waypoints + num_waypoint_scalars
    n = num_cont_pts
    k = order
    d = waypoint.dimension
    constraint_matrix = np.zeros((d*k,n*d+num_extra_spaces))
    if waypoint.side == "start":
        for i in range(d):
            constraint_matrix[i*k:i*k+k , i*n:i*n+k] = np.eye(k)
    elif waypoint.side == "end":
        for i in range(d):
            constraint_matrix[i*k:i*k+k , (i+1)*n-k:(i+1)*n] = np.eye(k)
    else:
        raise Exception("Error: Not a start or end Waypoint")
    def zero_velocity_terminal_waypoint_location_constraint(variables):
        constraints = np.dot(constraint_matrix, variables).flatten()
        return constraints
    lower_bound = np.repeat(waypoint.location.flatten(),k)
    upper_bound = np.repeat(waypoint.location.flatten(),k)
    if d == 2: constraints_key = np.repeat(np.array(["x","y"]),k)
    else: constraints_key = np.repeat(np.array(["x","y","z"]), k)
    if waypoint.side == "start": constraint_class = "Zero_Velocity_Start_Waypoint_Location"
    else: constraint_class = "Zero_Velocity_End_Waypoint_Location"
    constraint_function_data = ConstraintFunctionData(zero_velocity_terminal_waypoint_location_constraint, lower_bound, upper_bound,constraints_key,constraint_class)
    constraint = LinearConstraint(constraint_matrix, lb=lower_bound, ub=upper_bound)
    return constraint, constraint_function_data

def create_terminal_waypoint_derivative_constraints(waypoint: Waypoint, num_cont_pts: int, num_waypoint_scalars: int, isIndirect: bool = False):
    lower_bound = 0
    upper_bound = 0
    directionIsActive = waypoint.checkIfDirectionActive()
    velocityIsActive = waypoint.checkIfVelocityActive()
    accelerationIsActive = waypoint.checkIfAccelerationActive()
    side = waypoint.side
    if directionIsActive:
        direction_desired = waypoint.direction.flatten()
        if side == "start": scalar_ind = 0
        else: scalar_ind = -1
    if velocityIsActive:
        velocity_desired = waypoint.velocity.flatten()
        vel_des_mag = np.linalg.norm(velocity_desired)
    else:
        vel_des_mag = None
    if accelerationIsActive:
        acceleration_desired = waypoint.acceleration.flatten()
    constraints, constraints_key = initialize_derivative_constraints(waypoint)
    def waypoint_derivative_constraint_function(variables):
        control_points = get_control_points(variables, num_cont_pts, waypoint.dimension)
        scale_factor = get_scale_factor(variables, num_cont_pts, waypoint.dimension)
        marker = 0
        if directionIsActive:
            waypoint_scalars = get_waypoint_scalars(variables, num_waypoint_scalars, num_cont_pts, waypoint.dimension)
            waypoint_scalar = waypoint_scalars[scalar_ind]
            if vel_des_mag is not None and vel_des_mag <= 0:
                direction = get_terminal_direction_with_zero_velocity(side, control_points, waypoint_scalar)
            else:
                if isIndirect:
                    direction = get_terminal_direction_indirect(side, control_points, waypoint_scalar)
                else:
                    direction = get_terminal_direction(side, control_points, waypoint_scalar)
            constraints[marker:marker+waypoint.dimension] = (direction - direction_desired).flatten()
            marker += waypoint.dimension
        if velocityIsActive and vel_des_mag > 0:
            velocity = get_terminal_velocity(side, control_points, scale_factor)
            constraints[marker:marker+waypoint.dimension] = (velocity - velocity_desired).flatten()
            marker += waypoint.dimension
        if accelerationIsActive:
            acceleration = get_terminal_acceleration(side, control_points, scale_factor)
            constraints[marker:marker+waypoint.dimension] = (acceleration - acceleration_desired).flatten()
        return constraints
    if waypoint.side == "start": constraint_class = "Start_Waypoint_Derivatives"
    else: constraint_class = "End_Waypoint_Derivatives"
    constraint_function_data = ConstraintFunctionData(waypoint_derivative_constraint_function, lower_bound, upper_bound,constraints_key,constraint_class)
    waypoint_derivative_constraint = NonlinearConstraint(waypoint_derivative_constraint_function, lb= lower_bound, ub=upper_bound)
    return waypoint_derivative_constraint, constraint_function_data

def create_target_constraint(waypoint: Waypoint, num_cont_pts, num_intermediate_waypoints, num_waypoint_scalars, order):
    initial_point = waypoint.location
    target_velocity = waypoint.velocity
    num_extra_spaces = 1 + num_intermediate_waypoints + num_waypoint_scalars
    n = num_cont_pts
    mew = num_cont_pts - order
    k = order
    d = np.shape(initial_point)[0]
    constraint_matrix = np.zeros((d,n*d+num_extra_spaces))
    M_ = get_M_matrix(order)
    Gamma_f = np.ones((order+1,1))
    M_Gamma_f_T = np.dot(M_,Gamma_f).T
    for i in range(d):
        constraint_matrix[i, (i+1)*n-k-1 : (i+1)*n] = M_Gamma_f_T
    constraint_matrix[:,n*d] = -mew*target_velocity.flatten()
    def terminal_target_location_constraint(variables):
        constraints = np.dot(constraint_matrix, variables).flatten()
        return constraints
    lower_bound = initial_point.flatten()
    upper_bound = initial_point.flatten()
    if d ==2: constraints_key = np.array(["x","y"])
    else: constraints_key = np.array(["x","y","z"])
    constraint_class = "Target_Location"
    constraint_function_data = ConstraintFunctionData(terminal_target_location_constraint, lower_bound, upper_bound,constraints_key,constraint_class)
    constraint = LinearConstraint(constraint_matrix, lb=lower_bound, ub=upper_bound)
    return constraint, constraint_function_data

def create_orbit_target_constraint(waypoint: Waypoint, num_cont_pts, num_intermediate_waypoints, num_waypoint_scalars, order):
    orbit_center = waypoint.location
    orbit_linear_speed = waypoint.velocity
    num_extra_spaces = 2 + num_intermediate_waypoints + num_waypoint_scalars # +2 for scale factor and orbit radius
    n = num_cont_pts
    mew = num_cont_pts - order
    k = order
    d = np.shape(orbit_center)[0]
    constraint_matrix = np.zeros((d,n*d+num_extra_spaces))
    M_ = get_M_matrix(order)
    Gamma_f = np.ones((order+1,1))
    M_Gamma_f_T = np.dot(M_,Gamma_f).T
    for i in range(d):
        constraint_matrix[i, (i+1)*n-k-1 : (i+1)*n] = M_Gamma_f_T
    def terminal_orbit_target_location_constraint(variables):
        R = get_orbit_radius(variables)
        scale_factor = get_scale_factor(variables)
        angular_rate = orbit_linear_speed/R
        end_time = scale_factor*mew
        x = R*np.cos(angular_rate*end_time) + orbit_center.item(0)
        y = R*np.sin(angular_rate*end_time) + orbit_center.item(1)
        if d == 2:
            end_point = np.array([x,y])
        else:
            z = 0
            end_point = np.array([x,y,z])
        constraints = np.dot(constraint_matrix, variables).flatten() - end_point
        return constraints
    lower_bound = 0
    upper_bound = 0
    if d ==2: constraints_key = np.array(["x","y"])
    else: constraints_key = np.array(["x","y","z"])
    constraint_class = "Target_Orbit_Location"
    constraint_function_data = ConstraintFunctionData(terminal_orbit_target_location_constraint, lower_bound, upper_bound,constraints_key,constraint_class)
    constraint = NonlinearConstraint(terminal_orbit_target_location_constraint, lb=lower_bound, ub=upper_bound)
    return constraint, constraint_function_data

def create_terminal_waypoint_direction_constraint(waypoint: Waypoint, num_cont_pts: int, num_waypoint_scalars: int, isIndirect: bool = False):
    lower_bound = 0
    upper_bound = 0
    side = waypoint.side
    direction_desired = waypoint.direction.flatten()
    if side == "start": scalar_ind = 0
    else: scalar_ind = -1
    constraints, constraints_key = initialize_direction_constraint(waypoint)
    def waypoint_direction_constraint_function(variables):
        control_points = get_control_points(variables, num_cont_pts, waypoint.dimension)
        waypoint_scalars = get_waypoint_scalars(variables, num_waypoint_scalars, num_cont_pts, waypoint.dimension)
        waypoint_scalar = waypoint_scalars[scalar_ind]
        if isIndirect:
            direction = get_terminal_direction_indirect(side, control_points, waypoint_scalar)
        else:
            direction = get_terminal_direction(side, control_points, waypoint_scalar)
        constraints = (direction - direction_desired).flatten()
        return constraints
    if waypoint.side == "start": constraint_class = "Start_Waypoint_Direction"
    else: constraint_class = "End_Waypoint_Direction"
    constraint_function_data = ConstraintFunctionData(waypoint_direction_constraint_function, lower_bound, upper_bound,constraints_key,constraint_class)
    waypoint_derivative_constraint = NonlinearConstraint(waypoint_direction_constraint_function, lb= lower_bound, ub=upper_bound)
    return waypoint_derivative_constraint, constraint_function_data

def initialize_derivative_constraints(waypoint: Waypoint):
    length = 0
    constraints_key = np.array([])
    if waypoint.checkIfDirectionActive():
        length += waypoint.dimension
        constraints_key = np.concatenate((constraints_key, np.array(["x dir"])))
        constraints_key = np.concatenate((constraints_key, np.array(["y dir"])))
        if waypoint.dimension == 3: constraints_key = np.concatenate((constraints_key, np.array(["z dir"])))
    if waypoint.checkIfVelocityActive() and np.linalg.norm(waypoint.velocity) > 0:
        length += waypoint.dimension
        constraints_key = np.concatenate((constraints_key, np.array(["x vel"])))
        constraints_key = np.concatenate((constraints_key, np.array(["y vel"])))
        if waypoint.dimension == 3: constraints_key = np.concatenate((constraints_key, np.array(["z vel"])))
    if waypoint.checkIfAccelerationActive():
        length += waypoint.dimension
        constraints_key = np.concatenate((constraints_key, np.repeat("acceleration", waypoint.dimension)))
        constraints_key = np.concatenate((constraints_key, np.array(["x accel"])))
        constraints_key = np.concatenate((constraints_key, np.array(["y accel"])))
        if waypoint.dimension == 3: constraints_key = np.concatenate((constraints_key, np.array(["z accel"])))
    constraints = np.zeros(length)
    return constraints, constraints_key

def initialize_direction_constraint(waypoint:Waypoint):
    length = 0
    constraints_key = np.array([])
    length += waypoint.dimension
    constraints_key = np.concatenate((constraints_key, np.array(["x dir"])))
    constraints_key = np.concatenate((constraints_key, np.array(["y dir"])))
    if waypoint.dimension == 3: constraints_key = np.concatenate((constraints_key, np.array(["z dir"])))
    constraints = np.zeros(length)
    return constraints, constraints_key

def get_terminal_location(side, control_points):
    if side == "start":
        direction = control_points[:,0]/6 + 2*control_points[:,1]/3 + control_points[:,2]/6
    if side == "end":
        direction = control_points[:,-3]/6 + 2*control_points[:,-2]/3 + control_points[:,-1]/6
    return direction

def get_terminal_direction_indirect(side, control_points, waypoint_scalar):
    if side == "start":
        direction = (control_points[:,2] - control_points[:,0])/(2*waypoint_scalar)
    if side == "end":
        direction = waypoint_scalar*(control_points[:,-1] - control_points[:,-3])/(2*waypoint_scalar)
    return direction

def get_terminal_direction(side, control_points, waypoint_scalar):
    if side == "start":
        direction = waypoint_scalar*(control_points[:,2] - control_points[:,0])/2
    if side == "end":
        direction = waypoint_scalar*(control_points[:,-1] - control_points[:,-3])/2
    return direction

def get_terminal_direction_with_zero_velocity(side, control_points, waypoint_scalar):
    if side == "start":
        direction = waypoint_scalar*(control_points[:,3] - control_points[:,0])/2
    if side == "end":
        direction = waypoint_scalar*(control_points[:,-1] - control_points[:,-4])/2
    return direction

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

### convert to C++ code ###
def create_intermediate_waypoint_location_constraints(intermediate_locations, num_cont_pts, num_intermediate_waypoints, order):
    lower_bound = 0
    upper_bound = 0
    dimension = np.shape(intermediate_locations)[0]
    constraints = np.zeros((dimension, num_intermediate_waypoints))
    constraints_key = create_intermediate_waypoint_constraints_key(dimension, num_intermediate_waypoints)
    def intermediate_waypoint_constraint_function(variables):
        control_points = get_control_points(variables, num_cont_pts, dimension)
        intermediate_waypoint_scale_times = get_intermediate_waypoint_scale_times(variables, num_intermediate_waypoints)
        for i in range(num_intermediate_waypoints):
            desired_location = intermediate_locations[:,i]
            scale_time = intermediate_waypoint_scale_times[i]
            interval = int(scale_time)
            interval_cont_pts = control_points[:,interval:interval+order+1]
            location = evaluate_point_on_interval(interval_cont_pts, scale_time, interval, 1)
            constraints[:,i] = location.flatten() - desired_location
        return constraints.flatten()
    constraint_class = "Intermediate_Waypoint_Locations"
    constraint_function_data = ConstraintFunctionData(intermediate_waypoint_constraint_function, lower_bound, upper_bound,constraints_key,constraint_class)
    intermediate_waypoint_constraint = NonlinearConstraint(intermediate_waypoint_constraint_function, lb= lower_bound, ub=upper_bound)
    return intermediate_waypoint_constraint, constraint_function_data

### convert to C++ code ###
def create_intermediate_waypoint_velocity_constraints(intermediate_velocities, num_cont_pts, num_intermediate_waypoints, order):
    lower_bound = 0
    upper_bound = 0
    dimension = np.shape(intermediate_velocities)[0]
    constraints = np.zeros((dimension, num_intermediate_waypoints))
    constraints_key = create_intermediate_velocity_constraints_key(dimension, num_intermediate_waypoints)
    def intermediate_velocity_constraint_function(variables):
        control_points = get_control_points(variables, num_cont_pts, dimension)
        scale_factor = get_scale_factor(variables, num_cont_pts, dimension)
        intermediate_waypoint_scale_times = get_intermediate_waypoint_scale_times(variables, num_intermediate_waypoints)
        derivative_order = 1
        for i in range(num_intermediate_waypoints):
            desired_velocity = intermediate_velocities[:,i]
            scale_time = intermediate_waypoint_scale_times[i]
            interval = int(scale_time)
            interval_cont_pts = control_points[:,interval:interval+order+1]
            t_ = (scale_time - interval)*scale_factor
            velocity = evaluate_point_derivative_on_interval(interval_cont_pts, t_, 0, scale_factor,derivative_order)
            constraints[:,i] = velocity.flatten() - desired_velocity
        return constraints.flatten()
    constraint_class = "Intermediate_Waypoint_Velocities"
    ####
    constraint_function_data = ConstraintFunctionData(intermediate_velocity_constraint_function, lower_bound, upper_bound,constraints_key,constraint_class)
    intermediate_waypoint_constraint = NonlinearConstraint(intermediate_velocity_constraint_function, lb= lower_bound, ub=upper_bound)
    return intermediate_waypoint_constraint, constraint_function_data

def create_intermediate_waypoint_constraints_key(dimension, num_intermediate_waypoints):
    constraints_key = np.empty((dimension,num_intermediate_waypoints)).astype(str)
    for i in range(num_intermediate_waypoints):
        constraints_key[0,i] = "x" + str(i+2)
        constraints_key[1,i] = "y" + str(i+2)
        if dimension == 3: constraints_key[0,i] = "z " + str(i+2)
    return constraints_key.flatten()

def create_intermediate_velocity_constraints_key(dimension, num_intermediate_waypoints):
    constraints_key = np.empty((dimension,num_intermediate_waypoints)).astype(str)
    for i in range(num_intermediate_waypoints):
        constraints_key[0,i] = "xdot" + str(i+2)
        constraints_key[1,i] = "ydot" + str(i+2)
        if dimension == 3: constraints_key[0,i] = "zdot" + str(i+2)
    return constraints_key.flatten()


# def create_intermediate_waypoint_time_scale_constraint(num_cont_pts, num_intermediate_waypoints, dimension):
#     #ensures that waypoints are reached in thier proper order
#     num_extra_spaces = 1 + num_intermediate_waypoints
#     m = num_intermediate_waypoints
#     n = num_cont_pts
#     d = dimension
#     constraint_matrix = np.zeros((m-1,n*d+num_extra_spaces))
#     for i in range(m-1):
#         constraint_matrix[i,-i-1] = -1
#         constraint_matrix[i,-i-2] = 1
#     constraint = LinearConstraint(constraint_matrix, lb=-np.inf, ub=0)
#     return constraint