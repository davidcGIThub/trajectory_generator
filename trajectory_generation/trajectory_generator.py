"""
This module generates a 3rd order B-spline path between two waypoints,
waypoint directions, curvature constraint, and adjoining 
safe flight corridors.
"""
import os
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, Bounds
from trajectory_generation.constraint_functions.obstacle_constraints import ObstacleConstraints
from trajectory_generation.constraint_functions.turning_constraints import TurningConstraints
from trajectory_generation.control_point_conversions.bspline_to_minvo import get_composite_bspline_to_minvo_conversion_matrix
from trajectory_generation.constraint_data_structures.safe_flight_corridor import SFC_Data, SFC
from trajectory_generation.constraint_data_structures.obstacle import Obstacle
from trajectory_generation.constraint_data_structures.waypoint_data import Waypoint, WaypointData
from trajectory_generation.constraint_data_structures.dynamic_bounds import DerivativeBounds, TurningBound
from trajectory_generation.objectives.objective_variables import create_initial_objective_variables, \
    get_objective_variables, create_objective_variable_bounds
from trajectory_generation.objectives.objective_functions import minimize_acceleration_control_points_objective_function, \
    minimize_velocity_control_points_objective_function, minimize_jerk_control_points_objective_function
from trajectory_generation.constraint_functions.waypoint_constraints import create_terminal_waypoint_location_constraint, \
    create_intermediate_waypoint_location_constraints, create_intermediate_waypoint_time_scale_constraint, \
    create_terminal_waypoint_derivative_constraints
from trajectory_generation.constraint_functions.derivative_constraints import create_derivatives_constraint
from trajectory_generation.constraint_functions.sfc_constraints import create_safe_flight_corridor_constraint
from trajectory_generation.constraint_data_structures.constraint_function_data import ConstraintFunctionData
import time

class TrajectoryGenerator:
    """
    This class generates a 3rd order B-spline path between two waypoints,
    waypoint directions, curvature constraint, and adjoining 
    safe flight corridors.
    """

### TODO ####
# 2. add checks to make sure constraints are feasible with eachother
# 3. Change obstacle constraints to check only intervals that have an obstacle in the same SFC

    def __init__(self, dimension: int, 
                 num_intervals_free_space: int = 5):
        self._dimension = dimension
        self._order = 3
        self._turning_const_obj = TurningConstraints(self._dimension)
        self._obstacle_cons_obj = ObstacleConstraints(self._dimension)
        self._num_intervals_free_space = num_intervals_free_space
        
    def generate_trajectory(self, waypoint_data: WaypointData, derivative_bounds: DerivativeBounds = None, turning_bound: TurningBound = None,
                sfc_data: SFC_Data = None, obstacles: list = None, objective_function_type: str = "minimal_velocity_path"):
        num_intervals = self.__get_num_intervals(sfc_data)
        num_intermediate_waypoints = waypoint_data.get_num_intermediate_waypoints()
        point_sequence = self.__get_point_sequence(waypoint_data, sfc_data)
        num_cont_pts = self.__get_num_control_points(num_intervals)
        constraints, constraint_functions = self.__get_constraints(num_cont_pts, waypoint_data, \
            derivative_bounds, turning_bound, sfc_data, obstacles, num_intermediate_waypoints)
        objectiveFunction = self.__get_objective_function(objective_function_type)
        objective_variable_bounds = create_objective_variable_bounds(num_cont_pts, num_intermediate_waypoints, self._dimension, self._order)
        waypoint_sequence = waypoint_data.get_waypoint_locations()
        optimization_variables = create_initial_objective_variables(num_cont_pts, point_sequence, 
            num_intermediate_waypoints, waypoint_sequence, self._dimension, self._order)
        minimize_options = {'disp': False} #, 'maxiter': self.maxiter, 'ftol': tol}
        # perform optimization
        result = minimize(
            objectiveFunction,
            x0=optimization_variables,
            args=(num_cont_pts, self._dimension),
            method='SLSQP', 
            bounds=objective_variable_bounds,
            constraints=constraints, 
            options = minimize_options)
        optimized_control_points, optimized_scale_factor = self.__get_optimized_results(result, num_cont_pts)
        print("succes: " , result.success)
        print("status: " , result.status)
        print("message: " , result.message)
        print("num iterations: " , result.nit)
        # print("result: " , result)
        self.__display_violated_constraints(constraint_functions, result.success, result.x)
        return optimized_control_points, optimized_scale_factor
    
    def __get_optimized_results(self, result, num_cont_pts):
        control_points = np.reshape(result.x[0:num_cont_pts*self._dimension] ,(self._dimension,num_cont_pts))
        scale_factor = result.x[num_cont_pts*self._dimension]
        return control_points, scale_factor
    
    def __display_violated_constraints(self, constraint_functions: 'list[ConstraintFunctionData]',  success: bool, optimized_result):
        # if not success:
        num_constraint_functions = len(constraint_functions)
        constraint_tolerance = 10e-4
        # control_points, scale_factor = self.__get_optimized_results(optimized_result, num_cont_pts) 
        for i in range(num_constraint_functions):
            constraint_data = constraint_functions[i]
            constraint_function = constraint_data.constraint_function
            lower_bound = constraint_data.lower_bound
            upper_bound = constraint_data.upper_bound
            constraint_name = constraint_function.__name__
            constraints_key = constraint_data.key
            if constraint_name == "derivatives_constraint_function":
                output = constraint_function(optimized_result)
                print("output: " , output)
                violations = output > upper_bound + constraint_tolerance or output < lower_bound - constraint_tolerance
                if any(output > upper_bound + constraint_tolerance) or any(output < lower_bound - constraint_tolerance):
                    print("Derivative Constraints Violated: " , constraints_key[violations])

# def create_derivatives_constraint(derivative_bounds: DerivativeBounds, num_cont_pts, dimension, order):
#     num_vel_cont_pts = num_cont_pts - 1
#     M_v = get_composite_bspline_to_bezier_conversion_matrix(num_vel_cont_pts, order-1)
#     constraints = initialize_derivative_constraint_array(derivative_bounds)
#     def derivatives_constraint_function(variables):
#         control_points, scale_factor = get_objective_variables(variables, num_cont_pts, dimension)
#         velocity_control_points = (control_points[:,1:] - control_points[:,0:-1])/scale_factor
#         count = 0
#         if derivative_bounds.max_velocity is not None:
#             bezier_velocity_control_points = np.transpose(np.dot(M_v, np.transpose(velocity_control_points)))
#             constraints[count] = calculate_velocity_constraint(bezier_velocity_control_points, derivative_bounds)
#             count += 1
#             if derivative_bounds.max_upward_velocity is not None and dimension == 3:
#                 constraints[count] = calculate_upward_velocity_constraint(bezier_velocity_control_points, derivative_bounds)
#                 count += 1
#             if derivative_bounds.max_horizontal_velocity is not None and dimension == 3:
#                 constraints[count] = calculate_horizontal_velocity_constraint(bezier_velocity_control_points, derivative_bounds)
#                 count += 1
#         if derivative_bounds.max_acceleration is not None:
#             acceleration_constraint = calculate_acceleration_constraint(derivative_bounds, velocity_control_points, scale_factor)
#             constraints[count] = acceleration_constraint
#             count += 1
#         return constraints
#     lower_bound = - np.inf
#     upper_bound = 0
#     derivatives_constraint = NonlinearConstraint(derivatives_constraint_function , lb = lower_bound, ub = upper_bound)
#     constraint_function = derivatives_constraint_function
#     return derivatives_constraint, constraint_function

    
    
# SLSQP options:
# ftol : float
# Precision goal for the value of f in the stopping criterion.
# eps : float
# Step size used for numerical approximation of the jacobian.
# maxiter : int
# Maximum number of iterations.
# maxiter : int
# Maximum number of iterations to perform.
# disp : bool
# Set to True to print convergence messages.
    
    def __get_objective_function(self, objective_function_type):
        if objective_function_type == "minimal_distance_path":
            return minimize_velocity_control_points_objective_function
        elif objective_function_type == "minimal_velocity_path":
            return minimize_acceleration_control_points_objective_function
        elif objective_function_type == "minimal_acceleration_path":
            return minimize_jerk_control_points_objective_function
        else:
            raise Exception("Error, Invalid objective function type")

    def __get_num_intervals(self, sfc_data: SFC_Data):
        num_intervals = self._num_intervals_free_space
        if sfc_data is not None:
            num_intervals = sfc_data.get_num_intervals()
        return num_intervals
    
    def __get_num_control_points(self, num_intervals):
        num_control_points = num_intervals + self._order
        return int(num_control_points)
        
    def __get_point_sequence(self, waypoint_data:WaypointData, sfc_data:SFC_Data = None):
        if sfc_data is None:
            point_sequence = waypoint_data.get_waypoint_locations()
            return point_sequence
        else:
            return sfc_data.get_point_sequence()

    def __get_constraints(self, num_cont_pts: int, waypoint_data: WaypointData, 
            derivative_bounds: DerivativeBounds, turning_bound: TurningBound, 
            sfc_data: SFC_Data, obstacles: list, num_intermediate_waypoints):
        start_waypoint_location_constraint = create_terminal_waypoint_location_constraint(waypoint_data.start_waypoint, num_cont_pts, num_intermediate_waypoints, self._order)
        end_waypoint_location_constraint = create_terminal_waypoint_location_constraint(waypoint_data.end_waypoint, num_cont_pts, num_intermediate_waypoints, self._order)
        constraints = [start_waypoint_location_constraint, end_waypoint_location_constraint]
        constraint_functions = []
        if waypoint_data.start_waypoint.checkIfDerivativesActive():
            start_waypoint_derivatives_constraint = create_terminal_waypoint_derivative_constraints(waypoint_data.start_waypoint, num_cont_pts)
            constraints.append(start_waypoint_derivatives_constraint)
        if waypoint_data.end_waypoint.checkIfDerivativesActive():
            end_waypoint_derivatives_constraint = create_terminal_waypoint_derivative_constraints(waypoint_data.end_waypoint, num_cont_pts)
            constraints.append(end_waypoint_derivatives_constraint)
        if waypoint_data.intermediate_locations is not None:
            intermediate_waypoint_location_constraints = create_intermediate_waypoint_location_constraints(waypoint_data.intermediate_locations, num_cont_pts, num_intermediate_waypoints, self._order)
            constraints.append(intermediate_waypoint_location_constraints)
            if(num_intermediate_waypoints > 1):
                intermediate_waypoint_time_constraints = create_intermediate_waypoint_time_scale_constraint(num_cont_pts, num_intermediate_waypoints, self._dimension)
                constraints.append(intermediate_waypoint_time_constraints)
        if derivative_bounds is not None and derivative_bounds.checkIfDerivativesActive() is not None:
            derivatives_constraint, derivatives_constraint_function = create_derivatives_constraint( \
                derivative_bounds, num_cont_pts, self._dimension, self._order)
            constraints.append(derivatives_constraint)
            constraint_functions.append(derivatives_constraint_function)
        if turning_bound is not None and turning_bound.checkIfTurningBoundActive():
            turning_constraint = self._turning_const_obj.create_turning_constraint(turning_bound, num_cont_pts, self._dimension)
            constraints.append(turning_constraint)
        if sfc_data is not None:
            sfc_constraint = create_safe_flight_corridor_constraint(sfc_data, num_cont_pts, num_intermediate_waypoints, self._dimension, self._order)
            constraints.append(sfc_constraint)
        if (obstacles != None):
            obstacle_constraint = self._obstacle_cons_obj.create_obstacle_constraints(obstacles,num_cont_pts, self._dimension)
            constraints.append(obstacle_constraint)
        return tuple(constraints), constraint_functions
        