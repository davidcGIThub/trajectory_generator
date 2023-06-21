"""
This module generates a 3rd order B-spline path between two waypoints,
waypoint directions, curvature constraint, and adjoining 
safe flight corridors.
"""
import os
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, OptimizeResult
from trajectory_generation.constraint_functions.obstacle_constraints import ObstacleConstraints
from trajectory_generation.constraint_functions.turning_constraints import TurningConstraints
from trajectory_generation.control_point_conversions.bspline_to_minvo import get_composite_bspline_to_minvo_conversion_matrix
from trajectory_generation.constraint_data_structures.safe_flight_corridor import SFC_Data, SFC
from trajectory_generation.constraint_data_structures.obstacle import Obstacle
from trajectory_generation.constraint_data_structures.waypoint_data import Waypoint, WaypointData
from trajectory_generation.constraint_data_structures.dynamic_bounds import DerivativeBounds, TurningBound
from trajectory_generation.objectives.objective_variables import create_initial_objective_variables, \
    create_objective_variable_bounds
from trajectory_generation.objectives.objective_functions import minimize_acceleration_control_points_objective_function, \
    minimize_velocity_control_points_objective_function, minimize_jerk_control_points_objective_function
from trajectory_generation.constraint_functions.waypoint_constraints import create_terminal_waypoint_location_constraint, \
    create_intermediate_waypoint_location_constraints, create_terminal_waypoint_derivative_constraints, \
    create_intermediate_waypoint_velocity_constraints # create_intermediate_waypoint_time_scale_constraint
from trajectory_generation.constraint_functions.derivative_constraints import create_derivatives_constraint
from trajectory_generation.constraint_functions.sfc_constraints import create_safe_flight_corridor_constraint
from trajectory_generation.constraint_data_structures.constraint_function_data import ConstraintFunctionData
from trajectory_generation.constraint_data_structures.constraints_container import ConstraintsContainer
from trajectory_generation.constraint_functions.waypoint_constraints import get_terminal_velocity, get_terminal_acceleration, get_terminal_location
import time


class TrajectoryGenerator:
    """
    This class generates a 3rd order B-spline trajectory between two waypoints,
    waypoint directions, curvature constraint, and adjoining 
    safe flight corridors.
    """

### TODO ####
# 2. add checks to make sure constraints are feasible with eachother
# 3. Change obstacle constraints to check only intervals that have an obstacle in the same SFC

    def __init__(self, dimension: int):
        self._dimension = dimension
        self._order = 3
        self._turning_const_obj = TurningConstraints(self._dimension)
        self._obstacle_cons_obj = ObstacleConstraints(self._dimension)

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

    def generate_trajectory(self, constraints_container: ConstraintsContainer, \
            objective_function_type: str = "minimal_velocity_path", 
            num_intervals_free_space: int = 5,
            initial_control_points: npt.NDArray[np.float64] = None, \
            initial_scale_factor: float = None):
        waypoint_data = constraints_container.waypoint_constraints
        derivative_bounds = constraints_container.derivative_constraints
        turning_bound = constraints_container.turning_constraint
        obstacles = constraints_container.obstacle_constraints
        sfc_data = constraints_container.sfc_constraints
        num_intervals = self.__get_num_intervals(sfc_data,num_intervals_free_space, initial_control_points)
        num_cont_pts = self.__get_num_control_points(num_intervals)
        point_sequence = self.__get_point_sequence(waypoint_data, sfc_data)
        constraints, constraint_data_list = self.__get_constraints(num_cont_pts, waypoint_data, \
            derivative_bounds, turning_bound, sfc_data, obstacles)
        objectiveFunction = self.__get_objective_function(objective_function_type)
        objective_variable_bounds = create_objective_variable_bounds(num_cont_pts, waypoint_data, self._dimension, self._order)
        optimization_variables = create_initial_objective_variables(num_cont_pts, point_sequence, 
            waypoint_data, self._dimension, self._order, initial_control_points, initial_scale_factor)
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
        print("result: " , result)
        self.__display_violated_constraints(constraint_data_list, result)
        return optimized_control_points, optimized_scale_factor
    
    def get_terminal_waypoint_properties(self, control_points:np.ndarray, scale_factor:float, side:str):
        terminal_location = get_terminal_location(side, control_points)[:,None]
        terminal_velocity = get_terminal_velocity(side, control_points, scale_factor)[:,None]
        terminal_acceleration = get_terminal_acceleration(side, control_points, scale_factor)[:,None]
        terminal_waypoint = Waypoint(location=terminal_location,
                                     velocity=terminal_velocity)  
        return terminal_waypoint
    
    def __get_optimized_results(self, result: OptimizeResult, num_cont_pts: int):
        control_points = np.reshape(result.x[0:num_cont_pts*self._dimension] ,(self._dimension,num_cont_pts))
        scale_factor = result.x[num_cont_pts*self._dimension]
        # print("type result: " , )
        return control_points, scale_factor
    
    def __get_objective_function(self, objective_function_type: str):
        if objective_function_type == "minimal_distance_path":
            return minimize_velocity_control_points_objective_function
        elif objective_function_type == "minimal_velocity_path":
            return minimize_acceleration_control_points_objective_function
        elif objective_function_type == "minimal_acceleration_path":
            return minimize_jerk_control_points_objective_function
        else:
            raise Exception("Error, Invalid objective function type")

    def __get_num_intervals(self, sfc_data: SFC_Data, 
                            num_intervals_free_space: int,
                            initial_control_points: npt.NDArray[np.float64] = None):
        if initial_control_points is not None:
            num_control_points = np.shape(initial_control_points)[1]
            num_intervals = num_control_points - self._order
        elif sfc_data is not None:
            num_intervals = sfc_data.get_num_intervals()
        else:
            num_intervals = num_intervals_free_space
        return num_intervals
    
    def __get_num_control_points(self, num_intervals: int):
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
            sfc_data: SFC_Data, obstacles: 'list[Obstacle]'):
        num_intermediate_waypoints = waypoint_data.get_num_intermediate_waypoints()
        num_waypoint_scalars = waypoint_data.get_num_waypoint_scalars()
        start_waypoint_location_constraint, start_waypoint_constraint_function_data = \
            create_terminal_waypoint_location_constraint(waypoint_data.start_waypoint, num_cont_pts, \
                num_intermediate_waypoints, num_waypoint_scalars, self._order)
        end_waypoint_location_constraint, end_waypoint_constraint_function_data = \
            create_terminal_waypoint_location_constraint(waypoint_data.end_waypoint, num_cont_pts, \
                num_intermediate_waypoints, num_waypoint_scalars, self._order)
        constraints = [start_waypoint_location_constraint, end_waypoint_location_constraint]
        constraint_functions_data = [start_waypoint_constraint_function_data, end_waypoint_constraint_function_data]
        if waypoint_data.start_waypoint.checkIfDerivativesActive():
            start_waypoint_derivatives_constraint, start_waypoint_derivatives_constraint_function_data = \
                create_terminal_waypoint_derivative_constraints(waypoint_data.start_waypoint, num_cont_pts, num_waypoint_scalars)
            constraints.append(start_waypoint_derivatives_constraint)
            constraint_functions_data.append(start_waypoint_derivatives_constraint_function_data)
        if waypoint_data.end_waypoint.checkIfDerivativesActive():
            end_waypoint_derivatives_constraint, end_waypoint_derivatives_constraint_function_data = \
                create_terminal_waypoint_derivative_constraints(waypoint_data.end_waypoint, num_cont_pts, num_waypoint_scalars)
            constraints.append(end_waypoint_derivatives_constraint)
            constraint_functions_data.append(end_waypoint_derivatives_constraint_function_data)
        if waypoint_data.intermediate_locations is not None:
            intermediate_waypoint_location_constraints, intermediate_waypoint_location_constraint_function_data  = \
                create_intermediate_waypoint_location_constraints(waypoint_data.intermediate_locations, \
                    num_cont_pts, num_intermediate_waypoints, self._order)
            constraints.append(intermediate_waypoint_location_constraints)
            constraint_functions_data.append(intermediate_waypoint_location_constraint_function_data)
            # Ensures that the waypoint sequence is followed in order
            # if(num_intermediate_waypoints > 1):
            #     intermediate_waypoint_time_constraints = create_intermediate_waypoint_time_scale_constraint(num_cont_pts, num_intermediate_waypoints, self._dimension)
            #     constraints.append(intermediate_waypoint_time_constraints)
            if waypoint_data.intermediate_velocities is not None:
                intermediate_waypoint_velocity_constraints, intermediate_waypoint_velocity_constraint_function_data  = \
                create_intermediate_waypoint_velocity_constraints(waypoint_data.intermediate_velocities, \
                    num_cont_pts, num_intermediate_waypoints, self._order)
                constraints.append(intermediate_waypoint_velocity_constraints)
                constraint_functions_data.append(intermediate_waypoint_velocity_constraint_function_data)
        if derivative_bounds is not None and derivative_bounds.checkIfDerivativesActive():
            derivatives_constraint, derivatives_constraint_function = create_derivatives_constraint( \
                derivative_bounds, num_cont_pts, self._dimension, self._order)
            constraints.append(derivatives_constraint)
            constraint_functions_data.append(derivatives_constraint_function)
        if turning_bound is not None and turning_bound.checkIfTurningBoundActive():
            turning_constraint, turning_constraint_function_data = self._turning_const_obj.create_turning_constraint(\
                turning_bound, num_cont_pts, self._dimension)
            constraints.append(turning_constraint)
            constraint_functions_data.append(turning_constraint_function_data)
        if sfc_data is not None:
            sfc_constraint, sfc_constraint_function_data = create_safe_flight_corridor_constraint(sfc_data, \
                num_cont_pts, num_intermediate_waypoints, num_waypoint_scalars, self._dimension, self._order)
            constraints.append(sfc_constraint)
            constraint_functions_data.append(sfc_constraint_function_data)
        if (obstacles != None):
            obstacle_constraint, obstacle_constraint_function_data = self._obstacle_cons_obj.create_obstacle_constraints(obstacles,num_cont_pts, self._dimension)
            constraints.append(obstacle_constraint)
            constraint_functions_data.append(obstacle_constraint_function_data)
        return tuple(constraints), constraint_functions_data

    def __display_violated_constraints(self, constraint_data_list: 'list[ConstraintFunctionData]',  result: OptimizeResult):
        if not result.success:
            num_constraint_functions = len(constraint_data_list)
            print("\n CONSTRAINT VIOLATIONS")
            for i in range(num_constraint_functions):
                self.__print_violation(constraint_data_list[i], result.x)
            print("")

    def __print_violation(self, constraint_function_data: ConstraintFunctionData, optimized_result: npt.NDArray[np.float64]):
        output = constraint_function_data.constraint_function(optimized_result)
        violations = constraint_function_data.get_violations(output)
        num_violations = len(violations)
        if any(violations):
            error = constraint_function_data.get_error(output)
            violation_dict = {}
            for i in range(num_violations):
                if violations[i]:
                    violation_name = constraint_function_data.key[i]
                    if violation_name in violation_dict:
                                if error[i] > violation_dict[violation_name]:
                                    violation_dict[violation_name] = error[i]
                    else:
                        violation_dict[violation_name] = error[i]
            print(constraint_function_data.constraint_class, "Constraint Violations: " , violation_dict)
