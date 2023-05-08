"""
This module generates a 3rd order B-spline path between two waypoints,
waypoint directions, curvature constraint, and adjoining 
safe flight corridors.
"""
import os
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, Bounds
from trajectory_generation.matrix_evaluation import get_M_matrix, evaluate_point_on_interval
from TrajectoryConstraintsCCode.python_wrappers.obstacle_constraints import ObstacleConstraints
from TrajectoryConstraintsCCode.python_wrappers.turning_constraints import TurningConstraints
from trajectory_generation.bspline_to_minvo import get_composite_bspline_to_minvo_conversion_matrix
from trajectory_generation.bspline_to_bezier import get_composite_bspline_to_bezier_conversion_matrix
from trajectory_generation.safe_flight_corridor import SFC_Data, SFC
from trajectory_generation.obstacle import Obstacle
from trajectory_generation.waypoint_data import Waypoint, WaypointData
from trajectory_generation.dynamic_bounds import DerivativeBounds, TurningBound
import time

class TrajectoryGenerator:
    """
    This class generates a 3rd order B-spline path between two waypoints,
    waypoint directions, curvature constraint, and adjoining 
    safe flight corridors.
    """

### TODO ####
# 1. remove the scale factor constraint
# 2. add checks to make sure constraints are feasible with eachother
# 3. Change obstacle constraints to check only intervals that have an obstacle in the same SFC
# 4. add constraints to reach intermediate waypoints between start and end waypoint.

    def __init__(self, dimension: int, 
                 num_intervals_free_space: int = 5):
        self._dimension = dimension
        self._order = 3
        self._M = get_M_matrix(self._order)
        self._turning_const_obj = TurningConstraints(self._dimension)
        self._obstacle_cons_obj = ObstacleConstraints(self._dimension)
        self._num_intervals_free_space = num_intervals_free_space
        
    def generate_trajectory(self, waypoint_data: WaypointData, derivative_bounds: DerivativeBounds = None, turning_bound: TurningBound = None,
                sfc_data: SFC_Data = None, obstacles: list = None, objective_function_type: str = "minimal_velocity_path"):
        num_intervals = self.__get_num_intervals(sfc_data)
        num_intermediate_waypoints = waypoint_data.get_num_intermediate_waypoints()
        point_sequence = self.__get_point_sequence(waypoint_data, sfc_data)
        num_cont_pts = self.__get_num_control_points(num_intervals)
        constraints = self.__get_constraints(num_cont_pts, waypoint_data, derivative_bounds, turning_bound, sfc_data, obstacles, num_intermediate_waypoints)
        objectiveFunction = self.__get_objective_function(objective_function_type)
        objective_variable_bounds = self.__create_objective_variable_bounds(num_cont_pts, num_intermediate_waypoints)
        waypoint_sequence = waypoint_data.get_waypoint_locations()
        optimization_variables = self.__create_initial_objective_variables(num_cont_pts, point_sequence, num_intermediate_waypoints, waypoint_sequence)
        minimize_options = {'disp': False} #, 'maxiter': self.maxiter, 'ftol': tol}
        # perform optimization
        result = minimize(
            objectiveFunction,
            x0=optimization_variables,
            args=(num_cont_pts,),
            method='SLSQP', 
            bounds=objective_variable_bounds,
            constraints=constraints, 
            options = minimize_options)
        optimized_control_points = np.reshape(result.x[0:num_cont_pts*self._dimension] ,(self._dimension,num_cont_pts))
        scale_factor = result.x[num_cont_pts*self._dimension]
        return optimized_control_points, scale_factor
    
    def __get_objective_function(self, objective_function_type):
        if objective_function_type == "minimal_distance_path":
            return self.__minimize_velocity_control_points_objective_function
        elif objective_function_type == "minimal_velocity_path":
            return self.__minimize_acceleration_control_points_objective_function
        elif objective_function_type == "minimal_acceleration_path":
            return self.__minimize_jerk_control_points_objective_function
        else:
            raise Exception("Error, Invalid objective function type")
        
    def __create_objective_variable_bounds(self, num_cont_pts, num_intermediate_waypoints):
        lower_bounds = np.zeros(num_cont_pts*self._dimension + 1 + num_intermediate_waypoints) - np.inf
        upper_bounds = np.zeros(num_cont_pts*self._dimension + 1 + num_intermediate_waypoints) + np.inf
        lower_bounds[num_cont_pts*self._dimension] = 0.000001
        if num_intermediate_waypoints > 0:
            num_intervals = num_cont_pts - self._order
            upper_bounds[-num_intermediate_waypoints:] = num_intervals
        return Bounds(lb=lower_bounds, ub=upper_bounds)

    def __get_num_intervals(self, sfc_data: SFC_Data):
        num_intervals = self._num_intervals_free_space
        if sfc_data is not None:
            num_intervals = sfc_data.get_num_intervals()
        return num_intervals
        
    def __get_point_sequence(self, waypoint_data:WaypointData, sfc_data:SFC_Data = None):
        if sfc_data is None:
            point_sequence = waypoint_data.get_waypoint_locations()
            return point_sequence
        else:
            return sfc_data.get_point_sequence()
    
    def __create_initial_objective_variables(self, num_cont_pts, point_sequence, num_intermediate_waypoints,  waypoint_sequence):
        control_points = self.__create_initial_control_points(num_cont_pts, point_sequence)
        scale_factor = 1
        variables = np.concatenate((control_points.flatten(),[scale_factor]))
        if (num_intermediate_waypoints > 0):
            intermediate_waypoint_time_scales = self.__create_intermediate_waypoint_time_scales(waypoint_sequence, num_cont_pts)
            variables = np.concatenate((variables, intermediate_waypoint_time_scales))
        return variables
        
    def __get_objective_variables(self, variables, num_cont_pts):
        control_points = np.reshape(variables[0:num_cont_pts*self._dimension], \
                    (self._dimension,num_cont_pts))
        scale_factor = variables[num_cont_pts*self._dimension]
        return control_points, scale_factor
    
    def __get_intermediate_waypoint_scale_times(self, variables, num_middle_waypoints):
        intermediate_waypoint_times = variables[-num_middle_waypoints:]
        return intermediate_waypoint_times

    def __get_constraints(self, num_cont_pts: int, waypoint_data: WaypointData, 
            derivative_bounds: DerivativeBounds, turning_bound: TurningBound, 
            sfc_data: SFC_Data, obstacles: list, num_intermediate_waypoints):
        waypoints = np.concatenate((waypoint_data.start_waypoint.location, waypoint_data.end_waypoint.location),1)
        waypoint_constraint = self.__create_waypoint_constraint(waypoints, num_cont_pts, num_intermediate_waypoints)
        constraints = [waypoint_constraint]
        if waypoint_data.start_waypoint.checkIfDerivativesActive():
            print("start derivative active")
            start_waypoint_derivatives_constraint = self.__create_waypoint_derivative_constraints(waypoint_data.start_waypoint, num_cont_pts)
            constraints.append(start_waypoint_derivatives_constraint)
        if waypoint_data.end_waypoint.checkIfDerivativesActive():
            end_waypoint_derivatives_constraint = self.__create_waypoint_derivative_constraints(waypoint_data.end_waypoint, num_cont_pts)
            constraints.append(end_waypoint_derivatives_constraint)
        if waypoint_data.intermediate_locations is not None:
            intermediate_waypoint_constraints = self.__create_intermediate_waypoint_constraints(waypoint_data.intermediate_locations, num_cont_pts, num_intermediate_waypoints)
            constraints.append(intermediate_waypoint_constraints)
            if(num_intermediate_waypoints > 1):
                intermediate_waypoint_time_constraints = self.__create_intermediate_waypoint_time_scales_constraint(num_cont_pts, num_intermediate_waypoints)
                constraints.append(intermediate_waypoint_time_constraints)
        if derivative_bounds is not None and derivative_bounds.checkIfDerivativesActive() is not None:
            derivatives_constraint = self.__create_derivatives_constraint(derivative_bounds, num_cont_pts)
            constraints.append(derivatives_constraint)
        if turning_bound is not None and turning_bound.checkIfTurningBoundActive():
            turning_constraint = self.__create_turning_constraint(turning_bound, num_cont_pts)
            constraints.append(turning_constraint)
        if sfc_data is not None:
            sfc_constraint = self.__create_safe_flight_corridor_constraint(sfc_data, num_cont_pts, num_intermediate_waypoints)
            constraints.append(sfc_constraint)
        if (obstacles != None):
            obstacle_constraint = self.__create_obstacle_constraints(obstacles, num_cont_pts)
            constraints.append(obstacle_constraint)
        return tuple(constraints)

    def __minimize_jerk_control_points_objective_function(self, variables, num_cont_pts):
        # for third order splines only
        control_points, scale_factor = self.__get_objective_variables(variables, num_cont_pts)
        jerk_cps = control_points[:,3:] - 3*control_points[:,2:-1] + 3*control_points[:,1:-2] - control_points[:,0:-3]
        square_jerk_control_points = np.sum(jerk_cps**2,0)
        objective = np.sum(square_jerk_control_points) + scale_factor
        return objective
    
    def __minimize_velocity_control_points_objective_function(self, variables, num_cont_pts):
        # for third order splines only
        control_points, scale_factor = self.__get_objective_variables(variables, num_cont_pts)
        velocity_cps =  control_points[:,0:-1] - control_points[:,1:]
        velocity_control_points_squared_sum = np.sum(velocity_cps**2,0)
        objective = np.sum(velocity_control_points_squared_sum) + scale_factor
        return objective
    
    def __minimize_acceleration_control_points_objective_function(self, variables, num_cont_pts):
        # for third order splines only
        control_points, scale_factor = self.__get_objective_variables(variables, num_cont_pts)
        acceleration_cps =  control_points[:,2:] - 2*control_points[:,1:-1] + control_points[:,0:-2]
        accel_control_points_squared_sum = np.sum(acceleration_cps**2,0)
        objective = np.sum(accel_control_points_squared_sum) + scale_factor
        return objective

    def __create_waypoint_constraint(self, waypoints, num_cont_pts, num_intermediate_waypoints):
        num_end_waypoints = 2
        num_extra_spaces = 1 + num_intermediate_waypoints
        m = num_end_waypoints
        n = num_cont_pts
        k = self._order
        d = self._dimension
        constraint_matrix = np.zeros((m*d,n*d))
        Gamma_0 = np.zeros((self._order+1,1))
        Gamma_0[self._order,0] = 1
        Gamma_f = np.ones((self._order+1,1))
        M_Gamma_0_T = np.dot(self._M,Gamma_0).T
        M_Gamma_f_T = np.dot(self._M,Gamma_f).T
        for i in range(self._dimension):
            constraint_matrix[i*m   ,  i*n        : i*n+k+1] = M_Gamma_0_T
            constraint_matrix[i*m+1 , (i+1)*n-k-1 : (i+1)*n] = M_Gamma_f_T
        constraint_matrix = np.concatenate((constraint_matrix,np.zeros((m*d,num_extra_spaces))),1)
        constraint = LinearConstraint(constraint_matrix, lb=waypoints.flatten(), ub=waypoints.flatten())
        return constraint
    
    ### convert to c++ code
    def __create_intermediate_waypoint_constraints(self, intermediate_locations, num_cont_pts, num_intermediate_waypoints):
        lower_bound = 0
        upper_bound = 0
        def intermediate_waypoint_constraint_function(variables):
            control_points, scale_factor = self.__get_objective_variables(variables, num_cont_pts)
            intermediate_waypoint_scale_times = self.__get_intermediate_waypoint_scale_times(variables, num_intermediate_waypoints)
            constraints = np.zeros((self._dimension, num_intermediate_waypoints))
            for i in range(num_intermediate_waypoints):
                desired_location = intermediate_locations[:,i]
                scale_time = intermediate_waypoint_scale_times[i]
                interval = int(time)
                interval_cont_pts = control_points[:,interval:interval+self._order+1]
                location = evaluate_point_on_interval(interval_cont_pts, scale_time-interval, 0, 1)
                constraints[:,i] = location.flatten() - desired_location
            return constraints.flatten()
        intermediate_waypoint_constraint = NonlinearConstraint(intermediate_waypoint_constraint_function, lb= lower_bound, ub=upper_bound)
        return intermediate_waypoint_constraint
    
    def __create_intermediate_waypoint_time_scales_constraint(self, num_cont_pts, num_intermediate_waypoints):
        num_extra_spaces = 2 + num_intermediate_waypoints
        m = num_intermediate_waypoints
        n = num_cont_pts
        d = self._dimension
        constraint_matrix = np.zeros((m-1,n*d+num_extra_spaces))
        for i in range(m-1):
            constraint_matrix[i,-i-1] = -1
            constraint_matrix[i,-i-2] = 1
        constraint = LinearConstraint(constraint_matrix, lb=-np.inf, ub=0)
        return constraint

    def __create_start_waypoint_derivative_constraints(self, start_waypoint: Waypoint, num_cont_pts):
        lower_bound = 0
        upper_bound = 0
        if start_waypoint.checkIfVelocityActive():
            start_velocity_desired = start_waypoint.velocity.flatten()
        if start_waypoint.checkIfAccelerationActive():
            start_acceleration_desired = start_waypoint.acceleration.flatten()
        startVelocityIsActive = start_waypoint.checkIfVelocityActive()
        startAccelerationIsActive = start_waypoint.checkIfAccelerationActive()
        def start_waypoint_derivative_constraint_function(variables):
            control_points, scale_factor = self.__get_objective_variables(variables, num_cont_pts)
            constraints = np.array([])
            if startVelocityIsActive:
                start_velocity_direction = (control_points[:,2] - control_points[:,0])/(2*scale_factor)
                constraints = start_velocity_direction - start_velocity_desired
            if startAccelerationIsActive:
                start_acceleration_direction = (control_points[:,0] - 2*control_points[:,1] + control_points[:,2])/(scale_factor*scale_factor)
                constraints_2 = start_acceleration_direction - start_acceleration_desired
                constraints = np.concatenate((constraints, constraints_2))
            return constraints.flatten()
        start_waypoint_derivative_constraint = NonlinearConstraint(start_waypoint_derivative_constraint_function, lb= lower_bound, ub=upper_bound)
        return start_waypoint_derivative_constraint
    
    ### make start and end constraint into a function can use for both
    def __create_end_waypoint_derivative_constraints(self, end_waypoint: Waypoint, num_cont_pts):
        lower_bound = 0
        upper_bound = 0
        if end_waypoint.checkIfVelocityActive():
            end_velocity_desired = end_waypoint.velocity.flatten()
        if end_waypoint.checkIfAccelerationActive():
            end_acceleration_desired = end_waypoint.acceleration.flatten()
        endVelocityIsActive = end_waypoint.checkIfVelocityActive()
        endAccelerationIsActive = end_waypoint.checkIfAccelerationActive()
        def end_waypoint_derivative_constraint_function(variables):
            control_points, scale_factor = self.__get_objective_variables(variables, num_cont_pts)
            constraints = np.array([])
            if endVelocityIsActive:
                end_velocity_direction = (control_points[:,-1] - control_points[:,-3])/(2*scale_factor)
                constraints = end_velocity_direction - end_velocity_desired
            if endAccelerationIsActive:
                end_acceleration_direction = (control_points[:,-3] - 2*control_points[:,-2] + control_points[:,-1])/(scale_factor*scale_factor)
                constraints_2 = end_acceleration_direction - end_acceleration_desired
                constraints = np.concatenate((constraints, constraints_2))
            return constraints.flatten()
        end_waypoint_derivative_constraint = NonlinearConstraint(end_waypoint_derivative_constraint_function, lb= lower_bound, ub=upper_bound)
        return end_waypoint_derivative_constraint
    
    def __create_waypoint_derivative_constraints(self, waypoint: Waypoint, num_cont_pts: int):
        print("waypoint: " , waypoint.side)
        lower_bound = 0
        upper_bound = 0
        if waypoint.checkIfVelocityActive():
            velocity_desired = waypoint.velocity.flatten()
        if waypoint.checkIfAccelerationActive():
            acceleration_desired = waypoint.acceleration.flatten()
        velocityIsActive = waypoint.checkIfVelocityActive()
        accelerationIsActive = waypoint.checkIfAccelerationActive()
        constraints = self.__initialize_derivative_constraints(waypoint)
        side = waypoint.side
        def waypoint_derivative_constraint_function(variables):
            control_points, scale_factor = self.__get_objective_variables(variables, num_cont_pts)
            marker = 0
            if velocityIsActive:
                velocity = self.__get_terminal_velocity(side, control_points, scale_factor)
                constraints[marker:self._dimension] = (velocity - velocity_desired).flatten()
                marker += self._dimension
            if accelerationIsActive:
                acceleration = self.__get_terminal_acceleration(side, control_points, scale_factor)
                constraints[marker:marker+self._dimension] = (acceleration - acceleration_desired).flatten()
            return constraints
        waypoint_derivative_constraint = NonlinearConstraint(waypoint_derivative_constraint_function, lb= lower_bound, ub=upper_bound)
        return waypoint_derivative_constraint
    
    def __initialize_derivative_constraints(self, waypoint: Waypoint):
        length = 0
        if waypoint.checkIfVelocityActive():
            length += self._dimension
        if waypoint.checkIfAccelerationActive():
            length += self._dimension
        return np.zeros(length)
    
    def __get_terminal_velocity(self, side, control_points, scale_factor):
        if side == "start":
            velocity = (control_points[:,2] - control_points[:,0])/(2*scale_factor)
        if side == "end":
            velocity = (control_points[:,-1] - control_points[:,-3])/(2*scale_factor)
        return velocity

    def __get_terminal_acceleration(self, side, control_points, scale_factor):
        if side == "start":
            acceleration = (control_points[:,0] - 2*control_points[:,1] + control_points[:,2])/(scale_factor*scale_factor)
        if side == "end":
            acceleration = (control_points[:,-3] - 2*control_points[:,-2] + control_points[:,-1])/(scale_factor*scale_factor)
        return acceleration

    def __create_derivatives_constraint(self, derivative_bounds: DerivativeBounds, num_cont_pts):
        num_vel_cont_pts = num_cont_pts - 1
        M_v = get_composite_bspline_to_bezier_conversion_matrix(num_vel_cont_pts, self._order-1)
        constraints = self.__initialize_derivative_constraint_array(derivative_bounds)
        def derivatives_constraint_function(variables):
            control_points, scale_factor = self.__get_objective_variables(variables, num_cont_pts)
            velocity_control_points = (control_points[:,1:] - control_points[:,0:-1])/scale_factor
            count = 0
            if derivative_bounds.max_velocity is not None:
                bezier_velocity_control_points = np.transpose(np.dot(M_v, np.transpose(velocity_control_points)))
                constraints[count] = self.__calculate_velocity_constraint(bezier_velocity_control_points, derivative_bounds)
                count += 1
                if derivative_bounds.max_upward_velocity is not None and self._dimension == 3:
                    constraints[count] = self.__calculate_upward_velocity_constraint(bezier_velocity_control_points, derivative_bounds)
                    count += 1
            if derivative_bounds.max_acceleration is not None:
                acceleration_constraint = self.__calculate_acceleration_constraint(derivative_bounds, velocity_control_points, scale_factor)
                constraints[count] = acceleration_constraint
                count += 1
            return constraints
        lower_bound = - np.inf
        upper_bound = 0
        derivatives_constraint = NonlinearConstraint(derivatives_constraint_function , lb = lower_bound, ub = upper_bound)
        return derivatives_constraint
    
    def __initialize_derivative_constraint_array(self, derivative_bounds):
        length = 0
        if derivative_bounds.max_velocity is not None:
            length += 1
            if derivative_bounds.max_upward_velocity is not None:
                length += 1
        if derivative_bounds.max_acceleration is not None:
            length += 1
        return np.zeros(length)
    
    def __calculate_velocity_constraint(self, bezier_velocity_control_points, derivative_bounds):
        velocity_bound = np.max(np.linalg.norm(bezier_velocity_control_points,2,0))
        constraint = velocity_bound - derivative_bounds.max_velocity
        return constraint
    
    def __calculate_upward_velocity_constraint(self, bezier_velocity_control_points, derivative_bounds):
        min_z_vel = np.max(bezier_velocity_control_points[2,:])
        constraint = min_z_vel - derivative_bounds.max_upward_velocity
        return constraint
    
    def __calculate_acceleration_constraint(self, derivative_bounds, velocity_control_points, scale_factor):
        acceleration_control_points = (velocity_control_points[:,1:] - velocity_control_points[:,0:-1])/scale_factor
        if derivative_bounds.gravity is not None and self._dimension == 3:
            gravity = np.array([[0],[0],[derivative_bounds.gravity]])
            acceleration_control_points = acceleration_control_points - gravity
        acceleration_bound = np.max(np.linalg.norm(acceleration_control_points,2,0))
        constraint = acceleration_bound - derivative_bounds.max_acceleration
        return constraint

    def __create_turning_constraint(self, turning_bound: TurningBound, num_cont_pts):
        def centripetal_acceleration_constraint_function(variables):
            control_points, scale_factor = self.__get_objective_variables(variables, num_cont_pts)
            const = self._turning_const_obj.get_spline_centripetal_acceleration_constraint(control_points, turning_bound.max_turning_bound, scale_factor)
            return const
        def angular_rate_constraint_function(variables):
            control_points, scale_factor = self.__get_objective_variables(variables, num_cont_pts)
            return self._turning_const_obj.get_spline_angular_rate_constraint(control_points, turning_bound.max_turning_bound, scale_factor)
        def curvature_constraint_function(variables):
            control_points, scale_factor = self.__get_objective_variables(variables, num_cont_pts)
            return self._turning_const_obj.get_spline_curvature_constraint(control_points,turning_bound.max_turning_bound)
        if turning_bound.bound_type == "curvature":
            constraint_function = curvature_constraint_function
        elif turning_bound.bound_type == "centripetal_acceleration":
            constraint_function = centripetal_acceleration_constraint_function
        elif turning_bound.bound_type == "angular_rate":
            constraint_function = angular_rate_constraint_function
        else:
            raise Exception("Not valid turning bound type")
        lower_bound = - np.inf
        upper_bound = 0
        turning_constraint = NonlinearConstraint(constraint_function , lb = lower_bound, ub = upper_bound)
        return turning_constraint
    

    def __create_safe_flight_corridor_constraint(self, sfc_data: SFC_Data, num_cont_pts, num_intermediate_waypoints):
        # create the rotation matrix.
        num_extra_spaces = 1 + num_intermediate_waypoints
        num_corridors = self.__get_num_corridors(sfc_data)
        num_minvo_cont_pts = (num_cont_pts - self._order)*(self._order+1)
        intervals_per_corridor = sfc_data.get_intervals_per_corridor()
        sfc_list = sfc_data.get_sfc_list()
        M_rot = self.get_composite_sfc_rotation_matrix(intervals_per_corridor, sfc_list, num_minvo_cont_pts)
        # create the bspline to minvo conversion matrix 
        M_minvo = get_composite_bspline_to_minvo_conversion_matrix(\
            num_cont_pts, self._order)
        zero_block = np.zeros((num_minvo_cont_pts,num_cont_pts))
        zero_col = np.zeros((num_minvo_cont_pts, num_extra_spaces))
        if self._dimension == 2:
            M_minvo = np.block([[M_minvo, zero_block, zero_col],
                                        [zero_block, M_minvo, zero_col]])
        if self._dimension == 3:
            M_minvo = np.block([[M_minvo,    zero_block, zero_block, zero_col],
                                [zero_block, M_minvo   , zero_block, zero_col],
                                [zero_block, zero_block, M_minvo   , zero_col]])
        conversion_matrix = M_rot @ M_minvo
        #create bounds
        lower_bounds = np.zeros((self._dimension, num_minvo_cont_pts))
        upper_bounds = np.zeros((self._dimension, num_minvo_cont_pts))
        index = 0
        for corridor_index in range(num_corridors):
            num_intervals = intervals_per_corridor[corridor_index]
            lower_bound, upper_bound = sfc_list[corridor_index].getRotatedBounds()
            num_points = num_intervals*(self._order+1)
            lower_bounds[:,index:index+num_points] = lower_bound
            upper_bounds[:,index:index+num_points] = upper_bound
            index = index+num_points
        safe_corridor_constraints = LinearConstraint(conversion_matrix, lb=lower_bounds.flatten(), ub=upper_bounds.flatten())
        return safe_corridor_constraints
    
    def __create_obstacle_constraints(self, obstacles, num_cont_pts):
        def obstacle_constraint_function(variables):
            control_points, scale_factor = self.__get_objective_variables(variables, num_cont_pts)
            # return self._obstacle_cons_obj.getObstacleConstraintsForIntervals(control_points, obstacle.radius, obstacle.center)
            radii = np.zeros(len(obstacles))
            centers = np.zeros((self._dimension,len(obstacles)))
            for i in range(len(obstacles)):
                radii[i] = obstacles[i].radius
                centers[0,i] = obstacles[i].center[0,0]
                centers[1,i] = obstacles[i].center[1,0]
                if self._dimension == 3:
                    centers[2,i] = obstacles[i].center[2,0]
            return self._obstacle_cons_obj.getObstaclesConstraintsForSpline(control_points, radii, centers)
        lower_bound = 0
        upper_bound = np.inf
        obstacle_constraint = NonlinearConstraint(obstacle_constraint_function , lb = lower_bound, ub = upper_bound)
        return obstacle_constraint
    
    def get_composite_sfc_rotation_matrix(self, intervals_per_corridor, sfcs, num_minvo_cont_pts):
        num_corridors = len(intervals_per_corridor)
        M_len = num_minvo_cont_pts*self._dimension
        M_rot = np.zeros((M_len, M_len))
        num_cont_pts_per_interval = self._order + 1
        interval_count = 0
        dim_step = num_minvo_cont_pts
        for corridor_index in range(num_corridors):
            rotation = sfcs[corridor_index].rotation.T
            num_intervals = intervals_per_corridor[corridor_index]
            for interval_index in range(num_intervals):
                for cont_pt_index in range(num_cont_pts_per_interval):
                    index = interval_count*num_cont_pts_per_interval+cont_pt_index
                    M_rot[index, index] = rotation[0,0]
                    M_rot[index, index + dim_step] = rotation[0,1]
                    M_rot[index + dim_step, index] = rotation[1,0]
                    M_rot[index + dim_step, index + dim_step] = rotation[1,1]
                    if self._dimension == 3:
                        M_rot[2*dim_step + index, index] = rotation[2,0]
                        M_rot[2*dim_step + index, index + dim_step] = rotation[2,1]
                        M_rot[2*dim_step + index, index + 2*dim_step] = rotation[2,2]
                        M_rot[dim_step + index, index + 2*dim_step] = rotation[1,2]
                        M_rot[index, index + 2*dim_step] = rotation[0,2]
                interval_count += 1
        return M_rot

    def __create_initial_control_points(self, total_num_cont_pts, point_sequence):
        num_segments = np.shape(point_sequence)[1] - 1
        if num_segments < 2:
            start_point = point_sequence[:,0]
            end_point = point_sequence[:,1]
            control_points = np.linspace(start_point,end_point,total_num_cont_pts).T
        else:
            control_points = np.empty(shape=(self._dimension,total_num_cont_pts))
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
    
    def __create_intermediate_waypoint_time_scales(self , point_sequence, num_cont_pts):
        num_intervals = num_cont_pts - self._order
        num_segments = np.shape(point_sequence)[1] - 1
        intermediate_waypoint_times = np.array([0.5])
        if num_segments > 2:
            distances = np.linalg.norm(point_sequence[:,1:] - point_sequence[:,0:-1],2,0)
            for i in range(num_segments-1):
                distances[i+1] = distances[i+1] + distances[i]
            norm_distances = distances/distances[num_segments-1]
            intermediate_waypoint_times = norm_distances[0:-1]*num_intervals
        return intermediate_waypoint_times

    def __get_num_control_points(self, num_intervals):
        num_control_points = num_intervals + self._order
        return int(num_control_points)
    
    def __get_num_corridors(self, sfc_data:SFC_Data = None):
        if sfc_data is None:
            return int(0)
        else:
            return sfc_data.get_num_corridors()
        