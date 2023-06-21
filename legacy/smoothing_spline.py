"""
This module generates a 3rd order B-spline path between two waypoints,
waypoint directions, curvature constraint, and adjoining 
safe flight corridors.
"""
import os
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from trajectory_generation.matrix_evaluation import matrix_bspline_evaluation_for_dataset, \
    matrix_bspline_derivative_evaluation_for_dataset, count_number_of_control_points

class SmoothingSpline:
    """
    This class generates a new spline from a previous one
    """

    def __init__(self, order, dimension):
        self._dimension = dimension
        self._order = order
        
    def generate_new_control_points(self, old_control_points, old_scale_factor, old_order, max_velocity = None):
        num_cont_pts = self.__get_num_control_points(old_control_points, old_order)
        initial_control_points = self.create_initial_control_points(old_control_points, old_order, num_cont_pts)
        scale_factor = self.__get_new_scale_factor(initial_control_points, old_scale_factor, old_control_points, old_order)
        objective_function = self.__get_objective_function(num_cont_pts, old_order, old_control_points, old_scale_factor,scale_factor)
        point_constraint = self.__get_point_constraints(num_cont_pts, old_order, old_control_points,old_scale_factor, scale_factor)
        result = minimize(
            objective_function,
            x0=initial_control_points.flatten(),
            constraints=(point_constraint),
            method='SLSQP')
        optimized_control_points = np.reshape(result.x[0:num_cont_pts*self._dimension] ,(self._dimension,num_cont_pts))
        return optimized_control_points, scale_factor
    
    # def generate_control_points(self, location_data, time_data, old_num_intervals, end_time):
    #     num_cont_pts = self.__get_num_control_points_from_data(old_num_intervals)
    #     initial_control_points = self.create_initial_control_points_from_data()
    #     scale_factor = self.__get_scale_factor_from_data(old_num_intervals, end_time)
    #     objective_function = self.__get_objective_function_from_data(location_data, time_data, num_cont_pts)
    #     point_constraint = self.__get_point_constraints(num_cont_pts, old_order, old_control_points,old_scale_factor, scale_factor)
    #     result = minimize(
    #         objective_function,
    #         x0=initial_control_points.flatten(),
    #         constraints=(point_constraint),
    #         method='SLSQP')
    #     optimized_control_points = np.reshape(result.x[0:num_cont_pts*self._dimension] ,(self._dimension,num_cont_pts))
    #     return optimized_control_points, scale_factor
    
    def __get_objective_function(self, num_cont_pts, old_order, old_control_points, old_scale_factor, scale_factor):
        old_points = matrix_bspline_evaluation_for_dataset(old_order, old_control_points, self._resolution)
        def smoother(variables):
            control_points = self.__get_objective_variables(variables, num_cont_pts)
            points = matrix_bspline_evaluation_for_dataset(self._order, control_points, self._resolution)
            points_difference = (old_points - points)**2
            return np.sum(points_difference)
        return smoother

    # def __get_objective_function_from_data(self, location_data, time_data, num_cont_pts):
    #     def smoother(variables):
    #         control_points = self.__get_objective_variables(variables, num_cont_pts)
    #         points = matrix_bspline_evaluation_given_time_data(self._order, control_points, time_data)
    #         points_difference = (location_data - points)**2
    #         return np.sum(points_difference)
    #     return smoother
    
    def __get_point_constraints(self, num_cont_pts, old_order, old_control_points,old_scale_factor, scale_factor):
        # constraining the initial and final location, velocity, and acceleration
        num_pts = 2
        old_points = matrix_bspline_evaluation_for_dataset(old_order, old_control_points, num_pts)
        old_velocity_points = matrix_bspline_derivative_evaluation_for_dataset(old_order, 1, old_scale_factor, old_control_points, num_pts)
        old_acceleration_points = matrix_bspline_derivative_evaluation_for_dataset(old_order, 2, old_scale_factor, old_control_points, num_pts)
        def point_constraint_function(variables):
            control_points = self.__get_objective_variables(variables, num_cont_pts)
            points = matrix_bspline_evaluation_for_dataset(self._order, control_points, num_pts)
            points_difference = (old_points - points)
            velocity_points = matrix_bspline_derivative_evaluation_for_dataset(self._order, 1, scale_factor, control_points, num_pts)
            velocity_difference = (old_velocity_points - velocity_points)
            acceleration_points = matrix_bspline_derivative_evaluation_for_dataset(self._order, 2, scale_factor, control_points, num_pts)
            acceleration_difference = (old_acceleration_points - acceleration_points)
            return np.concatenate((points_difference.flatten(), velocity_difference.flatten(), acceleration_difference.flatten()))
        lower_bound = 0
        upper_bound = 0
        point_constraint = NonlinearConstraint(point_constraint_function, lower_bound, upper_bound)
        return point_constraint

    def __get_objective_variables(self, variables, num_cont_pts):
        control_points = np.reshape(variables[0:num_cont_pts*self._dimension], \
                    (self._dimension,num_cont_pts))
        return control_points

    def __get_num_control_points(self, old_control_points, old_order):
        old_num_intervals = count_number_of_control_points(old_control_points) - old_order
        new_num_intervals = int(old_num_intervals*2.5)
        new_num_control_points = new_num_intervals + self._order
        return new_num_control_points
    
    def __get_new_scale_factor(self, control_points, old_scale_factor, old_control_points, old_order):
        old_num_intervals = count_number_of_control_points(old_control_points) - old_order
        end_time = old_num_intervals*old_scale_factor
        num_intervals = count_number_of_control_points(control_points) - self._order
        scale_factor = end_time/ num_intervals
        return scale_factor

    def create_initial_control_points(self, old_pts, old_order, num_cont_pts):
        old_num_pts = count_number_of_control_points(old_pts)
        distances = np.linalg.norm(old_pts[:,1:] - old_pts[:,0:-1],2,0)
        old_num_segments = old_num_pts - 1
        num_segments = num_cont_pts - 1
        for i in range(old_num_segments-1):
            distances[i+1] = distances[i+1] + distances[i]
        distance_between_cont_pts = distances[old_num_segments-1] / (num_segments)
        old_segment_num = 0
        current_distance = 0
        prev_point_location = old_pts[:,0]
        step_distance = 0
        new_cont_pts = np.zeros((self._dimension, num_cont_pts))
        for i in range(num_segments):
            old_interval_start_point = old_pts[:,old_segment_num]
            old_interval_end_point = old_pts[:,old_segment_num+1]
            vector_to_point = old_interval_end_point - old_interval_start_point
            unit_vector_to_point = vector_to_point / (np.linalg.norm(vector_to_point))
            new_cont_pts[:,i] = prev_point_location + unit_vector_to_point*step_distance
            prev_point_location = new_cont_pts[:,i]
            step_distance = distance_between_cont_pts
            current_distance = current_distance + step_distance
            if distances[old_segment_num] < current_distance:
                temp = np.copy(distances)
                temp = temp - current_distance
                temp[temp < 0] = np.inf
                old_segment_num = np.argmin(temp)
                step_distance = current_distance - distances[old_segment_num-1]
                prev_point_location = old_pts[:,old_segment_num]
        new_cont_pts[:,-1] = old_pts[:,-1]
        return new_cont_pts