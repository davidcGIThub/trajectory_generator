import numpy as np
from trajectory_generation.matrix_evaluation import matrix_bspline_evaluation_for_dataset, \
    evaluate_point_on_interval, evaluate_point_derivative_on_interval, \
    count_number_of_control_points, matrix_bspline_evaluation_for_timedataset
from scipy.optimize import minimize, NonlinearConstraint

class SplineConcatenater:

    def __init__(self, order:int , dimension:int , resolution:int):
        self._resolution = resolution # num points per interval
        self._dimension = dimension
        self._order = order

    def concatenate_splines(self, order_list: 'list[int]', 
                            control_point_array_list: 'list[np.ndarray]', 
                            scale_factor_list: 'list[float]'):
        location_data, time_data = self.concatenate_spline_data(order_list, control_point_array_list, scale_factor_list)
        num_cont_pts = self.__get_num_control_points(order_list, control_point_array_list)
        scale_factor = self.__get_scale_factor(time_data[-1], num_cont_pts)
        objective_function = self.__create_objective_function(location_data, time_data, num_cont_pts, scale_factor)
        constraints = self.__create_endpoint_constraints(order_list, control_point_array_list, scale_factor_list, num_cont_pts, scale_factor)
        initial_control_points = self.__create_initial_control_points(location_data, num_cont_pts)
        result = minimize(
            objective_function,
            x0=initial_control_points.flatten(),
            constraints=(constraints),
            method='SLSQP')
        optimized_control_points = np.reshape(result.x[0:num_cont_pts*self._dimension] ,(self._dimension,num_cont_pts))
        
        return optimized_control_points, scale_factor
    
    def concatenate_spline_data(self, 
                                order_list: 'list[int]', 
                                control_point_array_list: 'list[np.ndarray]', 
                                scale_factor_list: 'list[float]'):
        previous_end_time = 0
        full_location_data = np.empty((self._dimension, 0))
        full_time_data = np.empty(0)
        for i in range(len(control_point_array_list)):
            control_points = control_point_array_list[i]
            order = order_list[i]
            scale_factor = scale_factor_list[i]
            number_of_control_points = count_number_of_control_points(control_points)
            num_intervals = number_of_control_points - order
            num_points = num_intervals*self._resolution
            spline_data = matrix_bspline_evaluation_for_dataset(order, control_points, num_points)
            spline_duration = num_intervals*scale_factor
            time_data = np.linspace(0,spline_duration,num_points) + previous_end_time
            previous_end_time = previous_end_time + spline_duration
            full_location_data = np.concatenate((full_location_data, spline_data),1)
            full_time_data = np.concatenate((full_time_data, time_data))
        return full_location_data, full_time_data
    
    def __create_objective_function(self, location_data, time_data, num_cont_pts, scale_factor):
        def smoother(variables):
            control_points = self.__get_objective_variables(variables, num_cont_pts)
            points = matrix_bspline_evaluation_for_timedataset(self._order, control_points, time_data, scale_factor)
            objective = np.sum((points - location_data)**2)
            return objective
        return smoother

    def __create_endpoint_constraints(self, order_list: 'list[int]', 
                                      control_point_array_list: 'list[np.ndarray]', 
                                      scale_factor_list: 'list[float]',
                                      num_cont_pts, scale_factor):
        start_order = order_list[0]
        start_control_points = control_point_array_list[0][:,0:start_order+1]
        start_scale_factor = scale_factor_list[0]
        start_location = evaluate_point_on_interval(start_control_points, 0, 0, start_scale_factor)
        start_velocity = evaluate_point_derivative_on_interval(start_control_points, 0,0, start_scale_factor, 1)
        start_acceleration = evaluate_point_derivative_on_interval(start_control_points, 0,0, start_scale_factor, 2)
        end_order = order_list[-1]
        end_control_points = control_point_array_list[-1][:,-(end_order+1):]
        end_scale_factor = scale_factor_list[-1]
        end_location = evaluate_point_on_interval(end_control_points, end_scale_factor, 0, end_scale_factor)
        end_velocity = evaluate_point_derivative_on_interval(end_control_points, end_scale_factor, 0, end_scale_factor, 1)
        end_acceleration = evaluate_point_derivative_on_interval(end_control_points, end_scale_factor, 0, end_scale_factor, 1)
        endpoint_constraint_values = np.concatenate((start_location, start_velocity, start_acceleration,
                                          end_location, end_velocity, end_acceleration),1)
        def terminal_point_constraint_function(variables):
            control_points = self.__get_objective_variables(variables, num_cont_pts)
            initial_control_points = control_points[:,0:self._order+1]
            initial_location = evaluate_point_on_interval(initial_control_points, 0, 0, scale_factor)
            initial_velocity = evaluate_point_derivative_on_interval(initial_control_points, 0,0, scale_factor, 1)
            initial_acceleration = evaluate_point_derivative_on_interval(initial_control_points, 0,0, scale_factor, 2)
            final_control_points = control_points[:,-(self._order+1):]
            final_location = evaluate_point_on_interval(final_control_points, scale_factor, 0, scale_factor)
            final_velocity = evaluate_point_derivative_on_interval(final_control_points, scale_factor, 0, scale_factor, 1)
            final_acceleration = evaluate_point_derivative_on_interval(final_control_points, scale_factor, 0, scale_factor, 2)
            endpoint_values = np.concatenate((initial_location, initial_velocity, initial_acceleration,
                                              final_location  , final_velocity  , final_acceleration),1)
            constraints = (endpoint_constraint_values - endpoint_values).flatten()
            return constraints
        lower_bound = 0
        upper_bound = 0
        terminal_point_constraint = NonlinearConstraint(terminal_point_constraint_function, lower_bound, upper_bound)
        return terminal_point_constraint
    
    def __create_initial_control_points(self, location_data, num_cont_points):
        control_points = np.zeros((self._dimension, num_cont_points))
        num_data_points = np.shape(location_data)[1]
        mapping_array = np.linspace(0,num_data_points-1, num_cont_points)
        for i in range(num_cont_points):
            j = int(mapping_array[i])
            control_points[:,i] = location_data[:,j]
        return control_points
    
    def __get_objective_variables(self, variables, num_cont_pts):
        control_points = np.reshape(variables[0:num_cont_pts*self._dimension], \
                    (self._dimension,num_cont_pts))
        return control_points
    
    def __get_num_control_points(self, order_list: 'list[int]', 
                            control_point_array_list: 'list[np.ndarray]'):
        total_num_intervals = 0
        for i in range(len(control_point_array_list)):
            num_cont_pts = count_number_of_control_points(control_point_array_list[i])
            order = order_list[i]
            num_intervals = num_cont_pts - order
            total_num_intervals += num_intervals
        number_of_new_control_points = total_num_intervals*3 + self._order
        return number_of_new_control_points
    
    def __get_scale_factor(self, spline_duration, num_cont_pts):
        num_intervals = num_cont_pts - self._order
        scale_factor = spline_duration/num_intervals
        return scale_factor





        



    