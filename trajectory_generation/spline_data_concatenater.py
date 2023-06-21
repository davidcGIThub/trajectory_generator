import numpy as np
from trajectory_generation.matrix_evaluation import matrix_bspline_evaluation_for_discrete_steps, \
    matrix_bspline_derivative_evaluation_for_discrete_steps

class SplineDataConcatenater:

    def __init__(self, dimension):
        self._dimension = dimension
    
    def concatenate_spline_data(self, dt: float, start_time: float,
                                order_list: 'list[int]', 
                                control_point_array_list: 'list[np.ndarray]', 
                                scale_factor_list: 'list[float]',
                                derivative_order:int = 0,):
        full_spline_data = np.empty((self._dimension, 0))
        full_time_data = np.empty(0)
        spline_start_time = start_time
        starting_offset = 0
        for i in range(len(control_point_array_list)):
            control_points = control_point_array_list[i]
            order = order_list[i]
            scale_factor = scale_factor_list[i]
            if derivative_order == 0:
                spline_data, time_data, time_remainder, spline_end_time = \
                    matrix_bspline_evaluation_for_discrete_steps(order, control_points, spline_start_time, starting_offset, dt, scale_factor)
            else:
                spline_data, time_data, time_remainder, spline_end_time = \
                    matrix_bspline_derivative_evaluation_for_discrete_steps(order, derivative_order, scale_factor, control_points, spline_start_time, starting_offset, dt)
            spline_start_time = spline_end_time
            starting_offset = dt - time_remainder
            full_spline_data = np.concatenate((full_spline_data, spline_data),1)
            full_time_data = np.concatenate((full_time_data, time_data))
        return full_spline_data, full_time_data
    

    # def concatenate_spline_turn_data(self, dt: float, start_time: float,
    #                             order_list: 'list[int]', 
    #                             control_point_array_list: 'list[np.ndarray]', 
    #                             scale_factor_list: 'list[float]',
    #                             derivative_order:int = 0,)



        



    