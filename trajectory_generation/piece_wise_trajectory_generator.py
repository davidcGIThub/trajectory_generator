import numpy as np
from trajectory_generation.trajectory_generator import TrajectoryGenerator
from trajectory_generation.constraint_data_structures.constraints_container import ConstraintsContainer
from trajectory_generation.spline_order_converter import SmoothingSpline
from bsplinegenerator.bsplines import BsplineEvaluation

class PieceWiseTrajectoryGenerator:

    def __init__(self, dimension, order=3):
        self._dimension = dimension
        self._order = order

    def generate_trajectory(self, constraints_container_list: np.ndarray, num_data_points_per_interval=100):
        traj_gen = TrajectoryGenerator(self._dimension)
        constraints_container = constraints_container_list[i]
        control_points, scale_factor = traj_gen.generate_trajectory(constraints_container)
        bspline = BsplineEvaluation(control_points, self._order, 0, scale_factor)
        location_data, time_data = bspline.get_spline_data(num_data_points_per_interval)
        end_time = bspline.get_end_time()
        for i in range(1,len(constraints_container_list)):
            traj_gen = TrajectoryGenerator(self._dimension)
            constraints_container = constraints_container_list[i]
            control_points, scale_factor = traj_gen.generate_trajectory(constraints_container)
            current_bspline = BsplineEvaluation(control_points, self._order, end_time, scale_factor)
            current_location_data, current_time_data = current_bspline.get_spline_data(num_data_points_per_interval)
            location_data = np.concatenate((location_data,current_location_data))
            time_data = np.concatenate((time_data,current_time_data))
            end_time = current_bspline.get_end_time()
        resolution = 100
        spline_smoother = SmoothingSpline(self._order, self._dimension, resolution)
        spline_smoother.generate_control_points()
        
