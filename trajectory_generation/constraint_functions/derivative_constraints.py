import numpy as np
from scipy.optimize import NonlinearConstraint
from trajectory_generation.constraint_data_structures.dynamic_bounds import DerivativeBounds
from trajectory_generation.control_point_conversions.bspline_to_bezier import get_composite_bspline_to_bezier_conversion_matrix
from trajectory_generation.objectives.objective_variables import get_control_points, get_scale_factor
from trajectory_generation.constraint_data_structures.constraint_function_data import ConstraintFunctionData
from trajectory_generation.constraint_functions.control_point_derivative_bounds import ControlPointDerivativeBounds

class DerivativeConstraints(object):

    def __init__(self, dimension):
        self.cont_pt_derivative_bounds = ControlPointDerivativeBounds(dimension)

    def create_derivatives_constraint(self, derivative_bounds: DerivativeBounds, num_cont_pts, dimension, order):
        num_vel_cont_pts = num_cont_pts - 1
        num_accel_cont_pts = num_cont_pts - 2
        M_v = get_composite_bspline_to_bezier_conversion_matrix(num_vel_cont_pts, order-1)
        M_a = get_composite_bspline_to_bezier_conversion_matrix(num_accel_cont_pts, order-2)
        constraints, constraint_key, length = self.initialize_derivative_constraint_array(derivative_bounds)
        def derivatives_constraint_function(variables):
            control_points = get_control_points(variables, num_cont_pts, dimension)
            scale_factor = get_scale_factor(variables, num_cont_pts, dimension)
            velocity_control_points = (control_points[:,1:] - control_points[:,0:-1])/scale_factor
            count = 0
            if derivative_bounds.min_velocity is not None or derivative_bounds.max_velocity is not None:
                bezier_velocity_control_points = np.transpose(np.dot(M_v, np.transpose(velocity_control_points)))
                if derivative_bounds.min_velocity is not None:
                    min_velocity_constraint = self.calculate_min_velocity_constraint(bezier_velocity_control_points, derivative_bounds)
                    constraints[count] = min_velocity_constraint
                    count += 1
                if derivative_bounds.max_velocity is not None:
                    constraints[count] = self.calculate_max_velocity_constraint(bezier_velocity_control_points, derivative_bounds)
                    count += 1
                    if derivative_bounds.max_upward_velocity is not None and dimension == 3:
                        constraints[count] = self.calculate_max_upward_velocity_constraint(bezier_velocity_control_points, derivative_bounds)
                        count += 1
                    if derivative_bounds.max_horizontal_velocity is not None and dimension == 3:
                        constraints[count] = self.calculate_max_horizontal_velocity_constraint(bezier_velocity_control_points, derivative_bounds)
                        count += 1
            if derivative_bounds.max_acceleration is not None:
                acceleration_constraint = self.calculate_max_acceleration_constraint(derivative_bounds, velocity_control_points, scale_factor, M_a)
                constraints[count] = acceleration_constraint
                count += 1
            return constraints
        lower_bound = np.zeros(length) - np.inf
        upper_bound = np.zeros(length)
        constraint_class = "Derivative"
        derivatives_constraint = NonlinearConstraint(derivatives_constraint_function , lb = lower_bound, ub = upper_bound)
        constraint_function_data = ConstraintFunctionData(derivatives_constraint_function, lower_bound, upper_bound, constraint_key, constraint_class)
        return derivatives_constraint, constraint_function_data

    def initialize_derivative_constraint_array(self, derivative_bounds: DerivativeBounds):
        length = 0
        constraint_key = np.array([])
        if derivative_bounds.min_velocity is not None:
            length += 1
            constraint_key = np.concatenate((constraint_key,["min_velocity"]))
        if derivative_bounds.max_velocity is not None:
            length += 1
            constraint_key = np.concatenate((constraint_key,["max_velocity"]))
            if derivative_bounds.max_upward_velocity is not None:
                length += 1
                constraint_key = np.concatenate((constraint_key,["max_upward_velocity"]))
            if derivative_bounds.max_horizontal_velocity is not None:
                length += 1
                constraint_key = np.concatenate((constraint_key,["max_horizontal_velocity"]))
        if derivative_bounds.max_acceleration is not None:
            length += 1
            constraint_key = np.concatenate((constraint_key,["max_acceleration"]))
        constraint_array = np.zeros(length)
        return constraint_array, constraint_key, length

    def calculate_max_horizontal_velocity_constraint(self, bezier_velocity_control_points, derivative_bounds: DerivativeBounds):
        velocity_bound = np.max(np.linalg.norm(bezier_velocity_control_points[0:2,:],2,0))
        constraint = velocity_bound - derivative_bounds.max_horizontal_velocity
        return constraint

    def calculate_max_velocity_constraint(self, bezier_velocity_control_points, derivative_bounds: DerivativeBounds):
        velocity_bound = np.max(np.linalg.norm(bezier_velocity_control_points,2,0))
        constraint = velocity_bound - derivative_bounds.max_velocity
        return constraint

    def calculate_max_upward_velocity_constraint(self, bezier_velocity_control_points, derivative_bounds: DerivativeBounds):
        min_z_vel = np.min(bezier_velocity_control_points[2,:])
        constraint = -min_z_vel - derivative_bounds.max_upward_velocity
        return constraint

    def calculate_max_acceleration_constraint(self, derivative_bounds: DerivativeBounds, velocity_control_points, scale_factor, M_a):
        acceleration_control_points = (velocity_control_points[:,1:] - velocity_control_points[:,0:-1])/scale_factor
        bezier_acceleration_control_points = np.transpose(np.dot(M_a, np.transpose(acceleration_control_points)))
        dimension = np.shape(velocity_control_points)[0]
        if derivative_bounds.gravity is not None and dimension == 3:
            gravity = np.array([[0],[0],[derivative_bounds.gravity]])
            bezier_acceleration_control_points = bezier_acceleration_control_points - gravity
        acceleration_bound = np.max(np.linalg.norm(bezier_acceleration_control_points,2,0))
        constraint = acceleration_bound - derivative_bounds.max_acceleration
        return constraint

    def calculate_min_velocity_constraint(self, bezier_velocity_control_points, derivative_bounds: DerivativeBounds):
        lower_bound_velocity = self.cont_pt_derivative_bounds.get_min_velocity_of_bez_vel_cont_pts(bezier_velocity_control_points)
        constraint = derivative_bounds.min_velocity - lower_bound_velocity
        return constraint