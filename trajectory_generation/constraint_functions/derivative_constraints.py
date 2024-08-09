import numpy as np
from scipy.optimize import NonlinearConstraint
from trajectory_generation.constraint_data_structures.dynamic_bounds import DerivativeBounds
from trajectory_generation.control_point_conversions.bspline_to_bezier import get_composite_bspline_to_bezier_conversion_matrix
from trajectory_generation.objectives.objective_variables import get_control_points, get_scale_factor
from trajectory_generation.constraint_data_structures.constraint_function_data import ConstraintFunctionData
from trajectory_generation.constraint_functions.control_point_derivative_bounds import ControlPointDerivativeBounds
from trajectory_generation.matrix_evaluation import evaluate_point_derivative_on_interval
from trajectory_generation.constraint_functions.min_velocity_evaluator import MinVelocityEvaluator
class DerivativeConstraints(object):

    def __init__(self, dimension):
        self._dimension = dimension
        self.cont_pt_derivative_bounds = ControlPointDerivativeBounds(dimension)
        self.min_vel_eval = MinVelocityEvaluator(dimension)

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
                    min_velocity_constraint = self.calculate_min_velocity_constraint(control_points,scale_factor, derivative_bounds)
                    # min_velocity_constraint = self.calculate_min_velocity_constraint(bezier_velocity_control_points, derivative_bounds)
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
            if derivative_bounds.max_acceleration is not None or derivative_bounds.max_jerk is not None:
                acceleration_control_points = (velocity_control_points[:,1:] - velocity_control_points[:,0:-1])/scale_factor
                if derivative_bounds.max_acceleration is not None:
                    acceleration_constraint = self.calculate_max_acceleration_constraint(derivative_bounds, acceleration_control_points, scale_factor, M_a)
                    constraints[count] = acceleration_constraint
                    count += 1
                if derivative_bounds.max_jerk is not None:
                    jerk_constraint = self.calculate_max_jerk_constraint(derivative_bounds, acceleration_control_points, scale_factor)
                    constraints[count] = jerk_constraint
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
        if derivative_bounds.max_jerk is not None:
            length += 1
            constraint_key = np.concatenate((constraint_key,["max_jerk"]))
        constraint_array = np.zeros(length)
        return constraint_array, constraint_key, length

    def calculate_max_horizontal_velocity_constraint(self, bezier_velocity_control_points, derivative_bounds: DerivativeBounds):
        velocity_bound = np.max(np.linalg.norm(bezier_velocity_control_points[0:2,:],2,0))
        constraint = velocity_bound - derivative_bounds.max_horizontal_velocity
        return constraint

    def calculate_max_velocity_constraint(self, bezier_velocity_control_points, derivative_bounds: DerivativeBounds):
        velocity_bound = np.max(np.linalg.norm(bezier_velocity_control_points,2,0))
        constraint = (velocity_bound - derivative_bounds.max_velocity)
        return constraint

    def calculate_max_upward_velocity_constraint(self, bezier_velocity_control_points, derivative_bounds: DerivativeBounds):
        min_z_vel = np.min(bezier_velocity_control_points[2,:])
        constraint = -min_z_vel - derivative_bounds.max_upward_velocity
        return constraint

    def calculate_max_acceleration_constraint(self, derivative_bounds: DerivativeBounds, acceleration_control_points, scale_factor, M_a):
        bezier_acceleration_control_points = np.transpose(np.dot(M_a, np.transpose(acceleration_control_points)))
        dimension = np.shape(acceleration_control_points)[0]
        if derivative_bounds.gravity is not None and dimension == 3:
            gravity = np.array([[0],[0],[derivative_bounds.gravity]])
            bezier_acceleration_control_points = bezier_acceleration_control_points - gravity
        acceleration_bound = np.max(np.linalg.norm(bezier_acceleration_control_points,2,0))
        constraint = acceleration_bound - derivative_bounds.max_acceleration
        return constraint
    
    def calculate_max_jerk_constraint(self, derivative_bounds: DerivativeBounds, acceleration_control_points, scale_factor):
        jerk_control_points = (acceleration_control_points[:,1:] - acceleration_control_points[:,0:-1])/scale_factor
        jerk_bound = np.max(np.linalg.norm(jerk_control_points,2,0))
        constraint = jerk_bound - derivative_bounds.max_jerk
        return constraint

    def calculate_min_velocity_constraint(self, control_points, scale_factor, derivative_bounds: DerivativeBounds):
        # lower_bound_velocity = self.cont_pt_derivative_bounds.get_min_velocity_of_bez_vel_cont_pts(bezier_velocity_control_points)
        lower_bound_velocity = self.min_vel_eval.get_min_velocity_spline(control_points,scale_factor)
        constraint =  (derivative_bounds.min_velocity - lower_bound_velocity)
        return constraint
    
    # derivative_bounds: DerivativeBounds, num_cont_pts, dimension, order
    def create_tangential_acceleration_constraint(self, derivative_bounds: DerivativeBounds, num_cont_pts, dimension, order):
        if order != 3:
            raise Exception("Tangential Acceleration constraint for splines other than 3rd order not implemented.")
        else:
            num_intervals = num_cont_pts - order
            def tangential_acceleraiton_constraint_function(variables):
                control_points = get_control_points(variables, num_cont_pts, dimension)
                scale_factor = get_scale_factor(variables, num_cont_pts, dimension)
                constraints = np.zeros((2,num_intervals))
                for i in range(num_intervals):
                    ctrl_pts = control_points[:,i:i+order+1]
                    high_tang_accel, low_tang_accel = \
                        self.calculate_tangential_acceleration_bounds_for_interval(ctrl_pts, scale_factor)
                    constraints[0,i] = high_tang_accel
                    constraints[1,i] = low_tang_accel
                return constraints.flatten()

        lower_bound = derivative_bounds.min_tangential_acceleration
        upper_bound = derivative_bounds.max_tangential_acceleration
        constraint_class = "Derivative"
        tang_accel_constraint = NonlinearConstraint(tangential_acceleraiton_constraint_function , \
                                                     lb = lower_bound, ub = upper_bound)
        constraint_key = np.repeat(["max_tangential_acceleration" , "min_tangential_acceleration"],num_intervals).tolist()
        constraint_function_data = ConstraintFunctionData(tangential_acceleraiton_constraint_function, \
                                                          lower_bound, upper_bound, constraint_key, constraint_class)
        return tang_accel_constraint, constraint_function_data
        
    
    def calculate_tangential_acceleration_bounds_for_interval(self,ctrl_pts, scale_factor):
        high_dot_term, low_dot_term = self.calculate_dot_term_bounds_for_interval(ctrl_pts, scale_factor)
        min_vel = self.min_vel_eval.get_min_velocity_spline(ctrl_pts,scale_factor)
        upper_bound = high_dot_term / min_vel
        lower_bound = low_dot_term / min_vel
        return upper_bound, lower_bound

    def calculate_dot_term_bounds_for_interval(self, ctrl_pts, scale_factor):
        c0, c1, c2 = self.get_third_order_dot_term_coefficients(ctrl_pts)
        roots = self.solve_quadratic(c2, c1, c0)
        t0 = 0
        tf = scale_factor
        max_dot_term = self.calculate_dot_term_at_point_on_interval(ctrl_pts, scale_factor, t0)
        min_dot_term = max_dot_term
        times = np.array([roots.item(0)*scale_factor, roots.item(1)*scale_factor, tf])
        for i in range(len(times)):
            t = times.item(i)
            if t < t0 or t > tf:
                pass
            else:
                temp_dot_term = self.calculate_dot_term_at_point_on_interval(ctrl_pts, scale_factor, t)
                if temp_dot_term > max_dot_term:
                    max_dot_term = temp_dot_term
                if temp_dot_term < min_dot_term:
                    min_dot_term = temp_dot_term
        return max_dot_term, min_dot_term
    
    def calculate_dot_term_at_point_on_interval(self, ctrl_pts, scale_factor, t):
        tj = 0
        velocity = evaluate_point_derivative_on_interval(ctrl_pts, t, tj, scale_factor,1).flatten()
        acceleration = evaluate_point_derivative_on_interval(ctrl_pts, t, tj, scale_factor,2).flatten()
        tang_accel = np.dot(acceleration, velocity)
        return tang_accel

    def get_third_order_dot_term_coefficients(self, ctrl_pts):
        if self._dimension == 3:
            p0x = ctrl_pts[0,0]
            p1x = ctrl_pts[0,1]
            p2x = ctrl_pts[0,2]
            p3x = ctrl_pts[0,3]
            p0y = ctrl_pts[1,0]
            p1y = ctrl_pts[1,1]
            p2y = ctrl_pts[1,2]
            p3y = ctrl_pts[1,3]
            p0z = ctrl_pts[2,0]
            p1z = ctrl_pts[2,1]
            p2z = ctrl_pts[2,2]
            p3z = ctrl_pts[2,3]
            c2 = (p0x - 3*p1x + 3*p2x - p3x)*(p0x/2 - (3*p1x)/2 + (3*p2x)/2 - p3x/2) \
                + (p0y - 3*p1y + 3*p2y - p3y)*(p0y/2 - (3*p1y)/2 + (3*p2y)/2 - p3y/2) \
                + (p0z - 3*p1z + 3*p2z - p3z)*(p0z/2 - (3*p1z)/2 + (3*p2z)/2 - p3z/2) \
                + (p0x - 3*p1x + 3*p2x - p3x)**2 + (p0y - 3*p1y + 3*p2y - p3y)**2 \
                + (p0z - 3*p1z + 3*p2z - p3z)**2
            c1 = - (3*p0x - 6*p1x + 3*p2x)*(p0x - 3*p1x + 3*p2x - p3x) \
                - (3*p0y - 6*p1y + 3*p2y)*(p0y - 3*p1y + 3*p2y - p3y) \
                - (3*p0z - 6*p1z + 3*p2z)*(p0z - 3*p1z + 3*p2z - p3z)
            c0 = (p0x - 2*p1x + p2x)**2 + (p0y - 2*p1y + p2y)**2 + (p0z - 2*p1z + p2z)**2 \
                + (p0x/2 - p2x/2)*(p0x - 3*p1x + 3*p2x - p3x) \
                + (p0y/2 - p2y/2)*(p0y - 3*p1y + 3*p2y - p3y) \
                + (p0z/2 - p2z/2)*(p0z - 3*p1z + 3*p2z - p3z)
        else:
            p0x = ctrl_pts[0,0]
            p1x = ctrl_pts[0,1]
            p2x = ctrl_pts[0,2]
            p3x = ctrl_pts[0,3]
            p0y = ctrl_pts[1,0]
            p1y = ctrl_pts[1,1]
            p2y = ctrl_pts[1,2]
            p3y = ctrl_pts[1,3]
            c2 = (p0x - 3*p1x + 3*p2x - p3x)*(p0x/2 - (3*p1x)/2 + (3*p2x)/2 - p3x/2) \
                + (p0y - 3*p1y + 3*p2y - p3y)*(p0y/2 - (3*p1y)/2 + (3*p2y)/2 - p3y/2) \
                + (p0x - 3*p1x + 3*p2x - p3x)**2 + (p0y - 3*p1y + 3*p2y - p3y)**2 
            c1 = - (3*p0x - 6*p1x + 3*p2x)*(p0x - 3*p1x + 3*p2x - p3x) \
                - (3*p0y - 6*p1y + 3*p2y)*(p0y - 3*p1y + 3*p2y - p3y)
            c0 = (p0x - 2*p1x + p2x)**2 + (p0y - 2*p1y + p2y)**2 \
                + (p0x/2 - p2x/2)*(p0x - 3*p1x + 3*p2x - p3x) \
                + (p0y/2 - p2y/2)*(p0y - 3*p1y + 3*p2y - p3y)
        return c0, c1, c2

    def solve_quadratic(self, a_term, b_term, c_term):
        if (b_term*b_term - 4*a_term*c_term == 0):
            root_1 = -b_term/(2*a_term)
            roots = np.array([root_1, np.inf])
        elif (b_term*b_term - 4*a_term*c_term < 0):
            roots = np.array([np.inf, np.inf])
        else:
            root_1 = (-b_term + np.sqrt(b_term*b_term - 4*a_term*c_term))/(2*a_term)
            root_2 = (-b_term - np.sqrt(b_term*b_term - 4*a_term*c_term))/(2*a_term)
            roots = np.array([root_1, root_2])
        return roots