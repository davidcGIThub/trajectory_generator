import ctypes 
import pathlib 
import os 
import numpy as np
from scipy.optimize import NonlinearConstraint
from trajectory_generation.objectives.objective_variables import get_objective_variables
from trajectory_generation.constraint_data_structures.dynamic_bounds import TurningBound
from trajectory_generation.constraint_data_structures.constraint_function_data import ConstraintFunctionData

script_dir = os.path.abspath(os.path.dirname(__file__))
libname_str = os.path.join(script_dir)
libname = pathlib.Path(libname_str)
lib = ctypes.CDLL(libname / "TrajectoryConstraintsCCode/build/src/libTrajectoryConstraints.so")

class TurningConstraints(object):

    def __init__(self, dimension):
        self._order = 3
        ND_POINTER_DOUBLE = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,flags="C")
        self._dimension = dimension
        if dimension == 2:
            lib.CrossTermBounds_2.argtypes = [ctypes.c_void_p]
            lib.CrossTermBounds_2.restype = ctypes.c_void_p
            lib.get_spline_curvature_bound_2.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int]
            lib.get_spline_curvature_bound_2.restype = ctypes.c_double
            lib.get_spline_angular_rate_bound_2.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double]
            lib.get_spline_angular_rate_bound_2.restype = ctypes.c_double
            lib.get_spline_centripetal_acceleration_bound_2.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double]
            lib.get_spline_centripetal_acceleration_bound_2.restype = ctypes.c_double
            self.obj = lib.CrossTermBounds_2(0)
        else: # value == 3
            lib.CrossTermBounds_3.argtypes = [ctypes.c_void_p]
            lib.CrossTermBounds_3.restype = ctypes.c_void_p
            lib.get_spline_curvature_bound_3.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int]
            lib.get_spline_curvature_bound_3.restype = ctypes.c_double
            lib.get_spline_angular_rate_bound_3.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double]
            lib.get_spline_angular_rate_bound_3.restype = ctypes.c_double
            lib.get_spline_centripetal_acceleration_bound_3.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double]
            lib.get_spline_centripetal_acceleration_bound_3.restype = ctypes.c_double
            self.obj = lib.CrossTermBounds_3(0)

    def get_spline_curvature_constraint(self, cont_pts, max_curvature):
        num_cont_pts = np.shape(cont_pts)[1]
        cont_pts_array = cont_pts.flatten().astype('float64')
        if self._dimension == 2:
            bound = lib.get_spline_curvature_bound_2(self.obj, cont_pts_array, num_cont_pts)
        else: # value = 3
            bound = lib.get_spline_curvature_bound_3(self.obj, cont_pts_array, num_cont_pts)
        constraint = np.array([bound - max_curvature])
        return constraint
    
    def get_spline_angular_rate_constraint(self, cont_pts, max_angular_rate, scale_factor):
        num_cont_pts = np.shape(cont_pts)[1]
        cont_pts_array = cont_pts.flatten().astype('float64')
        if self._dimension == 2:
            bound = lib.get_spline_angular_rate_bound_2(self.obj, cont_pts_array, num_cont_pts, scale_factor)
        else: # value = 3
            bound = lib.get_spline_angular_rate_bound_3(self.obj, cont_pts_array, num_cont_pts, scale_factor)
        constraint = np.array([bound - max_angular_rate])
        return constraint
    
    def get_spline_centripetal_acceleration_constraint(self, cont_pts, max_centripetal_acceleration, scale_factor):
        num_cont_pts = np.shape(cont_pts)[1]
        cont_pts_array = cont_pts.flatten().astype('float64')
        if self._dimension == 2:
            bound = lib.get_spline_centripetal_acceleration_bound_2(self.obj, cont_pts_array, num_cont_pts, scale_factor)
        else: # value = 3
            bound = lib.get_spline_centripetal_acceleration_bound_3(self.obj, cont_pts_array, num_cont_pts, scale_factor)
        constraint = np.array([bound - max_centripetal_acceleration])
        return constraint
    
    def create_turning_constraint(self, turning_bound: TurningBound, num_cont_pts, dimension):
        def centripetal_acceleration_constraint_function(variables):
            control_points, scale_factor = get_objective_variables(variables, num_cont_pts, dimension)
            const = self.get_spline_centripetal_acceleration_constraint(control_points, turning_bound.max_turning_bound, scale_factor)
            return const
        def angular_rate_constraint_function(variables):
            control_points, scale_factor = get_objective_variables(variables, num_cont_pts, dimension)
            return self.get_spline_angular_rate_constraint(control_points, turning_bound.max_turning_bound, scale_factor)
        def curvature_constraint_function(variables):
            control_points, scale_factor = get_objective_variables(variables, num_cont_pts, dimension)
            return self.get_spline_curvature_constraint(control_points,turning_bound.max_turning_bound)
        if turning_bound.bound_type == "curvature":
            constraint_function = curvature_constraint_function
        elif turning_bound.bound_type == "centripetal_acceleration":
            constraint_function = centripetal_acceleration_constraint_function
        elif turning_bound.bound_type == "angular_rate":
            constraint_function = angular_rate_constraint_function
        else:
            raise Exception("Not valid turning bound type")
        lower_bound = np.array([-np.inf])
        upper_bound = np.array([0])
        constraint_key = np.array([turning_bound.bound_type])
        constraint_function_data = ConstraintFunctionData(constraint_function, lower_bound, upper_bound, constraint_key)
        turning_constraint = NonlinearConstraint(constraint_function , lb = lower_bound, ub = upper_bound)
        return turning_constraint, constraint_function_data
    
# control_points = np.array([[4, 1, 4, 5, 6, 5],
#                            [2, 2, 0, 4, 3, 2],
#                            [7, 0, 1, 7, 8, 1]])
# scale_factor = 1
# dimension = 3
# num_control_points = 5
# turn_const = TurningConstraints(dimension)
# max_curvature = 3.1
# max_angular_rate = 0
# max_centripetal_acceleration = 0
# curvature_constraint = turn_const.get_spline_curvature_constraint(control_points, max_curvature)
# angular_rate_constraint = turn_const.get_spline_angular_rate_constraint(control_points, max_angular_rate,  scale_factor)
# centripetal_acceleration_constraint = turn_const.get_spline_centripetal_acceleration_constraint(control_points,max_centripetal_acceleration, scale_factor)
# print("curvature_constraint: " , curvature_constraint)
# print("angular_rate_constraint: " , angular_rate_constraint)
# print("centripetal_acceleration_constraint: " , centripetal_acceleration_constraint)