import ctypes 
import pathlib 
import os 
import numpy as np
from scipy.optimize import NonlinearConstraint
from trajectory_generation.objectives.objective_variables import get_control_points, get_scale_factor
from trajectory_generation.constraint_data_structures.dynamic_bounds import TurningBound
from trajectory_generation.constraint_data_structures.constraint_function_data import ConstraintFunctionData
from trajectory_generation.constraint_data_structures.waypoint_data import WaypointData
from bsplinegenerator.bsplines import BsplineEvaluation

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
        constraint = np.array([bound - max_curvature])*100
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
    
    def create_turning_constraint(self, turning_bound: TurningBound, num_cont_pts, 
                                  dimension, waypoint_data: WaypointData):
        def centripetal_acceleration_constraint_function(variables):
            control_points = get_control_points(variables, num_cont_pts, dimension)
            if waypoint_data.start_waypoint.checkIfZeroVel():
                control_points = control_points[:,1:]
            if waypoint_data.end_waypoint.checkIfZeroVel():
                control_points = control_points[:,0:-1]
            scale_factor = get_scale_factor(variables, num_cont_pts, dimension)
            const = self.get_spline_centripetal_acceleration_constraint(control_points, turning_bound.max_turning_bound, scale_factor)
            return const
        def angular_rate_constraint_function(variables):
            control_points = get_control_points(variables, num_cont_pts, dimension)
            if waypoint_data.start_waypoint.checkIfZeroVel():
                control_points = control_points[:,1:]
            if waypoint_data.end_waypoint.checkIfZeroVel():
                control_points = control_points[:,0:-1]
            scale_factor = get_scale_factor(variables, num_cont_pts, dimension)
            constraint = self.get_spline_angular_rate_constraint(control_points, turning_bound.max_turning_bound, scale_factor)
            return constraint
        def curvature_constraint_function(variables):
            control_points = get_control_points(variables, num_cont_pts, dimension)
            if waypoint_data.start_waypoint.checkIfZeroVel():
                control_points = control_points[:,1:]
            if waypoint_data.end_waypoint.checkIfZeroVel():
                control_points = control_points[:,0:-1]
            return self.get_spline_curvature_constraint(control_points,turning_bound.max_turning_bound)
        if turning_bound.bound_type == "curvature":
            constraint_function = curvature_constraint_function
        elif turning_bound.bound_type == "centripetal_acceleration":
            constraint_function = centripetal_acceleration_constraint_function
        elif turning_bound.bound_type == "angular_rate":
            constraint_function = angular_rate_constraint_function
        else:
            raise Exception("Not valid turning bound type")
        lower_bound = -np.inf
        upper_bound = 0
        constraint_key = np.array([turning_bound.bound_type])
        constraint_class = "Turning"
        constraint_function_data = ConstraintFunctionData(constraint_function, lower_bound, upper_bound, constraint_key, constraint_class)
        turning_constraint = NonlinearConstraint(constraint_function , lb = lower_bound, ub = upper_bound)
        return turning_constraint, constraint_function_data
    
# control_points = np.array([[-4.83421641e+00, -5.08289179e+00 ,-4.83421641e+00 ,-4.54704421e+00,
#   -2.68488265e+00, -3.06836818e-03 , 2.69674342e+00 , 4.52514537e+00,
#    4.81542044e+00,  5.09228978e+00 , 4.81542044e+00],
#  [-5.97971340e+00 , 5.14374272e-01 , 3.92221631e+00 , 4.20371352e+00,
#    2.86687036e+00 ,-6.21545367e-03 ,-2.89622805e+00 ,-4.21499103e+00,
#   -3.92631472e+00, -5.12325069e-01,  5.97561499e+00]])
# scale_factor:  0.17682017335415903
# dimension = 2
# num_control_points = 4
# order = 3

# turn_const = TurningConstraints(dimension)
# max_curvature = 3.1
# max_angular_rate = 0
# max_centripetal_acceleration = 47.12388980384689
# curvature_constraint = turn_const.get_spline_curvature_constraint(control_points, max_curvature)
# angular_rate_constraint = turn_const.get_spline_angular_rate_constraint(control_points, max_angular_rate,  scale_factor)
# num_intervals = 11
# for i in range(num_intervals):
#     ctrl_pts = control_points[:,i:i+order+1]
#     constraint = turn_const.get_spline_centripetal_acceleration_constraint(ctrl_pts,max_centripetal_acceleration, scale_factor)
#     centr_bound = constraint + max_centripetal_acceleration
#     print("centr_bound: ", centr_bound)
#     bspline = BsplineEvaluation()
#     centr_data, time_data = bspline.get_centripetal_acceleration_data(10000)
#     print("true bound: " , np.max(centr_data))
# print("curvature_constraint: " , curvature_constraint)
# print("angular_rate_constraint: " , angular_rate_constraint)
# print("centripetal_acceleration_constraint: " , centripetal_acceleration_constraint)