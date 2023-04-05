import ctypes 
import pathlib 
import os 
import numpy as np

script_dir = os.path.abspath(os.path.dirname(__file__))
libname_str = os.path.join(script_dir)
libname = pathlib.Path(libname_str)
lib = ctypes.CDLL(libname / "libPathObjectivesAndConstraints.so")

class CurvatureConstraints(object):

    def __init__(self, dimension):
        self._order = 3
        ND_POINTER_DOUBLE = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,flags="C")
        nd_pointer_c_double = np.ctypeslib.ndpointer(dtype=ctypes.c_double)
        self._dimension = dimension
        if dimension == 2:
            lib.CurvatureConstraints_2.argtypes = [ctypes.c_void_p]
            lib.CurvatureConstraints_2.restype = ctypes.c_void_p
            lib.get_spline_curvature_constraint_2.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double]
            lib.get_spline_curvature_constraint_2.restype = ctypes.c_double
            lib.get_interval_curvature_constraints_2.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double]
            lib.get_interval_curvature_constraints_2.restype = nd_pointer_c_double
            self.obj = lib.CurvatureConstraints_2(0)
        else: # value == 3
            lib.CurvatureConstraints_3.argtypes = [ctypes.c_void_p]
            lib.CurvatureConstraints_3.restype = ctypes.c_void_p
            lib.get_spline_curvature_constraint_3.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double]
            lib.get_spline_curvature_constraint_3.restype = ctypes.c_double
            lib.get_interval_curvature_constraints_3.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double]
            lib.get_interval_curvature_constraints_3.restype = nd_pointer_c_double
            self.obj = lib.CurvatureConstraints_3(0)

    def get_spline_curvature_constraint(self, cont_pts, max_curvature):
        num_cont_pts = np.shape(cont_pts)[1]
        cont_pts_array = cont_pts.flatten().astype('float64')
        if self._dimension == 2:
            constraint = lib.get_spline_curvature_constraint_2(self.obj, cont_pts_array, num_cont_pts, max_curvature)
        else: # value = 3
            constraint = lib.get_spline_curvature_constraint_3(self.obj, cont_pts_array, num_cont_pts, max_curvature)
        return constraint
    
    def get_interval_curvature_constraints(self, cont_pts, max_curvature):
        num_cont_pts = np.shape(cont_pts)[1]
        cont_pts_array = cont_pts.flatten().astype('float64')
        num_intervals = num_cont_pts - self._order
        nd_pointer_c_double = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=(num_intervals))
        if self._dimension == 2:
            lib.get_interval_curvature_constraints_2.restype = nd_pointer_c_double
            constraints = lib.get_interval_curvature_constraints_2(self.obj, cont_pts_array, num_cont_pts, max_curvature)
        else: # value = 3
            lib.get_interval_curvature_constraints_3.restype = nd_pointer_c_double
            constraints = lib.get_interval_curvature_constraints_3(self.obj, cont_pts_array, num_cont_pts, max_curvature)
        return constraints
    
# control_points = np.array([[4, 1, 4, 5, 6, 5],
#                            [2, 2, 0, 4, 3, 2],
#                            [7, 0, 1, 7, 8, 1]])
# max_curvature = 2
# dimension = 3
# num_control_points = 5
# curve_const = CurvatureConstraints(dimension,num_control_points)
# curvature_constraint = curve_const.get_spline_curvature_constraint(control_points, max_curvature)
# print("curvature_constraint: " , curvature_constraint)

# curvature_constraints = curve_const.get_interval_curvature_constraints(control_points, max_curvature)
# print("curvature_constraints: " , curvature_constraints)