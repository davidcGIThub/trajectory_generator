import ctypes 
import pathlib 
import os 
import numpy as np


script_dir = os.path.abspath(os.path.dirname(__file__))
libname_str = os.path.join(script_dir)
libname = pathlib.Path(libname_str)
lib = ctypes.CDLL(libname / "TrajectoryConstraintsCCode/build/src/libTrajectoryConstraints.so")

class ControlPointDerivativeBounds(object):

    def __init__(self, dimension):
        self._order = 3
        ND_POINTER_DOUBLE = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,flags="C")
        self._dimension = dimension
        if dimension == 2:
            lib.ControlPointDerivativeBounds_2.argtypes = [ctypes.c_void_p]
            lib.ControlPointDerivativeBounds_2.restype = ctypes.c_void_p
            lib.find_min_velocity_of_bez_vel_cont_pts_2.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int]
            lib.find_min_velocity_of_bez_vel_cont_pts_2.restype = ctypes.c_double
            self.obj = lib.ControlPointDerivativeBounds_2(0)
        else: # value == 3
            lib.ControlPointDerivativeBounds_3.argtypes = [ctypes.c_void_p]
            lib.ControlPointDerivativeBounds_3.restype = ctypes.c_void_p
            lib.find_min_velocity_of_bez_vel_cont_pts_3.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int]
            lib.find_min_velocity_of_bez_vel_cont_pts_3.restype = ctypes.c_double
            self.obj = lib.ControlPointDerivativeBounds_3(0)

    def get_min_velocity_of_bez_vel_cont_pts(self, bez_vel_cont_pts):
        num_cont_pts = np.shape(bez_vel_cont_pts)[1]
        cont_pts_array = bez_vel_cont_pts.flatten().astype('float64')
        if self._dimension == 2:
            bound = lib.find_min_velocity_of_bez_vel_cont_pts_2(self.obj, cont_pts_array, num_cont_pts)
        else: # value = 3
            bound = lib.find_min_velocity_of_bez_vel_cont_pts_3(self.obj, cont_pts_array, num_cont_pts)
        bound
        return bound
 
# dimension = 2
# cp_deriv_bounds = ControlPointDerivativeBounds(dimension)
# bez_vel_cont_pts = np.array([[0.89402549, -0.05285741,  2.10545513,  2.47300498, 3.79358126,  4.76115495],
#                          [5.11942253,  4.76115495, -0.10547684,  0.05273842, -0.10547684, -0.47275804]])
# bound = cp_deriv_bounds.get_min_velocity_of_bez_vel_cont_pts(bez_vel_cont_pts)
# print("bound: " , bound)