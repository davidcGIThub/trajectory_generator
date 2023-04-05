import ctypes 
import pathlib 
import os 
import numpy as np

script_dir = os.path.abspath(os.path.dirname(__file__))
libname_str = os.path.join(script_dir)
libname = pathlib.Path(libname_str)
lib = ctypes.CDLL(libname / "libPathObjectivesAndConstraints.so")

class ObjectiveFunctions(object):

    def __init__(self, dimension):
        ND_POINTER_DOUBLE = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,flags="C")
        self._dimension = dimension
        if dimension == 2:
            lib.ObjectiveFunctions_2.argtypes = [ctypes.c_void_p]
            lib.ObjectiveFunctions_2.restype = ctypes.c_void_p
            lib.minimize_acceleration_and_time_2.argtypes = [ctypes.c_void_p, ND_POINTER_DOUBLE, ctypes.c_int, 
                ctypes.c_double]
            lib.minimize_acceleration_and_time_2.restype = ctypes.c_double
            lib.minimize_distance_and_time_2.argtypes = [ctypes.c_void_p, ND_POINTER_DOUBLE, ctypes.c_int, 
                ctypes.c_double]
            lib.minimize_distance_and_time_2.restype = ctypes.c_double
            self.obj = lib.ObjectiveFunctions_2(0)
        else: # value == 3
            lib.ObjectiveFunctions_3.argtypes = [ctypes.c_void_p]
            lib.ObjectiveFunctions_3.restype = ctypes.c_void_p
            lib.minimize_acceleration_and_time_3.argtypes = [ctypes.c_void_p, ND_POINTER_DOUBLE, ctypes.c_int, 
                ctypes.c_double]
            lib.minimize_acceleration_and_time_3.restype = ctypes.c_double
            lib.minimize_distance_and_time_3.argtypes = [ctypes.c_void_p, ND_POINTER_DOUBLE, ctypes.c_int, 
                ctypes.c_double]
            lib.minimize_distance_and_time_3.restype = ctypes.c_double
            self.obj = lib.ObjectiveFunctions_3(0)

    def minimize_acceleration_and_time(self, cont_pts, scale_factor):
        num_cont_pts = np.shape(cont_pts)[1]
        cont_pts_array = cont_pts.flatten().astype('float64')
        if self._dimension == 2:
            objective = lib.minimize_acceleration_and_time_2(self.obj, cont_pts_array, num_cont_pts, scale_factor)
        else: # value = 3
            objective = lib.minimize_acceleration_and_time_3(self.obj, cont_pts_array, num_cont_pts, scale_factor)
        return objective
    
    def minimize_distance_and_time(self, cont_pts, scale_factor):
        num_cont_pts = np.shape(cont_pts)[1]
        cont_pts_array = cont_pts.flatten().astype('float64')
        if self._dimension == 2:
            objective = lib.minimize_distance_and_time_2(self.obj, cont_pts_array, num_cont_pts, scale_factor)
        else: # value = 3
            objective = lib.minimize_distance_and_time_3(self.obj, cont_pts_array, num_cont_pts, scale_factor)
        return objective
    

# control_points = np.array([[7.91705873, 9.88263331, 0.27303466, 7.50604049, 4.61073475, 5.98801717, 1.52432928, 3.8850049, 1.61195392, 8.22471529],
#                            [5.22947263, 1.33282499, 3.51583204, 8.62435967, 3.03096953, 0.84672315, 0.54028843, 7.24686189, 4.79897482, 5.00498365]])
# control_points = np.array([[2, 3, 2, 7, 8, 5, 6, 2],
#                            [6, 8, 3, 0, 3, 8, 2, 4],
#                            [9, 2, 0, 9, 8, 2, 9, 3]])
# scale_factor = 1
# obj_func = ObjectiveFunctions(3)
# objective = obj_func.minimize_acceleration_and_time(control_points, scale_factor)
# print("objective: " , objective)


