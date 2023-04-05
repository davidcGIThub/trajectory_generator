import ctypes 
import pathlib 
import os 
import numpy as np

script_dir = os.path.abspath(os.path.dirname(__file__))
libname_str = os.path.join(script_dir)
libname = pathlib.Path(libname_str)
lib = ctypes.CDLL(libname / "libPathObjectivesAndConstraints.so")

class WaypointConstraints(object):

    def __init__(self, dimension):
        ND_POINTER_DOUBLE = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,flags="C")
        ND_POINTER_BOOL = np.ctypeslib.ndpointer(dtype=bool, ndim=1,flags="C")
        ND_POINTER_C_DOUBLE = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=(dimension,2))
        ND_POINTER_C_DOUBLE_2 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=(2))
        self._dimension = dimension
        if dimension == 2:
            lib.WaypointConstraints_2.argtypes = [ctypes.c_void_p]
            lib.WaypointConstraints_2.restype = ctypes.c_void_p
            lib.velocity_at_waypoints_constraints_2.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double, ND_POINTER_DOUBLE, ND_POINTER_BOOL]
            lib.velocity_at_waypoints_constraints_2.restype = ND_POINTER_C_DOUBLE
            lib.acceleration_at_waypoints_constraints_2.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double, ND_POINTER_DOUBLE]
            lib.acceleration_at_waypoints_constraints_2.restype = ND_POINTER_C_DOUBLE
            lib.curvature_at_waypoints_constraints_2.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double, ND_POINTER_DOUBLE, ND_POINTER_BOOL]
            lib.curvature_at_waypoints_constraints_2.restype = ND_POINTER_C_DOUBLE_2
            lib.direction_at_waypoints_constraints_2.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double, ND_POINTER_DOUBLE]
            lib.direction_at_waypoints_constraints_2.restype = ND_POINTER_C_DOUBLE
            self.obj = lib.WaypointConstraints_2(0)
        else: # value == 3
            lib.WaypointConstraints_3.argtypes = [ctypes.c_void_p]
            lib.WaypointConstraints_3.restype = ctypes.c_void_p
            lib.velocity_at_waypoints_constraints_3.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double, ND_POINTER_DOUBLE, ND_POINTER_BOOL]
            lib.velocity_at_waypoints_constraints_3.restype = ND_POINTER_C_DOUBLE
            lib.acceleration_at_waypoints_constraints_3.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double, ND_POINTER_DOUBLE]
            lib.acceleration_at_waypoints_constraints_3.restype = ND_POINTER_C_DOUBLE
            lib.curvature_at_waypoints_constraints_3.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double, ND_POINTER_DOUBLE, ND_POINTER_BOOL]
            lib.curvature_at_waypoints_constraints_3.restype = ND_POINTER_C_DOUBLE_2
            lib.direction_at_waypoints_constraints_3.argtypes = [ctypes.c_void_p, 
                ND_POINTER_DOUBLE, ctypes.c_int, ctypes.c_double, ND_POINTER_DOUBLE]
            lib.direction_at_waypoints_constraints_3.restype = ND_POINTER_C_DOUBLE
            self.obj = lib.WaypointConstraints_3(0)

    def velocity_at_waypoints_constraints(self, cont_pts, scale_factor, desired_velocities, switches):
        num_cont_pts = np.shape(cont_pts)[1]
        cont_pts_array = cont_pts.flatten().astype('float64')
        des_vel_array = desired_velocities.flatten().astype('float64')
        switches_array = switches.flatten().astype('bool')
        if self._dimension == 2:
            objective = lib.velocity_at_waypoints_constraints_2(self.obj, cont_pts_array, num_cont_pts, scale_factor, des_vel_array, switches_array)
        else: # value = 3
            objective = lib.velocity_at_waypoints_constraints_3(self.obj, cont_pts_array, num_cont_pts, scale_factor, des_vel_array, switches_array)
        return objective
    
    def acceleration_at_waypoints_constraints(self, cont_pts, scale_factor, desired_accelerations):
        num_cont_pts = np.shape(cont_pts)[1]
        cont_pts_array = cont_pts.flatten().astype('float64')
        des_accel_array = desired_accelerations.flatten().astype('float64')
        if self._dimension == 2:
            objective = lib.acceleration_at_waypoints_constraints_2(self.obj, cont_pts_array, num_cont_pts, scale_factor, des_accel_array)
        else: # value = 3
            objective = lib.acceleration_at_waypoints_constraints_3(self.obj, cont_pts_array, num_cont_pts, scale_factor, des_accel_array)
        return objective
    
    def curvature_at_waypoints_constraints(self, cont_pts, scale_factor, desired_curvatures, switches):
        num_cont_pts = np.shape(cont_pts)[1]
        cont_pts_array = cont_pts.flatten().astype('float64')
        des_curvature_array = desired_curvatures.flatten().astype('float64')
        switches_array = switches.flatten().astype('bool')
        if self._dimension == 2:
            objective = lib.curvature_at_waypoints_constraints_2(self.obj, cont_pts_array, num_cont_pts, scale_factor, des_curvature_array, switches_array)
        else: # value = 3
            objective = lib.curvature_at_waypoints_constraints_3(self.obj, cont_pts_array, num_cont_pts, scale_factor, des_curvature_array, switches_array)
        return objective

    def direction_at_waypoints_constraints(self, cont_pts, scale_factor, desired_directions):
        num_cont_pts = np.shape(cont_pts)[1]
        cont_pts_array = cont_pts.flatten().astype('float64')
        des_vel_array = desired_directions.flatten().astype('float64')
        if self._dimension == 2:
            objective = lib.direction_at_waypoints_constraints_2(self.obj, cont_pts_array, num_cont_pts, scale_factor, des_vel_array)
        else: # value = 3
            objective = lib.direction_at_waypoints_constraints_3(self.obj, cont_pts_array, num_cont_pts, scale_factor, des_vel_array)
        return objective
    
# control_points = np.array([[3, 0, 9, 5, 7, 8, 0, 1],
#                             [1, 6, 2, 2, 1, 4, 1, 4]])
# desired_velocities = np.array([[7, 3],
#                                [1, 4]])
# desired_velocities = np.array([[4, 4],
#                                [2, 7]])
# switches = np.array([False, False])
# ### desired_velocities = desired_velocities / np.linalg.norm(desired_velocities,2,0)
# scale_factor = 1
# const_func = WaypointConstraints(2)
# vel_at_waypoints_constraints = const_func.velocity_at_waypoints_constraints(control_points, scale_factor, desired_velocities, switches)
# print("vel_at_waypoints_constraints: " , vel_at_waypoints_constraints)

# accel_at_waypoints_constraints = const_func.acceleration_at_waypoints_constraints(control_points, scale_factor, desired_velocities)
# print("accel_at_waypoints_constraints: " , accel_at_waypoints_constraints)

# control_points = np.array([[0, 9, 0, 8, 5, 8, 6, 7],
#                             [1, 5, 1, 6, 5, 4, 2, 7],
#                             [5, 4, 5, 6, 8, 7, 5, 9]])
# desired_curvatures = np.array([0, 1.5574517574154207])
# desired_curvatures = np.array([3, 0.54])
# const_func_3 = WaypointConstraints(3)
# curvature_at_waypoints_constraints = const_func_3.curvature_at_waypoints_constraints(control_points, scale_factor, desired_curvatures,switches)
# print("curve_at_waypoints_constraints: " , curvature_at_waypoints_constraints)

# control_points = np.array([[3., 3, 3., 6.37746426, 12.83502874, 20.08143935,25.50999347, 27.87858676],
#         [0.14504009,  3.97819311,  7.94218749, 11.97716993, 15.83648356, 18.98674159, 20.50662921, 18.98674159]])
# desired_directions = np.array([[0, 1],
#                                [1, 0]])
# const_func_2 = WaypointConstraints(2)
# directions_at_waypoints_constraints = const_func_2.direction_at_waypoints_constraints(control_points, scale_factor, desired_directions)
# print("directions_at_waypoints_constraints: " , directions_at_waypoints_constraints)