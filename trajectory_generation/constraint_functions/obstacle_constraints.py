import ctypes 
import pathlib 
import os 
import numpy as np
from scipy.optimize import NonlinearConstraint
from trajectory_generation.objectives.objective_variables import get_control_points, get_scale_factor
from trajectory_generation.constraint_data_structures.constraint_function_data import ConstraintFunctionData

script_dir = os.path.abspath(os.path.dirname(__file__))
libname_str = os.path.join(script_dir)
libname = pathlib.Path(libname_str)
lib = ctypes.CDLL(libname / "TrajectoryConstraintsCCode/build/src/libTrajectoryConstraints.so")

class ObstacleConstraints(object):

    def __init__(self, dimension):
        ND_POINTER_DOUBLE = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,flags="C")
        ND_POINTER_C_DOUBLE = np.ctypeslib.ndpointer(dtype=ctypes.c_double)
        self._dimension = dimension
        if dimension == 2:
            lib.ObstacleConstraints_2.argtypes = [ctypes.c_void_p]
            lib.ObstacleConstraints_2.restype = ctypes.c_void_p
            lib.getObstacleConstraintsForIntervals_2.argtypes = [ctypes.c_void_p, ND_POINTER_DOUBLE, ctypes.c_int, 
                ctypes.c_double, ND_POINTER_DOUBLE]
            lib.getObstacleConstraintsForIntervals_2.restype = ND_POINTER_C_DOUBLE
            lib.getObstacleConstraintForSpline_2.argtypes = [ctypes.c_void_p, ND_POINTER_DOUBLE, ctypes.c_int, 
                ctypes.c_double, ND_POINTER_DOUBLE]
            lib.getObstacleConstraintForSpline_2.restype = ctypes.c_double
            lib.getObstaclesConstraintsForSpline_2.argtypes = [ctypes.c_void_p, ND_POINTER_DOUBLE, ND_POINTER_DOUBLE, 
                ctypes.c_int, ND_POINTER_DOUBLE, ctypes.c_int]
            lib.getObstaclesConstraintsForSpline_2.restype = ND_POINTER_C_DOUBLE
            self.obj = lib.ObstacleConstraints_2(0)
        else: # value == 3
            lib.ObstacleConstraints_3.argtypes = [ctypes.c_void_p]
            lib.ObstacleConstraints_3.restype = ctypes.c_void_p
            lib.getObstacleConstraintsForIntervals_3.argtypes = [ctypes.c_void_p, ND_POINTER_DOUBLE, ctypes.c_int, 
                ctypes.c_double, ND_POINTER_DOUBLE]
            lib.getObstacleConstraintsForIntervals_3.restype = ND_POINTER_C_DOUBLE
            lib.getObstacleConstraintForSpline_3.argtypes = [ctypes.c_void_p, ND_POINTER_DOUBLE, ctypes.c_int, 
                ctypes.c_double, ND_POINTER_DOUBLE]
            lib.getObstacleConstraintForSpline_3.restype = ctypes.c_double
            lib.getObstaclesConstraintsForSpline_3.argtypes = [ctypes.c_void_p, ND_POINTER_DOUBLE, ND_POINTER_DOUBLE, 
                ctypes.c_int, ND_POINTER_DOUBLE, ctypes.c_int]
            lib.getObstaclesConstraintsForSpline_3.restype = ND_POINTER_C_DOUBLE
            self.obj = lib.ObstacleConstraints_3(0)

    def getObstacleConstraintsForIntervals(self, cont_pts, obstacle_radius, obstacle_center):
        num_cont_pts = np.shape(cont_pts)[1]
        cont_pts_array = cont_pts.flatten().astype('float64')
        obstacle_center_array = obstacle_center.flatten().astype('float64')
        order = 3
        num_intervals = num_cont_pts - order
        ND_POINTER_C_DOUBLE = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=(num_intervals))
        if self._dimension == 2:
            lib.getObstacleConstraintsForIntervals_2.restype = ND_POINTER_C_DOUBLE
            distances = lib.getObstacleConstraintsForIntervals_2(self.obj, cont_pts_array, 
                num_cont_pts, obstacle_radius, obstacle_center_array)
        else: # value = 3
            lib.getObstacleConstraintsForIntervals_3.restype = ND_POINTER_C_DOUBLE
            distances = lib.getObstacleConstraintsForIntervals_3(self.obj, cont_pts_array, 
                num_cont_pts, obstacle_radius, obstacle_center_array)
        return distances
    
    def getObstaclesConstraintsForSpline(self, cont_pts, obstacle_radii, obstacle_centers):
        num_cont_pts = np.shape(cont_pts)[1]
        num_obstacles = np.shape(obstacle_centers)[1]
        cont_pts_array = cont_pts.flatten().astype('float64')
        obstacle_center_array = obstacle_centers.flatten().astype('float64')
        obstacle_radii_array = obstacle_radii.flatten().astype('float64')
        ND_POINTER_C_DOUBLE = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=(num_obstacles))
        if self._dimension == 2:
            lib.getObstaclesConstraintsForSpline_2.restype = ND_POINTER_C_DOUBLE
            distances = lib.getObstaclesConstraintsForSpline_2(self.obj, obstacle_center_array, 
                obstacle_radii_array, num_obstacles, cont_pts_array, num_cont_pts)
        else: # value = 3
            lib.getObstaclesConstraintsForSpline_3.restype = ND_POINTER_C_DOUBLE
            distances = lib.getObstaclesConstraintsForSpline_3(self.obj, obstacle_center_array, 
                obstacle_radii_array, num_obstacles, cont_pts_array, num_cont_pts)
        return distances

    def getObstacleConstraintForSpline(self, cont_pts, obstacle_radius, obstacle_center):
        num_cont_pts = np.shape(cont_pts)[1]
        cont_pts_array = cont_pts.flatten().astype('float64')
        obstacle_center_array = obstacle_center.flatten().astype('float64')
        if self._dimension == 2:
            distance = lib.getObstacleConstraintForSpline_2(self.obj, cont_pts_array, 
                num_cont_pts, obstacle_radius, obstacle_center_array)
        else: # value = 3
            distance = lib.getObstacleConstraintForSpline_3(self.obj, cont_pts_array, 
                num_cont_pts, obstacle_radius, obstacle_center_array)
        return distance
    
    def create_obstacle_constraints(self, obstacles, num_cont_pts, dimension):
        num_obstacles = len(obstacles)
        constraints_key = initialize_constraints_key(num_obstacles)
        def obstacle_constraint_function(variables):
            control_points = get_control_points(variables, num_cont_pts, dimension)
            scale_factor = get_scale_factor(variables, num_cont_pts, dimension)
            radii = np.zeros(len(obstacles))
            centers = np.zeros((self._dimension,len(obstacles)))
            for i in range(len(obstacles)):
                radii[i] = obstacles[i].radius
                centers[0,i] = obstacles[i].center[0,0]
                centers[1,i] = obstacles[i].center[1,0]
                if self._dimension == 3:
                    centers[2,i] = obstacles[i].center[2,0]
            return self.getObstaclesConstraintsForSpline(control_points, radii, centers)
        lower_bound = np.zeros(num_obstacles)
        upper_bound = np.zeros(num_obstacles) + np.inf
        constraint_class = "Obstacle"
        obstacle_constraint = NonlinearConstraint(obstacle_constraint_function , lb = lower_bound, ub = upper_bound)
        constraint_function_data = ConstraintFunctionData(obstacle_constraint_function, lower_bound, upper_bound, constraints_key, constraint_class)
        return obstacle_constraint, constraint_function_data
    
def initialize_constraints_key(num_obstacles):
    constraints_key = np.array([])
    for i in range(num_obstacles):
        constraints_key = np.concatenate((constraints_key,["obstacle " + str(i+1)]))
    return constraints_key

# control_points = np.array([[7.91705873, 9.88263331, 0.27303466, 7.50604049, 4.61073475, 5.98801717, 1.52432928, 3.8850049, 1.61195392, 8.22471529],
#                            [5.22947263, 1.33282499, 3.51583204, 8.62435967, 3.03096953, 0.84672315, 0.54028843, 7.24686189, 4.79897482, 5.00498365]])
# obst_const = ObstacleConstraints(2)

# radius = 1
# center = np.array([4.84435679, 6.42836434])
# distance = obst_const.getObstacleConstraintForSpline(control_points, radius, center)
# print("distance: " , distance)
# distances = obst_const.getObstacleConstraintsForIntervals(control_points, radius, center)
# print("distances: " , distances)
# centers = np.array([[1,2],[6,7]])
# radii = np.array([2,1])
# distances_obj = obst_const.getObstaclesConstraintsForSpline(control_points, radii, centers)
# print("distances_obj : " , distances_obj)