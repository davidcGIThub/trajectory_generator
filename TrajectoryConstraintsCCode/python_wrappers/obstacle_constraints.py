import ctypes 
import pathlib 
import os 
import numpy as np

script_dir = os.path.abspath(os.path.dirname(__file__))
libname_str = os.path.join(script_dir)
libname = pathlib.Path(libname_str)
lib = ctypes.CDLL(libname / "../build/src/libPathObjectivesAndConstraints.so")

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