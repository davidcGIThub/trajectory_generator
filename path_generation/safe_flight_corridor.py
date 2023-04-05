import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class SFC_2D:
    """Safe Flight Corridor Data Class"""
    # rotation @ translation gives true center of sfc
    dimensions: np.ndarray(dtype=np.float64, shape=(2,1))
    translation: np.ndarray(dtype=np.float64, shape=(2,1))
    rotation: np.ndarray(dtype=np.float64, shape=(2,2))

    def getRotatedBounds(self):
        max_bounds = self.translation + self.dimensions/2
        min_bounds = self.translation - self.dimensions/2
        return min_bounds, max_bounds
    
    def getPointsToPlot(self):
        min_bounds, max_bounds = self.getRotatedBounds()
        x_min = min_bounds.item(0)
        x_max = max_bounds.item(0)
        y_min = min_bounds.item(1)
        y_max = max_bounds.item(1)
        points_unrotated = np.array([[x_min, x_min, x_max, x_max, x_min],
                            [y_min, y_max, y_max, y_min, y_min]])
        points_rotated = self.rotation @ points_unrotated
        return points_rotated
    
def get2DRotationAndTranslationFromPoints(point_1,point_2):
    # returns rotation transforms x_vector to vector paralell
    distance = point_1 - point_2
    dx = distance.item(0)
    dy = distance.item(1)
    psi = np.arctan2(dy,dx)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    rotation = np.array([[c_psi, -s_psi],
                 [s_psi, c_psi]])
    translation = rotation.T @ (point_1 + point_2)/2
    min_length = np.linalg.norm(distance,2)
    return rotation, translation, min_length

def plot_2D_sfc(sfc: SFC_2D):
    points_rotated = sfc.getPointsToPlot()
    plt.plot(points_rotated[0,:], points_rotated[1,:])

def plot_2D_sfcs(sfcs: list):
    if sfcs != None:
        for sfc_index in range(len(sfcs)):
            plot_2D_sfc(sfcs[sfc_index])

@dataclass
class SFC_3D:
    """Safe Flight Corridor Data Class"""
    # rotation @ translation gives true center of sfc
    dimensions: np.ndarray(dtype=np.float64, shape=(3,1))
    translation: np.ndarray(dtype=np.float64, shape=(3,1))
    rotation: np.ndarray(dtype=np.float64, shape=(3,3))

    def getRotatedBounds(self):
        max_bounds = self.translation + self.dimensions/2
        min_bounds = self.translation - self.dimensions/2
        return min_bounds, max_bounds
    
    def getPointsToPlot(self):
        min_bounds, max_bounds = self.getRotatedBounds()
        x_min = min_bounds.item(0)
        x_max = max_bounds.item(0)
        y_min = min_bounds.item(1)
        y_max = max_bounds.item(1)
        z_min = min_bounds.item(2)
        z_max = max_bounds.item(2)
        points = np.array([[x_max, x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_max, x_max, x_max, x_min, x_min, x_min, x_min],
                           [y_min, y_min, y_max, y_max, y_max, y_max, y_min, y_min, y_min, y_max, y_max, y_min, y_min, y_min, y_max, y_max],
                           [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max, z_min, z_min, z_max, z_max, z_max, z_min, z_min, z_max]])
        points = self.rotation @ points
        return points

def get3DRotationAndTranslationFromPoints(point_1,point_2):
    # returns rotation transforms x_vector to vector paralell
    distance_1 = point_1 - point_2
    dx_1 = distance_1.item(0)
    dz_1 = distance_1.item(2)
    theta = np.arctan2(dz_1,dx_1)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    Ry = np.array([[c_theta, 0, s_theta],
                [0, 1, 0],
                [-s_theta, 0, c_theta]])
    distance_2 = Ry @ distance_1
    dx_2 = distance_2.item(0)
    dy_2 = distance_2.item(1)
    psi = np.arctan2(dy_2,dx_2)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    Rz = np.array([[c_psi, -s_psi, 0],
                 [s_psi, c_psi, 0],
                 [0,       0,       1]])
    rotation = Ry.T @ Rz
    translation = rotation.T @ (point_1 + point_2)/2
    min_length = min_length = np.linalg.norm(distance_1,2)
    return rotation, translation, min_length

def plot_3D_sfc(sfc: SFC_3D, ax):
    points_rotated = sfc.getPointsToPlot()
    ax.plot(points_rotated[0,:], points_rotated[1,:],points_rotated[2,:])

def plot_3D_sfcs(sfcs: list, ax):
    if sfcs != None:
        for sfc_index in range(len(sfcs)):
            plot_3D_sfc(sfcs[sfc_index], ax)



