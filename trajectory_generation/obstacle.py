import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class Obstacle:
    center: np.ndarray
    radius: np.double

def plot_2D_obstacle(obstacle: Obstacle, ax):
    circle = plt.Circle((obstacle.center.item(0), obstacle.center.item(1)), 
                        obstacle.radius, color='r')
    ax.add_patch(circle)

def plot_2D_obstacles(obstacles: list, ax):
    for i in range(len(obstacles)):
        plot_2D_obstacle(obstacles[i], ax)

def plot_3D_obstacle(obstacle: Obstacle, ax):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = obstacle.radius * np.cos(u)*np.sin(v) + obstacle.center.item(0)
    y = obstacle.radius * np.sin(u)*np.sin(v) + obstacle.center.item(1)
    z = obstacle.radius * np.cos(v) + obstacle.center.item(2)
    ax.plot_surface(x, y, z, color="r")

def plot_3D_obstacles(obstacles: list, ax):
    for i in range(len(obstacles)):
        plot_3D_obstacle(obstacles[i], ax)
