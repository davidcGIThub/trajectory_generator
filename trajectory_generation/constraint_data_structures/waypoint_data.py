import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class Waypoint:
    location: np.ndarray
    direction: np.ndarray = None
    velocity: np.ndarray = None
    acceleration: np.ndarray = None
    jerk: np.ndarray = None
    dimension: int = None
    side: str = None
    is_target: bool = None

    def checkIfDerivativesActive(self):
        if self.checkIfAccelerationActive() or self.checkIfDirectionActive():
            return True
        elif self.checkIfVelocityActive and not self.checkIfZeroVel():
            return True
        else:
            return False
    
    def checkIfDirectionActive(self):
        return (self.direction is not None)
        
    def checkIfVelocityActive(self):
        return (self.velocity is not None)
    
    def checkIfAccelerationActive(self):
        return (self.acceleration is not None)
    
    def checkIfZeroVel(self):
        if self.velocity is not None and np.linalg.norm(self.velocity) <= 0:
            return True
        else:
            return False
    
    def __post_init__(self):
        self.dimension = len(self.location.flatten())
        if self.velocity is not None and len(self.velocity.flatten()) != self.dimension:
            raise Exception("Error: Velocity is not the same dimension as location")
        if self.acceleration is not None and len(self.acceleration.flatten()) != self.dimension:
            raise Exception("Error: Acceleration is not the same dimension as location")
        if self.jerk is not None and len(self.jerk.flatten()) != self.dimension:
            raise Exception("Error: Jerk is not the same dimension as location")
        if self.direction is not None:
            if self.velocity is not None:
                if np.linalg.norm(self.velocity) > 0 :
                    self.direction = None
                    print("Using velocity constraint - cannot use both velocity and direction constraint")

class WaypointData:

    def __init__(self, waypoint_sequence: 'list[Waypoint]'):
        if len(waypoint_sequence) < 2:
            raise Exception("Waypoint sequence must have at least two waypoints")
        self.start_waypoint = None
        self.end_waypoint = None
        self.dimension = None
        self.intermediate_locations = None
        self.intermediate_velocities = None
        self.__initialize_terminal_waypoints(waypoint_sequence)
        self.__initialize_intermediate_waypoints(waypoint_sequence)

    def __initialize_terminal_waypoints(self, waypoint_sequence: 'list[Waypoint]'):
        self.start_waypoint = waypoint_sequence[0]
        self.end_waypoint = waypoint_sequence[-1]
        if self.start_waypoint.dimension != self.end_waypoint.dimension:
            raise Exception("Waypoint dimensions do not match")
        self.dimension = self.start_waypoint.dimension
        self.start_waypoint.side = "start"
        self.end_waypoint.side = "end"

    def __initialize_intermediate_waypoints(self, waypoint_sequence: 'list[Waypoint]'):
        num_intermediate_waypoints = len(waypoint_sequence) - 2
        velocities_on = False
        if num_intermediate_waypoints > 0:
            self.intermediate_locations = np.zeros((self.dimension, num_intermediate_waypoints))
            self.intermediate_velocities = np.zeros((self.dimension, num_intermediate_waypoints))
            for i in range(num_intermediate_waypoints):
                waypoint = waypoint_sequence[i+1]
                if waypoint.dimension != self.dimension:
                    raise Exception("Waypoint dimensions do not match")
                self.intermediate_locations[:,i] = waypoint.location.flatten()
                if waypoint.velocity is not None:
                    velocities_on = True
                    self.intermediate_velocities[:,i] = waypoint.velocity.flatten()
        if velocities_on == False:
            self.intermediate_velocities = None

    def get_waypoint_locations(self):
        point_sequence = self.start_waypoint.location
        if self.intermediate_locations is not None:
            point_sequence = np.concatenate((point_sequence, self.intermediate_locations),1)
        point_sequence = np.concatenate((point_sequence, self.end_waypoint.location),1)
        return point_sequence

    def get_num_intermediate_waypoints(self):
        if self.intermediate_locations is not None:
            return np.shape(self.intermediate_locations)[1]
        else:
            return 0
        
    def get_num_waypoint_scalars(self):
        num_waypoint_scalars = 0
        if self.start_waypoint.direction is not None:
            num_waypoint_scalars += 1
        if self.end_waypoint.direction is not None:
            num_waypoint_scalars += 1
        return num_waypoint_scalars
        
def plot2D_waypoints(waypoint_data: WaypointData, ax):
    locations = waypoint_data.get_waypoint_locations()
    ax.scatter(locations[0,:],locations[1,:],facecolors='none', edgecolors="r", label="waypoint constraints")
    if waypoint_data.start_waypoint.checkIfVelocityActive():
        start_pos = waypoint_data.start_waypoint.location
        if not waypoint_data.start_waypoint.checkIfZeroVel():
            start_dir = waypoint_data.start_waypoint.velocity
            ax.quiver(start_pos.item(0), start_pos.item(1), 
                    start_dir.item(0), start_dir.item(1), color = "r")
    if waypoint_data.end_waypoint.checkIfVelocityActive():
        end_pos = waypoint_data.end_waypoint.location
        if not waypoint_data.end_waypoint.checkIfZeroVel():
            end_dir = waypoint_data.end_waypoint.velocity
            ax.quiver(end_pos.item(0), end_pos.item(1), 
                    end_dir.item(0), end_dir.item(1), color = "r")
    if waypoint_data.intermediate_locations is not None:
        ax.scatter(waypoint_data.intermediate_locations[0,:], 
                   waypoint_data.intermediate_locations[1,:],facecolors='none', edgecolors="r")

def plot3D_waypoints(waypoint_data: WaypointData, ax):
    locations = waypoint_data.get_waypoint_locations()
    ax.scatter(locations[0,:],locations[1,:],locations[2,:],color="b")
    distance = get_distance_between_start_and_end_waypoint(waypoint_data.start_waypoint, waypoint_data.end_waypoint)
    if waypoint_data.start_waypoint.checkIfVelocityActive():
        start_pos = waypoint_data.start_waypoint.location
        start_vel = waypoint_data.start_waypoint.velocity
        ax.quiver(start_pos.item(0), start_pos.item(1), start_pos.item(2), 
                  start_vel.item(0), start_vel.item(1), start_vel.item(2), 
                  length=distance/10, normalize=True)
    if waypoint_data.end_waypoint.checkIfVelocityActive():
        end_pos = waypoint_data.end_waypoint.location
        end_vel = waypoint_data.end_waypoint.velocity
        ax.quiver(end_pos.item(0), end_pos.item(1), end_pos.item(2), 
                  end_vel.item(0), end_vel.item(1), end_vel.item(2), 
                  length=distance/10, normalize=True)
    if waypoint_data.intermediate_locations is not None:
        ax.scatter(waypoint_data.intermediate_locations[0,:], 
                   waypoint_data.intermediate_locations[1,:],
                   waypoint_data.intermediate_locations[2,:],color="b")
        
def get_distance_between_start_and_end_waypoint(start_waypoint, end_waypoint):
    start_pos = start_waypoint.location
    end_pos = end_waypoint.location
    distance = np.linalg.norm(end_pos-start_pos)
    return distance