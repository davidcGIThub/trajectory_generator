import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class Waypoint:
    location: np.ndarray
    velocity: np.ndarray = None
    acceleration: np.ndarray = None
    jerk: np.ndarray = None
    side: str = None

    def checkIfDerivativesActive(self):
        return (self.checkIfVelocityActive() or self.checkIfAccelerationActive())
        
    def checkIfVelocityActive(self):
        return (self.velocity is not None)
    
    def checkIfAccelerationActive(self):
        return (self.acceleration is not None)


@dataclass
class WaypointData:
    start_waypoint: Waypoint = Waypoint(location=np.array([]))
    end_waypoint: Waypoint = Waypoint(location=np.array([]))
    intermediate_locations: np.ndarray = None

    def __post_init__(self):
        self.start_waypoint.side = "start"
        self.end_waypoint.side = "end"

    def get_waypoint_locations(self):
        point_sequence = self.start_waypoint.location
        if self.intermediate_locations is not None:
            point_sequence = np.concatenate((point_sequence, self.intermediate_locations),1)
        point_sequence = np.concatenate((point_sequence, self.end_waypoint.location),1)
        return point_sequence
    
    def get_distance(self):
        start_pos = self.start_waypoint.location
        end_pos = self.end_waypoint.location
        distance = np.linalg.norm(end_pos-start_pos)
        return distance

    def set_location_data(self, waypoint_sequence, set_vel = True):
        self.start_waypoint.location = waypoint_sequence[:,0][:,None]
        self.end_waypoint.location = waypoint_sequence[:,-1][:,None]
        if set_vel:
            self.start_waypoint.velocity = (waypoint_sequence[:,1] - waypoint_sequence[:,0])[:,None]
            self.end_waypoint.velocity = (waypoint_sequence[:,-1] - waypoint_sequence[:,-2])[:,None]
        if np.shape(waypoint_sequence)[1] > 2:
            self.intermediate_locations = waypoint_sequence[:,1:-1]

    def get_num_intermediate_waypoints(self):
        if self.intermediate_locations is not None:
            return np.shape(self.intermediate_locations)[1]
        else:
            return 0
        
def plot2D_waypoints(waypoint_data: WaypointData, ax):
    locations = waypoint_data.get_waypoint_locations()
    ax.scatter(locations[0,:],locations[1,:],color="b")
    distance = waypoint_data.get_distance()
    if waypoint_data.start_waypoint.checkIfVelocityActive():
        start_pos = waypoint_data.start_waypoint.location
        start_vel = waypoint_data.start_waypoint.velocity
        ax.quiver(start_pos.item(0), start_pos.item(1), 
                  start_vel.item(0), start_vel.item(1), color = "b")
    if waypoint_data.end_waypoint.checkIfVelocityActive():
        end_pos = waypoint_data.end_waypoint.location
        end_vel = waypoint_data.end_waypoint.velocity
        ax.quiver(end_pos.item(0), end_pos.item(1), 
                  end_vel.item(0), end_vel.item(1), color = "b")
    if waypoint_data.intermediate_locations is not None:
        ax.scatter(waypoint_data.intermediate_locations[0,:], 
                   waypoint_data.intermediate_locations[1,:],color="b")

def plot3D_waypoints(waypoint_data: WaypointData, ax):
    locations = waypoint_data.get_waypoint_locations()
    ax.scatter(locations[0,:],locations[1,:],locations[2,:],color="b")
    distance = waypoint_data.get_distance()
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