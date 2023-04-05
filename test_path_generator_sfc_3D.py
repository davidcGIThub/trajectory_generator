import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from bsplinegenerator.bsplines import BsplineEvaluation
from path_generation.path_generator import PathGenerator
from path_generation.safe_flight_corridor import SFC_3D, plot_3D_sfcs, get3DRotationAndTranslationFromPoints
import time

point_1 = np.array([[3],[4],[0]])
point_2 = np.array([[7],[10],[3]])
point_3 = np.array([[14],[7],[7]])
point_4 = np.array([[20],[14],[4]])
points = np.concatenate((point_1,point_2,point_3,point_4),1)
dimension = np.shape(point_1)[0]
R1, T1, min_len_1 = get3DRotationAndTranslationFromPoints(point_1, point_2)
R2, T2, min_len_2 = get3DRotationAndTranslationFromPoints(point_2, point_3)
R3, T3, min_len_3 = get3DRotationAndTranslationFromPoints(point_3, point_4)
sfc_1 = SFC_3D(np.array([[min_len_1+3],[2],[3]]), T1, R1)
sfc_2 = SFC_3D(np.array([[min_len_2 + 2],[3],[4]]), T2, R2)
sfc_3 = SFC_3D(np.array([[min_len_3+3],[2],[2]]), T3, R3)
sfcs = (sfc_1, sfc_2, sfc_3)

waypoint_directions = np.concatenate((point_2-point_1, point_4-point_3),1)
waypoint_curvatures = None #np.array([0,0])
point_sequence = np.concatenate((point_1,point_2,point_3,point_4),axis=1)
dimension = np.shape(point_sequence)[0]
max_curvature = 0.5
order = 3
initial_control_points = None


path_gen = PathGenerator(dimension)
start_time = time.time()
control_points = path_gen.generate_path(point_sequence, waypoint_directions, waypoint_curvatures, 
                                        max_curvature, sfcs)
end_time = time.time()
print("computation time: " , end_time - start_time)
print("control_points: " , control_points)
spline_start_time = 0
scale_factor = 1
bspline = BsplineEvaluation(control_points, order, spline_start_time, scale_factor, False)
number_data_points = 10000
spline_data, time_data = bspline.get_spline_data(number_data_points)
curvature_data, time_data = bspline.get_spline_curvature_data(1000)
# acceleration_data, time_data = bspline.get_spline_derivative_data(1000,2)
minvo_cps = bspline.get_minvo_control_points()
waypoints = np.concatenate((point_sequence[:,0][:,None], point_sequence[:,-1][:,None]),1)

# print("sfcs: " , sfcs)
plt.figure()
ax = plt.axes(projection='3d')
plot_3D_sfcs(sfcs,ax)
ax.plot(spline_data[0,:], spline_data[1,:],spline_data[2,:])
ax.scatter(waypoints[0,:],waypoints[1,:],waypoints[2,:])
# plt.scatter(minvo_cps[0,:],minvo_cps[1,:])
plt.show()

plt.figure()
plt.plot(time_data, curvature_data)
plt.show()

print("start curvature: " , curvature_data[0])
print("end curvature: " , curvature_data[-1])


