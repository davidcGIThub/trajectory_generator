import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from bsplinegenerator.bsplines import BsplineEvaluation
from path_generation.path_generator import PathGenerator
from path_generation.safe_flight_corridor import SFC_2D, plot_2D_sfcs, get2DRotationAndTranslationFromPoints
import time

# sfc_1 = SFC(np.array([[4],[6]]), np.array([[3],[5]]), np.eye(2))
# sfc_2 = SFC(np.array([[10],[2]]), np.array([[6],[7]]), np.eye(2))
# sfc_3 = SFC(np.array([[4],[8]]), np.array([[9],[10]]), np.eye(2))
# sfc_4 = SFC(np.array([[16],[4]]), np.array([[15],[12]]), np.eye(2))
# sfcs = (sfc_1, sfc_2)
# sfcs = (sfc_1, sfc_2, sfc_3)
# sfcs = (sfc_1, sfc_2, sfc_3, sfc_4)
# sfcs = None
# waypoint_directions = np.array([[0,0],[1,1]]) # 2
point_1 = np.array([[3],[4]])
point_2 = np.array([[7],[10]])
point_3 = np.array([[14],[7]])
point_4 = np.array([[20],[14]])
# point_4 = np.array([[25],[20]])
# point_4 = np.array([[18],[10]])
point_sequence = np.concatenate((point_1,point_2,point_3,point_4),axis=1)
waypoints = np.concatenate((point_sequence[:,0][:,None], point_sequence[:,-1][:,None]),1)
# waypoints = np.array([[2,16],[3,13]])
waypoint_directions = np.array([[0,0],[0,0]]) # 2
waypoint_curvatures = np.array([-1,0.1])
# waypoint_curvatures = None
# waypoint_directions = None
R1, T1, min_len_1 = get2DRotationAndTranslationFromPoints(point_1, point_2)
R2, T2, min_len_2 = get2DRotationAndTranslationFromPoints(point_2, point_3)
R3, T3, min_len_3 = get2DRotationAndTranslationFromPoints(point_3, point_4)
sfc_1 = SFC_2D(np.array([[min_len_1+5],[3]]), T1, R1)
sfc_2 = SFC_2D(np.array([[min_len_2+3],[4]]), T2, R2)
sfc_3 = SFC_2D(np.array([[min_len_3+5],[5]]), T3, R3)
sfcs = (sfc_1, sfc_2, sfc_3)
max_curvature = 0.5
order = 3
# waypoints = np.array([[2,9],[3,7.5]])
# waypoints = np.array([[2,8],[3,13]])
dimension = np.shape(point_sequence)[0]

# sfcs = None
# point_sequence = waypoints
# waypoint_curvatures = None
# waypoint_directions = None
# max_curvature = None

initial_control_points = None

curvature_method = "roots_numerator_and_denominator"
# curvature_method = "constrain_max_acceleration_and_min_velocity"


path_gen = PathGenerator(dimension)
start_time = time.time()
control_points = path_gen.generate_path(point_sequence, waypoint_directions, waypoint_curvatures, 
                                        max_curvature, sfcs)
# generate_path(point_sequence, waypoint_directions, waypoint_accelerations, max_curvature,
#                 sfcs)
print("control_points: " , control_points)
end_time = time.time()
print("computation time:" , end_time - start_time)
spline_start_time = 0
scale_factor = 1
bspline = BsplineEvaluation(control_points, order, spline_start_time, scale_factor, False)
number_data_points = 10000
spline_data, time_data = bspline.get_spline_data(number_data_points)

spline_data, time_data = bspline.get_spline_data(1000)
curvature_data, time_data = bspline.get_spline_curvature_data(1000)
# acceleration_data, time_data = bspline.get_spline_derivative_data(1000,2)
minvo_cps = bspline.get_minvo_control_points()

# print("sfcs: " , sfcs)
plt.figure()
plot_2D_sfcs(sfcs)
plt.plot(spline_data[0,:], spline_data[1,:])
plt.scatter(waypoints[0,:],waypoints[1,:])
# plt.scatter(minvo_cps[0,:],minvo_cps[1,:])
plt.show()

plt.figure()
plt.plot(time_data, curvature_data)
plt.show()

print("start curvature: " , curvature_data[0])
print("end curvature: " , curvature_data[-1])


