import numpy as np
from scipy.optimize import LinearConstraint
from trajectory_generation.constraint_data_structures.safe_flight_corridor import SFC_Data, SFC
from trajectory_generation.control_point_conversions.bspline_to_minvo import get_composite_bspline_to_minvo_conversion_matrix
from trajectory_generation.constraint_data_structures.constraint_function_data import ConstraintFunctionData

def create_safe_flight_corridor_constraint(sfc_data: SFC_Data, num_cont_pts, \
        num_intermediate_waypoints,num_waypoint_scalars, dimension, order):
    # create the rotation matrix.
    num_extra_spaces = 1 + num_intermediate_waypoints + num_waypoint_scalars
    num_corridors = get_num_corridors(sfc_data)
    num_minvo_cont_pts = (num_cont_pts - order)*(order+1)
    intervals_per_corridor = sfc_data.get_intervals_per_corridor()
    sfc_list = sfc_data.get_sfc_list()
    M_rot = get_composite_sfc_rotation_matrix(intervals_per_corridor, sfc_list, num_minvo_cont_pts, dimension, order)
    # create the bspline to minvo conversion matrix 
    M_minvo = get_composite_bspline_to_minvo_conversion_matrix(num_cont_pts, order)
    zero_block = np.zeros((num_minvo_cont_pts,num_cont_pts))
    zero_col = np.zeros((num_minvo_cont_pts, num_extra_spaces))
    if dimension == 2:
        M_minvo = np.block([[M_minvo, zero_block, zero_col],
                                    [zero_block, M_minvo, zero_col]])
    if dimension == 3:
        M_minvo = np.block([[M_minvo,    zero_block, zero_block, zero_col],
                            [zero_block, M_minvo   , zero_block, zero_col],
                            [zero_block, zero_block, M_minvo   , zero_col]])
    conversion_matrix = M_rot @ M_minvo
    #create bounds
    lower_bounds = np.zeros((dimension, num_minvo_cont_pts))
    upper_bounds = np.zeros((dimension, num_minvo_cont_pts))
    constraints_key = np.empty((dimension, num_minvo_cont_pts)).astype(str)
    index = 0
    for corridor_index in range(num_corridors):
        num_intervals = intervals_per_corridor[corridor_index]
        sfc = sfc_list[corridor_index]
        lower_bound, upper_bound = sfc.getRotatedBounds()
        num_points = num_intervals*(order+1)
        lower_bounds[:,index:index+num_points] = lower_bound
        upper_bounds[:,index:index+num_points] = upper_bound
        constraints_key[:,index:index+num_points] = "sfc " + str(corridor_index + 1) 
        index = index+num_points
    def sfc_constraint_function(variables):
        constraints = np.dot(conversion_matrix, variables).flatten()
        return constraints
    constraints_key = constraints_key.flatten()
    lower_bounds = lower_bounds.flatten()
    upper_bounds = upper_bounds.flatten()
    constraint_class = "Safe_Flight_Corridor"
    safe_corridor_constraints = LinearConstraint(conversion_matrix, lb=lower_bounds, ub=upper_bounds)
    sfc_constraint_function_data = ConstraintFunctionData(sfc_constraint_function, lower_bounds, upper_bounds, constraints_key, constraint_class)
    return safe_corridor_constraints, sfc_constraint_function_data

def get_composite_sfc_rotation_matrix(intervals_per_corridor, sfcs, num_minvo_cont_pts, dimension, order):
    num_corridors = len(intervals_per_corridor)
    M_len = num_minvo_cont_pts*dimension
    M_rot = np.zeros((M_len, M_len))
    num_cont_pts_per_interval = order + 1
    interval_count = 0
    dim_step = num_minvo_cont_pts
    for corridor_index in range(num_corridors):
        rotation = sfcs[corridor_index].rotation.T
        num_intervals = intervals_per_corridor[corridor_index]
        for interval_index in range(num_intervals):
            for cont_pt_index in range(num_cont_pts_per_interval):
                index = interval_count*num_cont_pts_per_interval+cont_pt_index
                M_rot[index, index] = rotation[0,0]
                M_rot[index, index + dim_step] = rotation[0,1]
                M_rot[index + dim_step, index] = rotation[1,0]
                M_rot[index + dim_step, index + dim_step] = rotation[1,1]
                if dimension == 3:
                    M_rot[2*dim_step + index, index] = rotation[2,0]
                    M_rot[2*dim_step + index, index + dim_step] = rotation[2,1]
                    M_rot[2*dim_step + index, index + 2*dim_step] = rotation[2,2]
                    M_rot[dim_step + index, index + 2*dim_step] = rotation[1,2]
                    M_rot[index, index + 2*dim_step] = rotation[0,2]
            interval_count += 1
    return M_rot

def get_num_corridors(sfc_data:SFC_Data = None):
    if sfc_data is None:
        return int(0)
    else:
        return sfc_data.get_num_corridors()