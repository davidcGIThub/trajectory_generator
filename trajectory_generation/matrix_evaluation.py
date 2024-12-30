import numpy as np
import scipy


def matrix_bspline_evaluation_for_dataset(order, control_points, num_points):
    """
    This function evaluates the B spline for a given time data-set
    """
    #initialize variables
    dimension = get_dimension(control_points)
    number_of_control_points = count_number_of_control_points(control_points)
    num_intervals = number_of_control_points - order
    #create steps matrix
    time_data = np.linspace(0,num_intervals,num_points)
    # Find M matrix
    M = get_M_matrix(order)
    #Evaluate spline data
    spline_data = np.zeros((dimension,num_points))
    marker = 0
    for i in range(num_intervals):
        P = control_points[:,i:i+order+1]
        if i == num_intervals - 1:
            steps_array = time_data[(time_data >= i) & (time_data <= i+1)] - i
        else:
            steps_array = time_data[(time_data >= i) & (time_data < i+1)] - i
        num_point_interval = len(steps_array)
        L = np.ones((order+1,num_point_interval))
        for i in range(order+1):
            L[i,:] = steps_array**(order-i)
        spline_data_over_interval = np.dot(np.dot(P,M),L)
        spline_data[:,marker:marker+num_point_interval] = spline_data_over_interval
        marker = marker + num_point_interval
    return spline_data

def matrix_bspline_evaluation_for_timedataset(order, control_points, time_data, scale_factor):
    """
    This function evaluates the B spline for a given time data-set
    """
    #initialize variables
    dimension = get_dimension(control_points)
    number_of_control_points = count_number_of_control_points(control_points)
    num_intervals = number_of_control_points - order
    #create steps matrix
    # Find M matrix
    M = get_M_matrix(order)
    #Evaluate spline data
    num_points = len(time_data)
    time_array = time_data/scale_factor
    spline_data = np.zeros((dimension,num_points))
    marker = 0
    for i in range(num_intervals):
        P = control_points[:,i:i+order+1]
        if i == num_intervals - 1:
            steps_array = time_array[(time_array >= i) & (time_array <= i+1)] - i
        else:
            steps_array = time_array[(time_array >= i) & (time_array < i+1)] - i
        num_point_interval = len(steps_array)
        L = np.ones((order+1,num_point_interval))
        for i in range(order+1):
            L[i,:] = steps_array**(order-i)
        spline_data_over_interval = np.dot(np.dot(P,M),L)
        spline_data[:,marker:marker+num_point_interval] = spline_data_over_interval
        marker = marker + num_point_interval
    return spline_data

def matrix_bspline_evaluation_for_discrete_steps(order, control_points, start_time, starting_offset, dt, scale_factor):
    """
    This function evaluates the B spline for a given time data-set
    """
    #initialize variables
    dimension = get_dimension(control_points)
    number_of_control_points = count_number_of_control_points(control_points)
    num_intervals = number_of_control_points - order
    duration = scale_factor*num_intervals
    num_samples = int((duration-starting_offset)/dt) + 1
    last_time_sample = (num_samples-1)*dt + starting_offset
    remainder_time = duration - last_time_sample
    #create steps matrix
    time_data = np.linspace(starting_offset,last_time_sample, num_samples)
    time_array = time_data / scale_factor
    # Find M matrix
    M = get_M_matrix(order)
    #Evaluate spline data
    spline_data = np.zeros((dimension,num_samples))
    marker = 0
    for i in range(num_intervals):
        P = control_points[:,i:i+order+1]
        if i == num_intervals - 1:
            steps_array = time_array[(time_array >= i) & (time_array <= i+1)] - i
        else:
            steps_array = time_array[(time_array >= i) & (time_array < i+1)] - i
        num_point_interval = len(steps_array)
        L = np.ones((order+1,num_point_interval))
        for i in range(order+1):
            L[i,:] = steps_array**(order-i)
        spline_data_over_interval = np.dot(np.dot(P,M),L)
        spline_data[:,marker:marker+num_point_interval] = spline_data_over_interval
        marker = marker + num_point_interval
    time_data = time_data + start_time
    spline_end_time = duration + start_time
    return spline_data, time_data, remainder_time, spline_end_time


def matrix_bspline_derivative_evaluation_for_dataset(order, derivative_order, scale_factor, control_points, num_points):
    """
    This function evaluates the B spline for a given time data-set
    """
    # Initialize variables
    dimension = get_dimension(control_points)
    number_of_control_points = count_number_of_control_points(control_points)
    num_intervals = number_of_control_points - order
    #create steps matrix
    time_data = np.linspace(0,num_intervals,num_points)
    # Find M matrix
    M = get_M_matrix(order)
    K = __create_k_matrix(order,derivative_order,scale_factor)
    # Evaluate Spline data
    marker = 0
    spline_derivative_data = np.zeros((dimension,num_points))
    for i in range(num_intervals):
        P = control_points[:,i:i+order+1]
        # Find M matrix if clamped
        if i == num_intervals - 1:
            steps_array = time_data[(time_data >= i) & (time_data <= i+1)] - i
        else:
            steps_array = time_data[(time_data >= i) & (time_data < i+1)] - i
        num_point_interval = len(steps_array)
        L_r = np.zeros((order+1,num_point_interval))
        for i in range(order-derivative_order+1):
            L_r[i,:] = steps_array**(order-derivative_order-i)
        spline_derivative_data_over_interval = np.dot(np.dot(P,M),np.dot(K,L_r))
        spline_derivative_data[:,marker:marker+num_point_interval] = spline_derivative_data_over_interval
        marker = marker + num_point_interval
    return spline_derivative_data

def matrix_bspline_derivative_evaluation_for_discrete_steps(order, derivative_order, scale_factor, control_points, start_time, starting_offset, dt):
    """
    This function evaluates the B spline for a given time data-set
    """
    # Initialize variables
    dimension = get_dimension(control_points)
    number_of_control_points = count_number_of_control_points(control_points)
    num_intervals = number_of_control_points - order
    #create steps matrix
    duration = scale_factor*num_intervals
    num_samples = int((duration-starting_offset)/dt) + 1
    last_time_sample = (num_samples-1)*dt + starting_offset
    remainder_time = duration - last_time_sample
    time_data = np.linspace(starting_offset,last_time_sample, num_samples)
    time_array = time_data/scale_factor
    # Find M matrix
    M = get_M_matrix(order)
    K = __create_k_matrix(order,derivative_order,scale_factor)
    # Evaluate Spline data
    marker = 0
    spline_derivative_data = np.zeros((dimension,num_samples))
    for i in range(num_intervals):
        P = control_points[:,i:i+order+1]
        # Find M matrix if clamped
        if i == num_intervals - 1:
            steps_array = time_array[(time_array >= i) & (time_array <= i+1)] - i
        else:
            steps_array = time_array[(time_array >= i) & (time_array < i+1)] - i
        num_point_interval = len(steps_array)
        L_r = np.zeros((order+1,num_point_interval))
        for i in range(order-derivative_order+1):
            L_r[i,:] = steps_array**(order-derivative_order-i)
        spline_derivative_data_over_interval = np.dot(np.dot(P,M),np.dot(K,L_r))
        spline_derivative_data[:,marker:marker+num_point_interval] = spline_derivative_data_over_interval
        marker = marker + num_point_interval
    time_data = time_data + start_time
    spline_end_time = duration + start_time
    return spline_derivative_data, time_data, remainder_time, spline_end_time

def __create_k_matrix(order,derivative_order,scale_factor):
    K = np.zeros((order+1,order+1))
    for i in range(order-derivative_order+1):
        K[i,i] = scipy.special.factorial(order-i)/scipy.special.factorial(order-derivative_order-i)
    K = K/scale_factor**(derivative_order)
    return K


def evaluate_point_on_interval(control_points, t, tj, scale_factor):
    order = np.shape(control_points)[1] - 1
    M = get_M_matrix(order)
    T = get_T_vector(order, t, tj, scale_factor)
    point = control_points @ M @ T
    return point

def evaluate_point_derivative_on_interval(control_points, t, tj, scale_factor,rth_derivative):
    order = np.shape(control_points)[1] - 1
    M = get_M_matrix(order)
    dT = get_T_derivative_vector(order,t,tj,rth_derivative,scale_factor)
    point = control_points @ M @ dT
    return point

def get_M_matrix(order):
    if order > 5:
        print("Error: Cannot compute higher than 5th order matrix evaluation")
        return None
    if order == 0:
        return 1
    if order == 1:
        M = __get_1_order_matrix()
    if order == 2:
        M = __get_2_order_matrix()
    elif order == 3:
        M = __get_3_order_matrix()
    elif order == 4:
        M = __get_4_order_matrix()
    elif order == 5:
        M = __get_5_order_matrix()
    else:
        raise Exception("Cannot return M matrix for spline of order " , order)
    return M

def get_T_derivative_vector(order,t,tj,rth_derivative,scale_factor):
    T = np.zeros((order+1,1))
    t_tj = t-tj
    for i in range(order-rth_derivative+1):
        T[i,0] = (t_tj**(order-rth_derivative-i))/(scale_factor**(order-i)) * scipy.special.factorial(order-i)/scipy.special.factorial(order-i-rth_derivative)
    return T

def get_T_vector(order,t,tj,scale_factor):
    T = np.ones((order+1,1))
    t_tj = t-tj
    for i in range(order+1):
        if i > order:
            T[i,0] = 0
        else:
            T[i,0] = (t_tj/scale_factor)**(order-i)
    return T

def __get_1_order_matrix():
    M = np.array([[-1,1],
                    [1,0]])
    return M

def __get_2_order_matrix():
    M = .5*np.array([[1,-2,1],
                        [-2,2,1],
                        [1,0,0]])
    return M

def __get_3_order_matrix():
    M = np.array([[-2 ,  6 , -6 , 2],
                    [ 6 , -12 ,  0 , 8],
                    [-6 ,  6 ,  6 , 2],
                    [ 2 ,  0 ,  0 , 0]])/12
    return M

def __get_4_order_matrix():
    M = np.array([[ 1 , -4  ,  6 , -4  , 1],
                    [-4 ,  12 , -6 , -12 , 11],
                    [ 6 , -12 , -6 ,  12 , 11],
                    [-4 ,  4  ,  6 ,  4  , 1],
                    [ 1 ,  0  ,  0 ,  0  , 0]])/24
    return M

def __get_5_order_matrix():
    M = np.array([[-1  ,  5  , -10 ,  10 , -5  , 1],
                    [ 5  , -20 ,  20 ,  20 , -50 , 26],
                    [-10 ,  30 ,  0  , -60 ,  0  , 66],
                    [ 10 , -20 , -20 ,  20 ,  50 , 26],
                    [-5  ,  5  ,  10 ,  10 ,  5  , 1 ],
                    [ 1  ,  0  ,  0  ,  0  ,  0  , 0]])/120
    return M


def get_dimension(control_points):
    if control_points.ndim == 1:
        dimension = 1
    else:
        dimension = len(control_points)
    return dimension

def count_number_of_control_points(control_points):
    if control_points.ndim == 1:
        number_of_control_points = len(control_points)
    else:
        number_of_control_points = len(control_points[0])
    return number_of_control_points
