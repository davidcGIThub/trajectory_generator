import numpy as np

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
    return M

def get_T_derivative_vector(order,t,tj,rth_derivative,scale_factor):
    T = np.zeros((order+1,1))
    t_tj = t-tj
    for i in range(order-rth_derivative+1):
        T[i,0] = (t_tj**(order-rth_derivative-i))/(scale_factor**(order-i)) * np.math.factorial(order-i)/np.math.factorial(order-i-rth_derivative)
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