#include "DerivativeEvaluator.hpp"
#include "CubicEquationSolver.hpp"
#include <iostream>
#include <stdexcept>

template <int D>
DerivativeEvaluator<D>::DerivativeEvaluator()
{

}

template <int D>
Eigen::Matrix<double,D,1> DerivativeEvaluator<D>::calculate_position_vector(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    Eigen::Vector4d T = get_third_order_T_vector(t, scale_factor);
    Eigen::Matrix<double,4,4> M = get_third_order_M_matrix();
    Eigen::Matrix<double,D,1> position_vector = control_points*M*T;
    return position_vector;
}

template <int D>
Eigen::Matrix<double,D,1> DerivativeEvaluator<D>::calculate_velocity_vector(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    Eigen::Vector4d dT = get_third_order_T_derivative_vector(t, scale_factor);
    Eigen::Matrix<double,4,4> M = get_third_order_M_matrix();
    Eigen::Matrix<double,D,1> velocity_vector = control_points*M*dT;
    return velocity_vector;
}

template <int D>
double DerivativeEvaluator<D>::calculate_velocity_magnitude(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    Eigen::Matrix<double,D,1> velocity_vector = calculate_velocity_vector(t, control_points, scale_factor);
    double velocity_magnitude = velocity_vector.norm();
    return velocity_magnitude;
}

template <int D>
Eigen::Matrix<double,D,1> DerivativeEvaluator<D>::calculate_acceleration_vector(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    Eigen::Vector4d ddT = get_third_order_T_second_derivative_vector(t, scale_factor);
    Eigen::Matrix<double, 4,4> M = get_third_order_M_matrix();
    Eigen::Matrix<double,D,1> acceleration_vector = control_points*M*ddT;
    return acceleration_vector;
}

template <int D>
double DerivativeEvaluator<D>::calculate_acceleration_magnitude(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    Eigen::Matrix<double,D,1> acceleration_vector = calculate_acceleration_vector(t, control_points, scale_factor);
    double acceleration_magnitude = acceleration_vector.norm();
    return acceleration_magnitude;
}

template <int D>
Eigen::Matrix<double, 4,4> DerivativeEvaluator<D>::get_third_order_M_matrix()
{
    Eigen::Matrix<double, 4,4> M;
    M <<  -1/6.0 ,   1/2.0 , -1/2.0 , 1/6.0,
            1/2.0 ,      -1 ,  0     , 2/3.0,
            -1/2.0 ,   1/2.0 ,  1/2.0 , 1/6.0,
            1/6.0 ,  0      ,  0     , 0;
    return M;
}

template <int D>
Eigen::Vector4d DerivativeEvaluator<D>::get_third_order_T_vector(double &t, double &scale_factor)
{
    Eigen::Vector4d t_vector;
    if (t < 0 || t/scale_factor > 1)
    {
        throw std::invalid_argument("t value should be between 0 and 1");
    }
    else
    {
        double alpha = scale_factor;
        t_vector << t*t*t/(alpha*alpha*alpha) , t*t/(alpha*alpha) , t/alpha , 1;
    }
    return t_vector;
}

template <int D>
Eigen::Vector4d DerivativeEvaluator<D>::get_third_order_T_derivative_vector(double &t, double &scale_factor)
{
    double alpha = scale_factor;
    Eigen::Vector4d t_vector;
    if (scale_factor == 1.0)
    {
        if (t < 0 || t > 1)
        {
            throw std::invalid_argument("t value should be between 0 and 1");
        }
        else
        {
            t_vector << 3*t*t , 2*t , 1, 0;
        }
    }
    else
    {
        t_vector << 3*t*t/(alpha*alpha*alpha) , 2*t/(alpha*alpha) , 1/alpha , 0;
    }
    return t_vector;
}


template <int D>
Eigen::Vector4d DerivativeEvaluator<D>::get_third_order_T_second_derivative_vector(double &t,  double &scale_factor)
{
    double alpha = scale_factor;
    Eigen::Vector4d t_vector;
    if (scale_factor == 1)
    {
        if (t < 0 || t > 1)
        {
            throw std::invalid_argument("t value should be between 0 and 1");
        }
        else
        {
            t_vector << 6*t , 2 , 0 , 0;
        }
    }
    else
    {
        t_vector << 6*t/(alpha*alpha*alpha ) , 2/(alpha*alpha) , 0 , 0;
    }
    return t_vector;
}

//Explicit template instantiations
template class DerivativeEvaluator<1>;
template class DerivativeEvaluator<2>;
template class DerivativeEvaluator<3>;