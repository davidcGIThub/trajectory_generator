#include "DerivativeEvaluator.hpp"
#include "CubicEquationSolver.hpp"
#include <iostream>
#include <stdexcept>

template <int D>
DerivativeEvaluator<D>::DerivativeEvaluator()
{

}

template <int D>
double DerivativeEvaluator<D>::find_min_velocity_of_spline(double cont_pts[], int num_control_points, double scale_factor)
{   
    double min_velocity = std::numeric_limits<double>::max();
    double velocity;
    int step = num_control_points;
    for (unsigned int i = 0; i < num_control_points-3; i++)
    {
        Eigen::Matrix<double,D,4> interval_control_points = cbind_help.array_section_to_eigen(cont_pts, num_control_points, i);
        std::array<double,2> vel_and_time = find_min_velocity_and_time(interval_control_points, scale_factor);
        velocity = vel_and_time[0];
        if (velocity < min_velocity)
        {
            min_velocity = velocity;
        }
    }
    return min_velocity;
}

template <int D>
double DerivativeEvaluator<D>::find_max_acceleration_of_spline(double cont_pts[], int num_control_points, double scale_factor)
{   
    double max_acceleration = std::numeric_limits<double>::min();
    double acceleration;
    int step = num_control_points;
    for (unsigned int i = 0; i < num_control_points-3; i++)
    {
        Eigen::Matrix<double,D,4> interval_control_points = cbind_help.array_section_to_eigen(cont_pts, num_control_points, i);
        std::array<double,2> accel_and_time = find_max_acceleration_and_time(interval_control_points, scale_factor);
        acceleration = accel_and_time[0];
        if (acceleration > max_acceleration)
        {
            max_acceleration = acceleration;
        }
    }
    return max_acceleration;
}

template <int D>
std::array<double,2> DerivativeEvaluator<D>::find_min_velocity_and_time(Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    Eigen::Matrix<double, 4,4> M = get_third_order_M_matrix();
    Eigen::Matrix<double, 4,4> J = M.transpose()*control_points.transpose()*control_points*M;
    double A = 36*J(0,0);
    double B = 12*J(0,1) + 24*J(1,0);
    double C = 8*J(1,1) + 12*J(2,0);
    double D_ = 4*J(2,1);
    std::array<double,3> roots = CubicEquationSolver::solve_equation(A, B, C, D_);
    double t0 = 0;
    double tf = 1.0;
    double time_at_min = t0;
    double min_velocity = calculate_velocity_magnitude(t0,control_points,scale_factor);
    double velocity_at_tf = calculate_velocity_magnitude(tf,control_points,scale_factor);
    if (velocity_at_tf < min_velocity)
    {
        min_velocity = velocity_at_tf;
        double time_at_min = tf;
    }
    for(int index = 0; index < 3; index++)
    {
        double root = roots[index];
        if(root > 0 && root < 1.0)
        {
            double velocity =  calculate_velocity_magnitude(root, control_points,scale_factor);
            if (velocity < min_velocity)
            {
                min_velocity = velocity;
                double time_at_min = root;
            }
        }
    }
    std::array<double,2> min_velocity_and_time = {min_velocity, time_at_min};
    return min_velocity_and_time;
}

template <int D>
std::array<double,2> DerivativeEvaluator<D>::find_max_acceleration_and_time(Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    double t0 = 0;
    double tf = 1.0;
    double time_at_max = 0;
    double max_acceleration = calculate_acceleration_magnitude(t0,control_points,scale_factor);
    double acceleration_at_tf = calculate_acceleration_magnitude(tf, control_points,scale_factor);
    if (acceleration_at_tf > max_acceleration)
    {
        max_acceleration = acceleration_at_tf;
        time_at_max = tf;
    }
    std::array<double,2> max_acceleration_and_time = {max_acceleration, time_at_max};
    return max_acceleration_and_time;
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
template class DerivativeEvaluator<2>;
template class DerivativeEvaluator<3>;