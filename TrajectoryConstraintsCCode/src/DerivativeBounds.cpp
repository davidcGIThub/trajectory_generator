#include "DerivativeBounds.hpp"
#include <iostream>
#include <stdexcept>

template <int D>
DerivativeBounds<D>::DerivativeBounds()
{

}

template <int D>
double DerivativeBounds<D>::find_min_velocity_of_spline(double cont_pts[], int num_control_points, double scale_factor)
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
double DerivativeBounds<D>::find_max_acceleration_of_spline(double cont_pts[], int num_control_points, double scale_factor)
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
std::array<double,2> DerivativeBounds<D>::find_min_velocity_and_time(Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    std::array<double,3> roots = get_velocity_roots(control_points, scale_factor);
    double t0 = 0;
    double tf = scale_factor;
    double time_at_min = t0;
    double min_velocity = d_eval.calculate_velocity_magnitude(t0,control_points,scale_factor);
    double velocity_at_tf = d_eval.calculate_velocity_magnitude(tf,control_points,scale_factor);
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
            double velocity = d_eval.calculate_velocity_magnitude(root, control_points,scale_factor);
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
std::array<double,2> DerivativeBounds<D>::find_max_velocity_and_time(Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    std::array<double,3> roots = get_velocity_roots(control_points, scale_factor);
    double t0 = 0;
    double tf = scale_factor;
    double time_at_max = t0;
    double max_velocity = d_eval.calculate_velocity_magnitude(t0,control_points,scale_factor);
    double velocity_at_tf = d_eval.calculate_velocity_magnitude(tf,control_points,scale_factor);
    if (velocity_at_tf > max_velocity)
    {
        max_velocity = velocity_at_tf;
        double time_at_max = tf;
    }
    for(int index = 0; index < 3; index++)
    {
        double root = roots[index];
        if(root > 0 && root < 1.0)
        {
            double velocity = d_eval.calculate_velocity_magnitude(root, control_points,scale_factor);
            if (velocity > max_velocity)
            {
                max_velocity = velocity;
                double time_at_max = root;
            }
        }
    }
    std::array<double,2> max_velocity_and_time = {max_velocity, time_at_max};
    return max_velocity_and_time;
}

template <int D>
std::array<double,3> DerivativeBounds<D>::get_velocity_roots(Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    Eigen::Matrix<double, 4,4> M = d_eval.get_third_order_M_matrix();
    Eigen::Matrix<double, 4,4> J = M.transpose()*control_points.transpose()*control_points*M;
    double A = 36*J(0,0);
    double B = 12*J(0,1) + 24*J(1,0);
    double C = 8*J(1,1) + 12*J(2,0);
    double D_ = 4*J(2,1);
    std::array<double,3> roots = CubicEquationSolver::solve_equation(A, B, C, D_);
    roots[0] = roots[0]*scale_factor;
    roots[1] = roots[1]*scale_factor;
    roots[2] = roots[2]*scale_factor;
    return roots;
}



template <int D>
std::array<double,2> DerivativeBounds<D>::find_max_acceleration_and_time(Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    double t0 = 0;
    double tf = scale_factor;
    double time_at_max = 0;
    double max_acceleration = d_eval.calculate_acceleration_magnitude(t0,control_points,scale_factor);
    double acceleration_at_tf = d_eval.calculate_acceleration_magnitude(tf, control_points,scale_factor);
    if (acceleration_at_tf > max_acceleration)
    {
        max_acceleration = acceleration_at_tf;
        time_at_max = tf;
    }
    std::array<double,2> max_acceleration_and_time = {max_acceleration, time_at_max};
    return max_acceleration_and_time;
}

template <int D>
double DerivativeBounds<D>::find_max_velocity_magnitude_in_single_dimension(Eigen::Matrix<double,D,4> &control_points, 
    double &scale_factor, unsigned int &dimension)
{
    double p0 = control_points.coeff(dimension-1,0);
    double p1 = control_points.coeff(dimension-1,1);
    double p2 = control_points.coeff(dimension-1,2);
    double p3 = control_points.coeff(dimension-1,3);
    double a_term = -6*(p0/6 - p1/2 + p2/2 - p3/6)/(scale_factor*scale_factor*scale_factor);
    double b_term = 2*(p0/2 - p1 + p2/2)/(scale_factor*scale_factor);
    double t = CubicEquationSolver::solve_linear_equation(a_term, b_term);
    Eigen::Matrix<double,1,4> control_points_single_dimension;
    control_points_single_dimension << p0, p1, p2, p3;
    double t0 = 0;
    double v_max = abs(d_eval_single.calculate_velocity_vector(t0,control_points_single_dimension,scale_factor).coeff(0));
    double v_final = abs(d_eval_single.calculate_velocity_vector(scale_factor,control_points_single_dimension,scale_factor).coeff(0));
    if (v_final > v_max)
    {
        v_max = v_final;
    }
    if (t > 0 && t < scale_factor)
    {
        double v_t = abs(d_eval_single.calculate_velocity_vector(t,control_points_single_dimension,scale_factor).coeff(0));
        if (v_t > v_max)
        {
            v_max = v_t;
        }
    }
    return v_max;
}

//Explicit template instantiations
template class DerivativeBounds<2>;
template class DerivativeBounds<3>;