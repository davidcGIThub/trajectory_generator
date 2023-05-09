#include "CrossTermEvaluator.hpp"
#include "CubicEquationSolver.hpp"
#include <iostream>
#include <stdexcept>

template <int D>
CrossTermEvaluator<D>::CrossTermEvaluator()
{

}

template <int D>
double CrossTermEvaluator<D>::calculate_cross_term_magnitude(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    Eigen::Matrix<double,D,1> velocity_vector = d_dt_eval.calculate_velocity_vector(t, control_points, scale_factor);
    double cross_term_magnitude = calculate_cross_term_magnitude(t, control_points, scale_factor, velocity_vector);
    return cross_term_magnitude;
}

template <int D>
double CrossTermEvaluator<D>::calculate_curvature(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    Eigen::Matrix<double,D,1> velocity_vector = d_dt_eval.calculate_velocity_vector(t, control_points, scale_factor);
    // double vel_cubed_mag = pow((velocity_vector.norm()),3);
    double vel_mag = (velocity_vector.norm());
    double cross_term_mag = calculate_cross_term_magnitude(t, control_points, scale_factor, velocity_vector);
    double curvature = 0;
    if (vel_mag < 0.00000001)
    {
        if (check_if_not_ascending_colinear(t, control_points, scale_factor))
        {
            curvature = std::numeric_limits<double>::max();
        }
        else
        {
            curvature = 0;
        }
    }
    else
    {
        curvature = cross_term_mag / pow(vel_mag,3);
    }
    return curvature;
}

template <int D>
bool CrossTermEvaluator<D>::check_if_not_ascending_colinear(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    bool isNonAscendingColinear{false};
    if (t > 0 && t < scale_factor)
    {
        double time_step = scale_factor/100;
        double t_a = t - time_step;
        double t_b = t;
        double t_c = t + time_step;
        if (t_a < 0){t_a = 0;};
        if (t_b < 0){t_b = scale_factor;};
        Eigen::Matrix<double,D,1> point_a = d_dt_eval.calculate_position_vector(t_a,control_points,scale_factor);
        Eigen::Matrix<double,D,1> point_b = d_dt_eval.calculate_position_vector(t_b,control_points,scale_factor);
        Eigen::Matrix<double,D,1> point_c = d_dt_eval.calculate_position_vector(t_c,control_points,scale_factor);
        Eigen::Matrix<double,D,1> step_1 = point_b - point_a;
        Eigen::Matrix<double,D,1> step_2 = point_c - point_b;
        if (step_1.cwiseAbs().isApprox(step_2.cwiseAbs(), 0.00000001))
        {
            if(!step_1.isApprox(step_2, 0.00000001))
            {
                isNonAscendingColinear = true;
            }
        }
    } 
    return isNonAscendingColinear;
}

template <int D>
double CrossTermEvaluator<D>::calculate_cross_term_magnitude(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor,
        Eigen::Matrix<double,D,1> &velocity_vector)
{
    Eigen::Matrix<double,D,1> acceleration_vector = d_dt_eval.calculate_acceleration_vector(t, control_points, scale_factor);
    double cross_term_magnitude = 0;
    if (D == 2)
    { 
        cross_term_magnitude = abs(velocity_vector(0)*acceleration_vector(1) - velocity_vector(1)*acceleration_vector(0));
    }
    else
    {
        double x = velocity_vector(1)*acceleration_vector(2) - velocity_vector(2)*acceleration_vector(1);
        double y = velocity_vector(2)*acceleration_vector(0) - velocity_vector(0)*acceleration_vector(2);
        double z = velocity_vector(0)*acceleration_vector(1) - velocity_vector(1)*acceleration_vector(0);
        cross_term_magnitude = sqrt(x*x + y*y + z*z);
    }
    return cross_term_magnitude;
}

//Explicit template instantiations
template class CrossTermEvaluator<2>;
template class CrossTermEvaluator<3>;