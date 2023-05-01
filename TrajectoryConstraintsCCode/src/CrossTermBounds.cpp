#include "CrossTermBounds.hpp"
#include "CubicEquationSolver.hpp"
#include <iostream>
#include <stdexcept>

template <int D>
CrossTermBounds<D>::CrossTermBounds()
{

}

template <int D>
double CrossTermBounds<D>::get_spline_curvature_bound(double cont_pts[], int num_control_points)
{
    double max_curvature{0};
    double curvature;
    for (unsigned int i = 0; i < num_control_points-3; i++)
    {
        Eigen::Matrix<double,D,4> interval_control_points = cbind_help.array_section_to_eigen(cont_pts, num_control_points, i);
        curvature = evaluate_interval_curvature_bound(interval_control_points);
        if (curvature > max_curvature)
        {
            max_curvature = curvature;
        }
    }
    return max_curvature;
}

template <int D>
double CrossTermBounds<D>::get_spline_angular_rate_bound(double cont_pts[], int num_control_points, double scale_factor)
{
    double max_angular_rate{0};
    double angular_rate;
    for (unsigned int i = 0; i < num_control_points-3; i++)
    {
        Eigen::Matrix<double,D,4> interval_control_points = cbind_help.array_section_to_eigen(cont_pts, num_control_points, i);
        angular_rate = evaluate_interval_angular_rate_bound(interval_control_points, scale_factor);
        if (angular_rate > max_angular_rate)
        {
            max_angular_rate = angular_rate;
        }
    }
    return max_angular_rate;
}

template <int D>
double CrossTermBounds<D>::get_spline_centripetal_acceleration_bound(double cont_pts[], int num_control_points, double scale_factor)
{
    double max_centripetal_acceleration{0};
    double centripetal_acceleration;
    for (unsigned int i = 0; i < num_control_points-3; i++)
    {
        Eigen::Matrix<double,D,4> interval_control_points = cbind_help.array_section_to_eigen(cont_pts, num_control_points, i);
        centripetal_acceleration = evaluate_interval_centripetal_acceleration_bound(interval_control_points, scale_factor);
        if (centripetal_acceleration > max_centripetal_acceleration)
        {
            max_centripetal_acceleration = centripetal_acceleration;
        }
    }
    return max_centripetal_acceleration;
}

template <int D>
double CrossTermBounds<D>::evaluate_interval_curvature_bound(Eigen::Matrix<double,D,4> &control_points)
{
    double scale_factor = 1;
    DerivativeBoundsData d_data = get_derivative_bound_data(control_points, scale_factor);
    double curvature_bound;
    double curvature;
    if (d_data.min_velocity <= 1.0e-8)
    {
        double acceleration_at_min_vel = 
            d_dt_eval.calculate_acceleration_magnitude(d_data.time_at_min_velocity,
                control_points, scale_factor);
        if(acceleration_at_min_vel <= 1.0e-8)
        {
            curvature_bound = 0;
        }
        else
        {
            curvature_bound = std::numeric_limits<double>::max();
        }
    }
    else
    {
        curvature_bound = d_data.max_acceleration/(d_data.min_velocity*d_data.min_velocity);
        curvature = d_data.max_cross_term/(d_data.min_velocity*d_data.min_velocity*d_data.min_velocity);
        if (curvature < curvature_bound)
        {
            curvature_bound = curvature;
        }
    }
    return curvature_bound;
}

template <int D>
double CrossTermBounds<D>::evaluate_interval_angular_rate_bound(Eigen::Matrix<double,D,4> &control_points,double &scale_factor)
{
    DerivativeBoundsData d_data = get_derivative_bound_data(control_points, scale_factor);
    double angular_rate_bound;
    double angular_rate;
    if (d_data.min_velocity <= 1.0e-8)
    {
        double acceleration_at_min_vel = 
            d_dt_eval.calculate_acceleration_magnitude(d_data.time_at_min_velocity,
                control_points, scale_factor);
        if(acceleration_at_min_vel <= 1.0e-8)
        {
            angular_rate_bound = 0;
        }
        else
        {
            angular_rate_bound = std::numeric_limits<double>::max();
        }
    }
    else
    {
        angular_rate_bound = d_data.max_acceleration/(d_data.min_velocity);
        angular_rate = d_data.max_cross_term/(d_data.min_velocity*d_data.min_velocity);
        if (angular_rate < angular_rate_bound)
        {
            angular_rate_bound = angular_rate;
        }
    }
    return angular_rate_bound;

}

template <int D>
double CrossTermBounds<D>::evaluate_interval_centripetal_acceleration_bound(Eigen::Matrix<double,D,4> &control_points,double &scale_factor)
{
    DerivativeBoundsData d_data = get_derivative_bound_data(control_points, scale_factor);
    double centripetal_acceleration_bound;
    double centripetal_acceleration;
    if (d_data.min_velocity <= 1.0e-8)
    {
        centripetal_acceleration_bound = 0;

    }
    else
    {
        centripetal_acceleration_bound = d_data.max_acceleration;
        centripetal_acceleration = d_data.max_cross_term/d_data.min_velocity;
        if (centripetal_acceleration < centripetal_acceleration_bound)
        {
            centripetal_acceleration_bound = centripetal_acceleration;
        }
    }
    return centripetal_acceleration_bound;

}

template <int D>
DerivativeBoundsData CrossTermBounds<D>::get_derivative_bound_data(Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    std::array<double,2> min_velocity_and_time = d_dt_bounds.find_min_velocity_and_time(control_points,scale_factor);
    double min_velocity = min_velocity_and_time[0];
    double time_at_min_velocity = min_velocity_and_time[1];
    double max_cross_term = find_maximum_cross_term(control_points,scale_factor);
    std::array<double,2> max_acceleration_and_time = d_dt_bounds.find_max_acceleration_and_time(control_points, scale_factor);
    double max_acceleration = max_acceleration_and_time[0];
    struct DerivativeBoundsData d_bound_data = {min_velocity, max_acceleration, max_cross_term, time_at_min_velocity};
    return d_bound_data;
}

template <int D>
double CrossTermBounds<D>::find_maximum_cross_term(Eigen::Matrix<double,D,4> &control_points, double &scale_factor)
{
    Eigen::Vector4d coeficients;
    if (D == 2) {coeficients = c_prop.get_2D_cross_coefficients(control_points);}
    else {coeficients = c_prop.get_3D_cross_coefficients(control_points);}
    double a_term = coeficients(0);
    double b_term = coeficients(1);
    double c_term = coeficients(2);
    double d_term = coeficients(3);
    std::array<double,3> roots = CubicEquationSolver::solve_equation(a_term,
        b_term, c_term, d_term);
    double t0 = 0;
    double tf = 1.0;
    double max_cross_term = c_eval.calculate_cross_term_magnitude(t0,control_points,scale_factor);
    double cross_term_at_tf = c_eval.calculate_cross_term_magnitude(tf,control_points,scale_factor);
    if (cross_term_at_tf > max_cross_term)
    {
        max_cross_term = cross_term_at_tf;
    }
    for(int index = 0; index < 3; index++)
    {
        double root = roots[index];
        if(root > 0 && root < 1.0)
        {
            double cross_term =  c_eval.calculate_cross_term_magnitude(root, control_points,scale_factor);
            if (cross_term > max_cross_term)
            {
                max_cross_term = cross_term;
            }
        }
    }
    return max_cross_term;
}

//Explicit template instantiations
template class CrossTermBounds<2>;
template class CrossTermBounds<3>;