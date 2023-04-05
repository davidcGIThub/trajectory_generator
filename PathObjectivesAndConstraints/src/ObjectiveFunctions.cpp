#include "ObjectiveFunctions.hpp"
#include <iostream>

template <int D>
ObjectiveFunctions<D>::ObjectiveFunctions()
{

}

template <int D>
double ObjectiveFunctions<D>::minimize_acceleration_and_time(double cont_pts[], int num_control_points, double scale_factor)
{
    double acceleration_integral = minimize_acceleration(cont_pts, num_control_points);
    return acceleration_integral + scale_factor;
}

template <int D>
double ObjectiveFunctions<D>::minimize_distance_and_time(double cont_pts[], int num_control_points, double scale_factor)
{
    double distance_integral = minimize_distance(cont_pts, num_control_points);
    return distance_integral + scale_factor;
}

template <int D>
double ObjectiveFunctions<D>::minimize_acceleration(double cont_pts[], int &num_control_points)
{
    int step = num_control_points;
    int order = 3;
    int num_intervals = num_control_points - order;
    double sum_of_integrals{0};
    Eigen::Matrix<double,D,4> interval_control_points;
    Eigen::Matrix<double, D, 1> p0;
    Eigen::Matrix<double, D, 1> p1;
    Eigen::Matrix<double, D, 1> p2;
    Eigen::Matrix<double, D, 1> p3; 
    Eigen::Matrix<double, D, 1> integral_vector;
    for (unsigned int i = 0; i<num_intervals; i++)
    {
        // interval_control_points = cbind_help.array_section_to_eigen(cont_pts, num_control_points, i);
        interval_control_points = cbind_help.array_section_to_eigen(cont_pts, num_control_points, i);
        p0 = interval_control_points.block(0,0,D,1);
        p1 = interval_control_points.block(0,1,D,1);
        p2 = interval_control_points.block(0,2,D,1);
        p3 = interval_control_points.block(0,3,D,1); 
        integral_vector = (p0 - 3*p1 + 3*p2 - p3);
        integral_vector = integral_vector.cwiseProduct(integral_vector);
        sum_of_integrals += integral_vector.sum();
    }
    return sum_of_integrals;
}

template <int D>
double ObjectiveFunctions<D>::minimize_distance(double cont_pts[], int &num_control_points)
{
    int step = num_control_points;
    int order = 3;
    int num_intervals = num_control_points - order;
    double sum_of_integrals{0};
    Eigen::Matrix<double,D,4> interval_control_points;
    Eigen::Matrix<double, D, 1> p0; 
    Eigen::Matrix<double, D, 1> p1;
    Eigen::Matrix<double, D, 1> p2;
    Eigen::Matrix<double, D, 1> p3;
    Eigen::Matrix<double, D, 1> a_temp;
    Eigen::Matrix<double, D, 1> a;
    Eigen::Matrix<double, D, 1> b; 
    Eigen::Matrix<double, D, 1> c_temp; 
    Eigen::Matrix<double, D, 1> c;
    Eigen::Matrix<double, D, 1> d;
    Eigen::Matrix<double, D, 1> f_temp;
    Eigen::Matrix<double, D, 1> f;
    Eigen::Matrix<double, D, 1> integral_vector;
    for (unsigned int i = 0; i<num_intervals; i++)
    {
        // interval_control_points = cbind_help.array_section_to_eigen(cont_pts, num_control_points, i);
        interval_control_points = cbind_help.array_section_to_eigen(cont_pts, num_control_points, i);
        p0 = interval_control_points.block(0,0,D,1);
        p1 = interval_control_points.block(0,1,D,1);
        p2 = interval_control_points.block(0,2,D,1);
        p3 = interval_control_points.block(0,3,D,1);
        a_temp = (p0/2 - (3*p1)/2 + (3*p2)/2 - p3/2);
        a = a_temp.cwiseProduct(a_temp);
        b = -2*(p0 - 2*p1 + p2).cwiseProduct(p0/2 - (3*p1)/2 + (3*p2)/2 - p3/2);
        c_temp = (p0 - 2*p1 + p2);
        c = c_temp.cwiseProduct(c_temp) + 2*(p0/2 - p2/2).cwiseProduct(p0/2 - (3*p1)/2 + (3*p2)/2 - p3/2);
        d = -2*(p0/2 - p2/2).cwiseProduct(p0 - 2*p1 + p2);
        f_temp = (p0/2 - p2/2);
        f =  f_temp.cwiseProduct(f_temp);
        integral_vector = a/5 + b/4 + c/3 + d/2 + f;
        sum_of_integrals += integral_vector.sum();
    }
    return sum_of_integrals;
}

//explicit instantiation

template class ObjectiveFunctions<2>;
template class ObjectiveFunctions<3>;