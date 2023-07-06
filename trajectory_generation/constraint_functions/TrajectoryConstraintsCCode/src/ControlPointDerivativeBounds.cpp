#include "ControlPointDerivativeBounds.hpp"
#include <iostream>
#include <stdexcept>

template <int D>
ControlPointDerivativeBounds<D>::ControlPointDerivativeBounds()
{

}


template <int D>
double ControlPointDerivativeBounds<D>::find_min_velocity_of_bez_vel_cont_pts(double bez_vel_cont_pts[], 
                                                                    int num_bez_vel_cont_pts)
{
    double min_velocity = std::numeric_limits<double>::max();
    double velocity;
    int step_size = 2;
    int order = 2;
    unsigned int j = 0;
    int num_intervals = int((num_bez_vel_cont_pts - 1)/order);
    for (unsigned int i = 0; i < num_intervals; i++)
    {
        j = i*step_size;
        Eigen::Matrix<double,D,3> interval_bez_vel_cont_pts = 
            cbind_help.array_section_to_eigen_3(bez_vel_cont_pts, num_bez_vel_cont_pts, j);
        std::cout << "bez cnt pts: " << interval_bez_vel_cont_pts << std::endl;
        velocity = find_min_velocity_hull_method(interval_bez_vel_cont_pts);
        std::cout << "velocity bound: " << velocity << std::endl;
        if (velocity < min_velocity)
        {
            min_velocity = velocity;
        }
    }
    return min_velocity;
}


template <int D>
double ControlPointDerivativeBounds<D>::find_min_velocity_hull_method(Eigen::Matrix<double,D,3> &interval_vel_cont_pts)
{
    const int num_points = 4;
    int max_iterations = 500;
    unsigned int initial_index = 0;
    double tolerance = 0.000001;
    double min_vel = mdm_obj.min_norm(interval_vel_cont_pts, max_iterations, initial_index, tolerance);
    return min_vel;
}

//Explicit template instantiations
template class ControlPointDerivativeBounds<2>;
template class ControlPointDerivativeBounds<3>;