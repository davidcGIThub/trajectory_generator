#include "CBindingHelper.hpp"
#include <iostream>

template<int D>
CBindingHelper<D>::CBindingHelper()
{
    
}

template<int D>
Eigen::Matrix<double,D,4> CBindingHelper<D>::array_section_to_eigen(double cont_pts[], int &num_cps, unsigned int &index)
{
    Eigen::Matrix<double,D,4> interval_control_points;
    if (D == 2)
    {
        interval_control_points << cont_pts[index], cont_pts[index+1], cont_pts[index+2], cont_pts[index+3],
            cont_pts[num_cps+index], cont_pts[num_cps+index+1], cont_pts[num_cps+index+2], cont_pts[num_cps+index+3];
    }
    else
    {
        interval_control_points << cont_pts[index], cont_pts[index+1], cont_pts[index+2], cont_pts[index+3],
            cont_pts[num_cps+index], cont_pts[num_cps+index+1], cont_pts[num_cps+index+2], cont_pts[num_cps+index+3],
            cont_pts[2*num_cps+index], cont_pts[2*num_cps+index+1], cont_pts[2*num_cps+index+2], cont_pts[2*num_cps+index+3];
    }
    return interval_control_points;
}

template <int D>
Eigen::MatrixXd CBindingHelper<D>::array_to_eigen(double cont_pts[], int &num_cps)
{
    Eigen::MatrixXd control_points(D,num_cps);
    for(unsigned int i = 0; i < num_cps; i++)
    {
        for (unsigned int j = 0; j < D; j++)
        {
            control_points(j,i) = cont_pts[i + num_cps*j];
        }
    }
    return control_points;
};

template <int D>
double CBindingHelper<D>::cross_term_magnitude(Eigen::Matrix<double,D,1> &velocity_vector,
            Eigen::Matrix<double,D,1> &acceleration_vector)
{
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

template <int D>
double CBindingHelper<D>::curvature_calculation(Eigen::Matrix<double,D,1> &velocity_vector,
            Eigen::Matrix<double,D,1> &acceleration_vector)
{
    double cross_term_mag = cross_term_magnitude(velocity_vector, acceleration_vector);
    double vel_cubed_mag = pow((velocity_vector.norm()),3);
    double curvature = 0;
    if (vel_cubed_mag > 0.00000001)
    {
        curvature = cross_term_mag / vel_cubed_mag;
    }
    return curvature;
}

// explicit instantiations
template class CBindingHelper<2>;
template class CBindingHelper<3>;