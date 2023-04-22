#include "RotationHelperFunctions.hpp"
#include <iostream>

template<int D>
RotationHelperFunctions<D>::RotationHelperFunctions()
{
    
}

template <int D>
double RotationHelperFunctions<D>::get2DXDirToVectorAngle(Eigen::Matrix<double, D, 1> &vector)
{
    double dx = vector.coeff(0);
    double dy = vector.coeff(1);
    double psi = atan2(dy,dx);
    return psi;
}

template <int D>
Eigen::Matrix<double,2,1> RotationHelperFunctions<D>::get3DXDirToVectorAngles(Eigen::Matrix<double, D, 1> &vector)
{
    double dx_1 = vector.coeff(0);
    double dz_1 = vector.coeff(2);
    double theta = atan2(dz_1,dx_1);
    Eigen::Matrix<double,D,D> R_y = get_3D_Y_rotation(theta);
    Eigen::Matrix<double,D,1> vector_2 = R_y * vector;
    double dx_2 = vector_2.coeff(0);
    double dy_2 = vector_2.coeff(1);
    double psi = atan2(dy_2,dx_2);
    Eigen::Matrix<double,2,1> angles;
    angles << psi , -theta;
    return angles;
}

template <int D>
Eigen::Matrix<double, D, D> RotationHelperFunctions<D>::get_2D_Z_rotation(double &psi)
{
    Eigen::Matrix<double,D,D> rotation_z;
    double c_psi = cos(psi);
    double s_psi = sin(psi);
    rotation_z << c_psi, -s_psi, 
                s_psi,  c_psi;
    return rotation_z;
}

template <int D>
Eigen::Matrix<double, D, D> RotationHelperFunctions<D>::get_3D_Y_rotation(double &theta)
{
    double c_theta = cos(theta);
    double s_theta = sin(theta);
    Eigen::Matrix<double,D,D> rotation_y;
    rotation_y << c_theta, 0, s_theta,
                0, 1,       0,
         -s_theta, 0, c_theta;
    return rotation_y;
}

template <int D>
Eigen::Matrix<double, D, D> RotationHelperFunctions<D>::get_3D_Z_rotation(double &psi)
{
    double c_psi = cos(psi);
    double s_psi = sin(psi);
    Eigen::Matrix<double,D,D> rotation_z;
    rotation_z << c_psi, -s_psi, 0,
                  s_psi,  c_psi, 0,
                      0,      0, 1;
    return rotation_z;
}

// explicit instantiations
template class RotationHelperFunctions<2>;
template class RotationHelperFunctions<3>;