#ifndef ROTATIONHELPERFUNCTIONS_HPP
#define ROTATIONHELPERFUNCTIONS_HPP
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "gtest/gtest_prod.h"

template <int D>
class RotationHelperFunctions
{
    public:
        RotationHelperFunctions();
        double get2DXDirToVectorAngle(Eigen::Matrix<double, D, 1> &vector);
        Eigen::Matrix<double,2,1> get3DXDirToVectorAngles(Eigen::Matrix<double, D, 1> &vector);
        Eigen::Matrix<double, D, D> get_2D_Z_rotation(double &psi);
        Eigen::Matrix<double, D, D> get_3D_Y_rotation(double &theta);
        Eigen::Matrix<double, D, D> get_3D_Z_rotation(double &psi);
};

#endif