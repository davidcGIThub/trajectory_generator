#ifndef CROSSTERMPROPERTIES_HPP
#define CROSSTERMPROPERTIES_HPP
#include <array>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "gtest/gtest_prod.h"
#include "CBindHelperFunctions.hpp"
#include "CrossTermEvaluator.hpp"

template <int D> // D is the dimension of the spline
class CrossTermProperties
{
    public:
        CrossTermProperties();
        Eigen::Vector4d get_2D_cross_coefficients(Eigen::Matrix<double,D,4> &control_points);
        Eigen::Vector4d get_3D_cross_coefficients(Eigen::Matrix<double,D,4> &control_points);
    private:
        CrossTermEvaluator<D> c_eval{};
};

#endif

