#ifndef CROSSTERMEVALUATOR_HPP
#define CROSSTERMEVALUATOR_HPP
#include <array>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "gtest/gtest_prod.h"
#include "DerivativeBounds.hpp"
#include "DerivativeEvaluator.hpp"

template <int D> // D is the dimension of the spline
class CrossTermEvaluator
{
    public:
        CrossTermEvaluator();
        double calculate_cross_term_magnitude(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        double calculate_curvature(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
    private:
        DerivativeEvaluator<D> d_dt_eval{};
        double calculate_cross_term_magnitude(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor,
            Eigen::Matrix<double,D,1> &velocity_vector);
        bool check_if_not_ascending_colinear(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
};

#endif

