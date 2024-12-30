#ifndef DERIVATIVEEVALUATOR_HPP
#define DERIVATIVEEVALUATOR_HPP
#include <array>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "CubicEquationSolver.hpp"
#include "gtest/gtest_prod.h"
#include "CBindHelperFunctions.hpp"

template <int D> // D is the dimension of the spline
class DerivativeEvaluator
{
    public:
        DerivativeEvaluator();
        Eigen::Matrix<double,D,1> calculate_position_vector(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        Eigen::Matrix<double,D,1> calculate_velocity_vector(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        double calculate_velocity_magnitude(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        Eigen::Matrix<double,D,1> calculate_acceleration_vector(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        double calculate_acceleration_magnitude(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        Eigen::Matrix<double, 4,4> get_third_order_M_matrix();
        Eigen::Vector4d get_third_order_T_vector(double &t, double &scale_factor);
        Eigen::Vector4d get_third_order_T_derivative_vector(double &t, double &scale_factor);
        Eigen::Vector4d get_third_order_T_second_derivative_vector(double &t, double &scale_factor);
    private:
        CBindHelperFunctions<D> cbind_help{};
};
#endif

