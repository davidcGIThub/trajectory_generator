#ifndef DERIVATIVEEVALUATOR_HPP
#define DERIVATIVEEVALUATOR_HPP
#include <array>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "CubicEquationSolver.hpp"
#include "gtest/gtest_prod.h"
#include "CBindingHelper.hpp"

template <int D> // D is the dimension of the spline
class DerivativeEvaluator
{
    public:
        DerivativeEvaluator();
        double find_min_velocity_of_spline(double cont_pts[], int num_control_points, double scale_factor);
        double find_max_acceleration_of_spline(double cont_pts[], int num_control_points, double scale_factor);
        std::array<double,2> find_min_velocity_and_time(Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        std::array<double,2> find_max_acceleration_and_time(Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        Eigen::Matrix<double,D,1> calculate_velocity_vector(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        double calculate_velocity_magnitude(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        Eigen::Matrix<double,D,1> calculate_acceleration_vector(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        double calculate_acceleration_magnitude(double &t, Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
    private:
        CBindingHelper<D> cbind_help{};
        Eigen::Matrix<double, 4,4> get_third_order_M_matrix();
        Eigen::Vector4d get_third_order_T_derivative_vector(double &t, double &scale_factor);
        Eigen::Vector4d get_third_order_T_second_derivative_vector(double &t, double &scale_factor);
    FRIEND_TEST(DerivativeTest, MatrixRetrieval);
    FRIEND_TEST(DerivativeTest, DerivativeVectorRetrieval);
    FRIEND_TEST(DerivativeTest, SecondDerivativeVectorRetrieval);
    FRIEND_TEST(DerivativeTest, DerivativeVectorRetrievalError);
    FRIEND_TEST(DerivativeTest, SecondDerivativeVectorRetrievalError);
};
#endif

