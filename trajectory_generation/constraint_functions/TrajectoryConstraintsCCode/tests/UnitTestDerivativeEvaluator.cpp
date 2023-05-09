#include "gtest/gtest.h"
#include "DerivativeEvaluator.hpp"

TEST(DerivativeEvaluationTest, MatrixRetrieval)
{
    DerivativeEvaluator<2> c_eval{};
    Eigen::Matrix<double, 4,4> true_M;
    true_M << -0.16666667,  0.5, -0.5, 0.16666667,
               0.5,          -1,    0, 0.66666667,
              -0.5,         0.5,  0.5, 0.16666667,
        0.16666667,           0,    0, 0;
    Eigen::Matrix<double, 4,4> M = c_eval.get_third_order_M_matrix();
    double tolerance = 0.00000001;
    EXPECT_TRUE(true_M.isApprox(M,tolerance));
}

TEST(DerivativeEvaluationTest, TVectorRetrieval)
{
    DerivativeEvaluator<2> c_eval{};
    double scale_factor = 1.5;
    Eigen::Vector4d true_t_vector;
    true_t_vector << 0.012703703703703701, 0.054444444444444434, 0.2333333333333333, 1;
    double t = 0.35;
    Eigen::Vector4d t_vector = c_eval.get_third_order_T_vector(t,scale_factor);
    double tolerance = 0.00001;
    EXPECT_TRUE(true_t_vector.isApprox(t_vector, tolerance));
}

TEST(DerivativeEvaluationTest, DerivativeVectorRetrieval)
{
    DerivativeEvaluator<2> c_eval{};
    double scale_factor = 1;
    Eigen::Vector4d true_dt_vector;
    true_dt_vector << 0.36757914, 0.70007536, 1, 0;
    double t = 0.35003768191677603;
    Eigen::Vector4d dt_vector = c_eval.get_third_order_T_derivative_vector(t,scale_factor);
    double tolerance = 0.00001;
    EXPECT_TRUE(true_dt_vector.isApprox(dt_vector, tolerance));
}

TEST(DerivativeEvaluationTest, SecondDerivativeVectorRetrieval)
{
    double scale_factor = 1;
    DerivativeEvaluator<2> c_eval{};
    Eigen::Vector4d true_ddt_vector;
    double t = 0.5199907026925293;
    true_ddt_vector << 3.11994422, 2, 0, 0;
    Eigen::Vector4d ddt_vector = c_eval.get_third_order_T_second_derivative_vector(t,scale_factor);
    double tolerance = 0.000001;
    EXPECT_TRUE(true_ddt_vector.isApprox(ddt_vector, tolerance));
}

TEST(DerivativeEvaluationTest, DerivativeVectorRetrievalError)
{
    // this tests _that_ the expected exception is thrown
    DerivativeEvaluator<2> c_eval{};
    double scale_factor = 1;
    EXPECT_THROW(
    try
    {
        double t = -0.4;
        c_eval.get_third_order_T_derivative_vector(t,scale_factor);
    }
    catch(std::invalid_argument const& e)
    {
        EXPECT_STREQ("t value should be between 0 and 1", e.what());
        throw;
    }, std::invalid_argument);
}

TEST(DerivativeEvaluationTest, SecondDerivativeVectorRetrievalError)
{
    // this tests _that_ the expected exception is thrown
    DerivativeEvaluator<2> c_eval{};
    EXPECT_THROW(
    try
    {
        double scale_factor = 1;
        double t = 1.3;
        c_eval.get_third_order_T_second_derivative_vector(t, scale_factor);
    }
    catch(std::invalid_argument const& e)
    {
        EXPECT_STREQ("t value should be between 0 and 1", e.what());
        throw;
    }, std::invalid_argument);
}

TEST(DerivativeEvaluationTest, PositionVector)
{
    DerivativeEvaluator<3> c_eval{};
    double scale_factor = 1.5;
    Eigen::Matrix<double, 3,4> control_points;
    control_points << 5, 8, 6, 7,
                      5, 4, 2, 7,
                      8, 7, 5, 9;
    Eigen::Vector3d true_position;
    true_position << 7.14750617, 3.42306173, 6.41990123;
    double time_t = 0.4;
    Eigen::Vector3d position = c_eval.calculate_position_vector(time_t, control_points,scale_factor);
    double tolerance = 0.000000001;
    EXPECT_TRUE(true_position.isApprox(position, tolerance));
}

TEST(DerivativeEvaluationTest, VelocityMagnitude)
{
    DerivativeEvaluator<2> c_eval{};
    double scale_factor = 1;
    double true_velocity = 2.8230311463199973;
    Eigen::Matrix<double, 2,4> control_points;
    control_points << 4, 7, 4, 4,
                      3, 5, 8, 4;
    double time_t = 0.35768880134338066;
    double velocity = c_eval.calculate_velocity_magnitude(time_t, control_points,scale_factor);
    double tolerance = 0.000000001;
    EXPECT_NEAR(true_velocity, velocity,tolerance);
}

TEST(DerivativeEvaluationTest, AccelerationMagnitude)
{
    DerivativeEvaluator<2> c_eval{};
    double true_acceleration = 2.2568129685516602;
    Eigen::Matrix<double, 2,4> control_points;
    control_points << 8, 3, 6, 4,
                      6, 5, 7, 3;
    double time_t = 0.4696969696969697;
    double scale_factor = 1;
    double acceleration = c_eval.calculate_acceleration_magnitude(time_t, control_points, scale_factor);
    double tolerance = 0.000001;
    EXPECT_NEAR(true_acceleration, acceleration,tolerance);
}

