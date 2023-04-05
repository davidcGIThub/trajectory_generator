#include "gtest/gtest.h"
#include "DerivativeEvaluator.hpp"

TEST(DerivativeTest, MatrixRetrieval)
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

TEST(DerivativeTest, DerivativeVectorRetrieval)
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

TEST(DerivativeTest, SecondDerivativeVectorRetrieval)
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

TEST(DerivativeTest, DerivativeVectorRetrievalError)
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

TEST(DerivativeTest, SecondDerivativeVectorRetrievalError)
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

TEST(DerivativeTest, VelocityMagnitude)
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

TEST(DerivativeTest, AccelerationMagnitude)
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

TEST(DerivativeTest, MinVelocity)
{
    DerivativeEvaluator<2> c_eval{};
    double scale_factor = 1;
    double true_min_velocity = 0.12523779030065027;
    Eigen::Matrix<double, 2,4> control_points;
    control_points << 0.1,  3.2,  2.01, 1.1,
                      0.01, 2.98, 2.1,  0.99; 
    std::array<double,2> min_velocity_and_time = c_eval.find_min_velocity_and_time(control_points,scale_factor);
    double min_velocity = min_velocity_and_time[0];
    double tolerance = 0.00001;
    EXPECT_NEAR(true_min_velocity, min_velocity,tolerance);
}

TEST(DerivativeTest, MaxAcceleration)
{
    DerivativeEvaluator<2> c_eval{};
    double true_max_acceleration = 9.055385138137417;
    Eigen::Matrix<double, 2,4> control_points;
    control_points << 7, 4, 2, 0,
                      3, 1, 8, 7;
    double scale_factor = 1;
    std::array<double,2> max_acceleration_and_time = c_eval.find_max_acceleration_and_time(control_points, scale_factor);
    double max_acceleration = max_acceleration_and_time[0];
    EXPECT_EQ(true_max_acceleration, max_acceleration);
}

TEST(DerivativeTest, MinVelocityOfSpline)
{
    DerivativeEvaluator<2> c_eval{};
    double scale_factor = 1;
    double true_min_velocity = 0.7068769432442363;
    int num_control_points = 8;
    double control_points[] = {   -0.89402549, -0.05285741,  1.10545513,  2.47300498,
         3.79358126,  4.76115495,  5.11942253,  4.76115495, -0.10547684,  0.05273842,
        -0.10547684, -0.47275804, -0.79306865, -0.76080139, -0.11946946,  1.23867924};
    double min_velocity = c_eval.find_min_velocity_of_spline(control_points, num_control_points,scale_factor);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_min_velocity, min_velocity, tolerance);
}


TEST(DerivativeTest, MaxAccelerationOfSpline)
{
    DerivativeEvaluator<2> c_eval{};
    double scale_factor = 1;
    double true_max_acceleration = 1.0135328890911517;
    int num_control_points = 8;
    double control_points[] = {   -0.89402549, -0.05285741,  1.10545513,  2.47300498,
         3.79358126,  4.76115495,  5.11942253,  4.76115495, -0.10547684,  0.05273842,
        -0.10547684, -0.47275804, -0.79306865, -0.76080139, -0.11946946,  1.23867924};
    double max_acceleration = c_eval.find_max_acceleration_of_spline(control_points, num_control_points,scale_factor);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_max_acceleration, max_acceleration, tolerance);
}

TEST(DerivativeTest, MinVelocityOfSpline3D)
{
    DerivativeEvaluator<3> c_eval{};
    double scale_factor = 1;
    double true_min_velocity = 1.5;
    int num_control_points = 8;
    double control_points[] = {3, 0, 5, 1, 9, 6, 6, 4,
                                6, 2, 6, 8, 0, 8, 0, 6,
                                0, 0, 8, 0, 4, 7, 1, 8};
    double min_velocity = c_eval.find_min_velocity_of_spline(control_points, num_control_points,scale_factor);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_min_velocity, min_velocity, tolerance);
}

TEST(DerivativeTest, MaxAccelerationOfSpline3D)
{
    DerivativeEvaluator<3> c_eval{};
    double scale_factor = 1;
    double true_max_acceleration = 19.697715603592208;
    int num_control_points = 8;
    double control_points[] = {3, 0, 5, 1, 9, 6, 6, 4,
                                6, 2, 6, 8, 0, 8, 0, 6,
                                0, 0, 8, 0, 4, 7, 1, 8};
    double max_acceleration = c_eval.find_max_acceleration_of_spline(control_points, num_control_points,scale_factor);
    double tolerance = 0.0000001;
    EXPECT_NEAR(true_max_acceleration, max_acceleration, tolerance);
}
