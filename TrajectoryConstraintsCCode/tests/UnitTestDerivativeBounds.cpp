#include "gtest/gtest.h"
#include "DerivativeBounds.hpp"

// Create test for max velocity, also check if these work for variable scale factors

TEST(DerivativeBoundsTest, MinVelocity)
{
    DerivativeBounds<2> d_bounds{};
    double scale_factor = 1;
    double true_min_velocity = 0.12523779030065027;
    Eigen::Matrix<double, 2,4> control_points;
    control_points << 0.1,  3.2,  2.01, 1.1,
                      0.01, 2.98, 2.1,  0.99; 
    std::array<double,2> min_velocity_and_time = d_bounds.find_min_velocity_and_time(control_points,scale_factor);
    double min_velocity = min_velocity_and_time[0];
    double tolerance = 0.00001;
    EXPECT_NEAR(true_min_velocity, min_velocity,tolerance);
}

TEST(DerivativeBoundsTest, MaxVelocity)
{
    DerivativeBounds<3> d_bounds{};
    double scale_factor = 1.5;
    double true_max_velocity = 3.7214639159987115;
    Eigen::Matrix<double, 3,4> control_points;
    control_points << 5, 8, 6, 7,
                    5, 11, 2, 7,
                    8, 7, 5, 9; 
    std::array<double,2> max_velocity_and_time = d_bounds.find_max_velocity_and_time(control_points,scale_factor);
    double max_velocity = max_velocity_and_time[0];
    double tolerance = 0.00001;
    EXPECT_NEAR(true_max_velocity, max_velocity,tolerance);
}

TEST(DerivativeBoundsTest, MaxVelocitySingleDim)
{
    DerivativeBounds<3> d_bounds{};
    double scale_factor = 1.5;
    double true_max_velocity = 3.5862069;
    unsigned int dimension = 2;
    Eigen::Matrix<double, 3,4> control_points;
    control_points << 5, 8, 6, 7,
                    5, 11, 2, 7,
                    8, 7, 5, 9; 
    double max_velocity = d_bounds.find_max_velocity_magnitude_in_single_dimension(control_points, scale_factor, dimension);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_max_velocity, max_velocity,tolerance);
}

TEST(DerivativeBoundsTest, MaxAcceleration)
{
    DerivativeBounds<2> d_bounds{};
    double true_max_acceleration = 9.055385138137417;
    Eigen::Matrix<double, 2,4> control_points;
    control_points << 7, 4, 2, 0,
                      3, 1, 8, 7;
    double scale_factor = 1;
    std::array<double,2> max_acceleration_and_time = d_bounds.find_max_acceleration_and_time(control_points, scale_factor);
    double max_acceleration = max_acceleration_and_time[0];
    EXPECT_EQ(true_max_acceleration, max_acceleration);
}

TEST(DerivativeBoundsTest, MinVelocityOfSpline)
{
    DerivativeBounds<2> d_bounds{};
    double scale_factor = 1;
    double true_min_velocity = 0.7068769432442363;
    int num_control_points = 8;
    double control_points[] = {   -0.89402549, -0.05285741,  1.10545513,  2.47300498,
         3.79358126,  4.76115495,  5.11942253,  4.76115495, -0.10547684,  0.05273842,
        -0.10547684, -0.47275804, -0.79306865, -0.76080139, -0.11946946,  1.23867924};
    double min_velocity = d_bounds.find_min_velocity_of_spline(control_points, num_control_points,scale_factor);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_min_velocity, min_velocity, tolerance);
}


TEST(DerivativeBoundsTest, MaxAccelerationOfSpline)
{
    DerivativeBounds<2> d_bounds{};
    double scale_factor = 1;
    double true_max_acceleration = 1.0135328890911517;
    int num_control_points = 8;
    double control_points[] = {   -0.89402549, -0.05285741,  1.10545513,  2.47300498,
         3.79358126,  4.76115495,  5.11942253,  4.76115495, -0.10547684,  0.05273842,
        -0.10547684, -0.47275804, -0.79306865, -0.76080139, -0.11946946,  1.23867924};
    double max_acceleration = d_bounds.find_max_acceleration_of_spline(control_points, num_control_points,scale_factor);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_max_acceleration, max_acceleration, tolerance);
}

TEST(DerivativeBoundsTest, MinVelocityOfSpline3D)
{
    DerivativeBounds<3> d_bounds{};
    double scale_factor = 1;
    double true_min_velocity = 1.5;
    int num_control_points = 8;
    double control_points[] = {3, 0, 5, 1, 9, 6, 6, 4,
                                6, 2, 6, 8, 0, 8, 0, 6,
                                0, 0, 8, 0, 4, 7, 1, 8};
    double min_velocity = d_bounds.find_min_velocity_of_spline(control_points, num_control_points,scale_factor);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_min_velocity, min_velocity, tolerance);
}

TEST(DerivativeBoundsTest, MaxAccelerationOfSpline3D)
{
    DerivativeBounds<3> d_bounds{};
    double scale_factor = 1;
    double true_max_acceleration = 19.697715603592208;
    int num_control_points = 8;
    double control_points[] = {3, 0, 5, 1, 9, 6, 6, 4,
                                6, 2, 6, 8, 0, 8, 0, 6,
                                0, 0, 8, 0, 4, 7, 1, 8};
    double max_acceleration = d_bounds.find_max_acceleration_of_spline(control_points, num_control_points,scale_factor);
    double tolerance = 0.0000001;
    EXPECT_NEAR(true_max_acceleration, max_acceleration, tolerance);
}
