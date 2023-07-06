#include "gtest/gtest.h"
#include "ControlPointDerivativeBounds.hpp"

// Create test for max velocity, also check if these work for variable scale factors

TEST(ControlPointDerivativeBoundsTest, MinVelocityHull)
{
    ControlPointDerivativeBounds<3> cpd_bounds{};
    double true_min_velocity = 5.393256464781546;
    Eigen::Matrix<double, 3,3> vel_control_points;
    vel_control_points << 1.69984, 3.13518, 3.75944,
                      4.23739,  3.8813, 4.23739,
                      3,             5,       1; 
    double min_velocity = cpd_bounds.find_min_velocity_hull_method(vel_control_points);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_min_velocity, min_velocity,tolerance);
}


TEST(ControlPointDerivativeBoundsTest, MinVelocityOfBezVelContPts)
{
    ControlPointDerivativeBounds<2> cpd_bounds{};
    double true_min_velocity = 2.0272234277287215;
    int num_bez_vel_cont_pts = 5;
    double bez_vel_cont_pts[] = {0.89402549,   2.10545513,  2.47300498, 3.79358126,  4.76115495,
                                5.11942253,   -0.10547684,  0.05273842, -0.10547684, -0.47275804};
    double min_velocity = cpd_bounds.find_min_velocity_of_bez_vel_cont_pts(bez_vel_cont_pts, num_bez_vel_cont_pts);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_min_velocity, min_velocity, tolerance);
}






