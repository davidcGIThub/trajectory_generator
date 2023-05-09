#include "gtest/gtest.h"
#include "CrossTermEvaluator.hpp"

TEST(CrossTermEvaluatorTest, CrossTermMagnitude)
{
    CrossTermEvaluator<2> c_eval{};
    double scale_factor = 1;
    double true_cross_term = 6.632352941176468;
    Eigen::Matrix<double, 2,4> control_points;
    control_points << 5, 6, 8, 3,
                      6, 9, 8, 1;
    double time_t = 0.14705882352941177;
    double cross_term = c_eval.calculate_cross_term_magnitude(time_t, control_points,scale_factor);
    EXPECT_EQ(true_cross_term, true_cross_term);
}

TEST(CrossTermEvaluatorTest, CurvatureMagnitude)
{
    CrossTermEvaluator<3> c_eval{};
    Eigen::Matrix<double,3,4> control_points;
    control_points << 5, 8, 6, 7,
                      5, 4, 2, 7,
                      8, 7, 5, 9;
    double scale_factor = 1.0;
    double t = 1.0;
    double true_curvature = 1.5574517574154207;
    double curvature = c_eval.calculate_curvature(t, control_points, scale_factor);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_curvature, curvature, tolerance);
}


TEST(CrossTermEvaluatorTest, CurvatureMagnitudeZeroVel)
{
    CrossTermEvaluator<3> c_eval{};
    Eigen::Matrix<double,3,4> control_points;
    control_points << 0, 0, 0, 4,
                      0, 0, 0, 7,
                      0, 0, 0, 9;
    double scale_factor = 1.0;
    double t = 0.0;
    double true_curvature = 0;
    double curvature = c_eval.calculate_curvature(t, control_points, scale_factor);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_curvature, curvature, tolerance);
}

TEST(CrossTermEvaluatorTest, CurvatureMagnitudeSharp180Turn)
{
    CrossTermEvaluator<3> c_eval{};
    Eigen::Matrix<double,3,4> control_points;
    control_points << 0, 3, 3, 0,
                      0, 5, 5, 0,
                      0, 2, 2, 0;
    double scale_factor = 1.0;
    double t = 0.5;
    double true_curvature = std::numeric_limits<double>::max();
    double curvature = c_eval.calculate_curvature(t, control_points, scale_factor);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_curvature, curvature, tolerance);
}
