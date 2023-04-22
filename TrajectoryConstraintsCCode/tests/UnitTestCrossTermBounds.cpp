#include "gtest/gtest.h"
#include "CrossTermBounds.hpp"


TEST(CrossTermTest, MaxCrossTerm)
{
    CrossTermBounds<2> c_eval{};
    double scale_factor = 1;
    double true_max_cross_term = 18.0;
    Eigen::Matrix<double, 2, 4> control_points;
    control_points << 0, 1, 4, 4,
                      6, 1, 4, 3;
    double max_cross_term = c_eval.find_maximum_cross_term(control_points,scale_factor);
    EXPECT_EQ(true_max_cross_term, max_cross_term);
}

TEST(CrossTermTest, RandomNormalSplineIntervalCurvatureBound)
{
    CrossTermBounds<2> c_eval{};
    Eigen::Matrix<double,2,4> control_points;
    control_points << 0, 6, 2, 1,
                      4, 3, 5, 2;
    double true_curvature_bound = 18.447094346279968;
    double curvature_bound = c_eval.evaluate_interval_curvature_bound(control_points);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_curvature_bound, curvature_bound,tolerance);
}

TEST(CrossTermTest, MaxCurvatureAtEndpointIntervalCurvatureBound)
{
    CrossTermBounds<2> c_eval{};
    Eigen::Matrix<double,2,4> control_points;
    control_points << 6, 1, 7, 1,
                      0, 1, 8, 9;
    double true_curvature_bound = 0.75;
    double curvature_bound = c_eval.evaluate_interval_curvature_bound(control_points);
    double tolerance = 0.0000001;
    EXPECT_NEAR(true_curvature_bound, curvature_bound, tolerance);
}

TEST(CrossTermTest, NoMovementIntervalCurvatureBound)
{
    CrossTermBounds<2> c_eval{};
    Eigen::Matrix<double,2,4> control_points;
    control_points << 2,2,2,2,
                      3,3,3,3;
    double true_curvature_bound = 0;
    double curvature_bound = c_eval.evaluate_interval_curvature_bound(control_points);
    double tolerance = 0.0000001;
    EXPECT_NEAR(true_curvature_bound, curvature_bound, tolerance);
}

TEST(CrossTermTest, SameTwoInitialCPsIntervalCurvatureBound)
{
    CrossTermBounds<2> c_eval{};
    Eigen::Matrix<double,2,4> control_points;
    control_points << 2, 2, 6, 8,
                      3, 3, 0, 5;
    double true_curvature_bound = 1.3193938001976513;
    double curvature_bound = c_eval.evaluate_interval_curvature_bound(control_points);
    double tolerance = 0.000001;
    EXPECT_NEAR(true_curvature_bound, curvature_bound, tolerance);
}

TEST(CrossTermTest, SameThreeInitialCPsIntervalCurvatureBound)
{
    CrossTermBounds<2> c_eval{};
    Eigen::Matrix<double,2,4> control_points;
    control_points << 2, 2, 2, 8,
                      3, 3, 3, 5;
    double true_curvature_bound = 0;
    double curvature_bound = c_eval.evaluate_interval_curvature_bound(control_points);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_curvature_bound, curvature_bound, tolerance);
}

TEST(CrossTermTest, SameTwoEndAndInitialCPsIntervalCurvatureBound)
{
    CrossTermBounds<2> c_eval{};
    Eigen::Matrix<double,2,4> control_points;
    control_points << 2, 2, 8, 8,
                      3, 3, 5, 5;
    double true_curvature_bound = 0;
    double curvature_bound = c_eval.evaluate_interval_curvature_bound(control_points);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_curvature_bound, curvature_bound, tolerance);
}

TEST(CrossTermTest, InfiniteCurvatureIntervalCurvatureBound)
{
    CrossTermBounds<2> c_eval{};
    Eigen::Matrix<double,2,4> control_points;
    control_points << 1, 3, 2, 0,
                      1, 3, 2, 0;
    double true_curvature_bound = std::numeric_limits<double>::max();
    double curvature_bound = c_eval.evaluate_interval_curvature_bound(control_points);
    double tolerance = 0.000001;
    EXPECT_NEAR(true_curvature_bound, curvature_bound, tolerance);
}

TEST(CrossTermTest, VeryLargeCurvatureIntervalCurvatureBound)
{
    CrossTermBounds<2> c_eval{};
    Eigen::Matrix<double,2,4> control_points;
    control_points << 0.1, 3.2, 2.01, 1.1,
                      0.01, 2.98, 2.1, 0.99;
    double true_curvature_bound = 367.5125371522953;
    double curvature_bound = c_eval.evaluate_interval_curvature_bound(control_points);
    double tolerance = 0.001;
    EXPECT_NEAR(true_curvature_bound, curvature_bound, tolerance);
}

TEST(CrossTermTest, StraightLineIntervalCurvatureBound)
{
    CrossTermBounds<2> c_eval{};
    Eigen::Matrix<double,2,4> control_points;
    control_points << 1,4,6,9,
                      1,4,6,9;
    double true_curvature_bound = 0;
    double curvature_bound = c_eval.evaluate_interval_curvature_bound(control_points);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_curvature_bound, curvature_bound, tolerance);
}

TEST(CrossTermTest, SplineCurvatureBound)
{
    CrossTermBounds<2> c_eval{};
    double true_max_curvature = 1.6000000063248174;
    int num_control_points = 8;
    double control_points[] = {-0.96055144, -0.04452866, 1.13866608, 2.49667343, 
        3.80204877, 4.76675134, 5.11662433,  4.76675134, -0.13352168, 0.06676084,
        -0.13352168, -0.58215741, -0.93804997, -0.84450962, -0.10254957,  1.25470791};
    double max_curvature = c_eval.get_spline_curvature_bound(control_points, num_control_points);
    double tolerance = 0.000001;
    EXPECT_NEAR(true_max_curvature, max_curvature, tolerance);
}

TEST(CrossTermTest, SplineCurvatureBound2)
{
    CrossTermBounds<2> c_eval{};
    double true_max_curvature = 2.028118191541948;
    int num_control_points = 8;
    double control_points[] = {-0.89402549, -0.05285741,  1.10545513, 2.47300498,
        3.79358126,  4.76115495, 5.11942253,  4.76115495, -0.10547684,  0.05273842,
        -0.10547684, -0.47275804, -0.79306865, -0.76080139, -0.11946946,  1.23867924};
    double max_curvature = c_eval.get_spline_curvature_bound(control_points, num_control_points);
    double tolerance = 0.000001;
    EXPECT_NEAR(true_max_curvature, max_curvature, tolerance);
}

TEST(CrossTermTest, SplineCurvatureBound3)
{
    CrossTermBounds<3> c_eval{};
    double true_max_curvature = 2.4603294220963416;
    int num_control_points = 4;
    double control_points[] = {4, 1, 4, 5,
                                2, 2, 0, 4,
                                7, 0, 1, 7};
    double max_curvature = c_eval.get_spline_curvature_bound(control_points, num_control_points);
    double tolerance = 0.000001;
    EXPECT_NEAR(true_max_curvature, max_curvature, tolerance);
}


TEST(CrossTermTest, SplineAngularRateBound)
{
    CrossTermBounds<2> c_eval{};
    double true_max_angular_rate = 1.0175086478172288;
    int num_control_points = 8;
    double scale_factor = 1.2;
    double control_points[] = {-0.96055144, -0.04452866, 1.13866608, 2.49667343, 
        3.80204877, 4.76675134, 5.11662433,  4.76675134, -0.13352168, 0.06676084,
        -0.13352168, -0.58215741, -0.93804997, -0.84450962, -0.10254957,  1.25470791};
    double max_angular_rate = c_eval.get_spline_angular_rate_bound(control_points, num_control_points,scale_factor);
    double tolerance = 0.000001;
    EXPECT_NEAR(true_max_angular_rate, max_angular_rate, tolerance);
}

TEST(CrossTermTest, SplineCentripetalAccelerationBound)
{
    CrossTermBounds<2> c_eval{};
    double true_max_centripetal_acceleration = 0.64707741729649182;
    int num_control_points = 8;
    double scale_factor = 1.2;
    double control_points[] = {-0.96055144, -0.04452866, 1.13866608, 2.49667343, 
        3.80204877, 4.76675134, 5.11662433,  4.76675134, -0.13352168, 0.06676084,
        -0.13352168, -0.58215741, -0.93804997, -0.84450962, -0.10254957,  1.25470791};
    double max_centripetal_acceleration = c_eval.get_spline_centripetal_acceleration_bound(control_points, num_control_points,scale_factor);
    double tolerance = 0.000001;
    EXPECT_NEAR(true_max_centripetal_acceleration, max_centripetal_acceleration, tolerance);
}