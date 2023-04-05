#include "gtest/gtest.h"
#include "CurvatureConstraints.hpp"

TEST(CurvatureConstraintTests, SplineCurvatureConstraint1)
{
    CurvatureConstraints<2> c_const{};
    double max_curvature = 0.5;
    double true_constraint = 1.1000000063248174;
    int num_control_points = 8;
    double control_points[] = {-0.96055144, -0.04452866, 1.13866608, 2.49667343, 
        3.80204877, 4.76675134, 5.11662433,  4.76675134, -0.13352168, 0.06676084,
        -0.13352168, -0.58215741, -0.93804997, -0.84450962, -0.10254957,  1.25470791};
    double constraint = c_const.get_spline_curvature_constraint(control_points, 
        num_control_points, max_curvature);
    double tolerance = 0.000001;
    EXPECT_NEAR(true_constraint, constraint, tolerance);
}

TEST(CurvatureConstraintTests, SplineCurvatureConstraint2)
{
    CurvatureConstraints<2> c_const{};
    double max_curvature = 3;
    double true_constraint = -0.9718818084580518;
    int num_control_points = 8;
    double control_points[] = {-0.89402549, -0.05285741,  1.10545513, 2.47300498,
        3.79358126,  4.76115495, 5.11942253,  4.76115495, -0.10547684,  0.05273842,
        -0.10547684, -0.47275804, -0.79306865, -0.76080139, -0.11946946,  1.23867924};
    double constraint = c_const.get_spline_curvature_constraint(control_points, 
        num_control_points, max_curvature);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_constraint, constraint, tolerance);
}

TEST(CurvatureConstraintTests, SplineCurvatureConstraint3)
{
    CurvatureConstraints<3> c_const{};
    double max_curvature = 2;
    double true_constraint = 0.46032942209634164;
    int num_control_points = 4;
    double control_points[] = {4, 1, 4, 5,
                                2, 2, 0, 4,
                                7, 0, 1, 7};
    double constraint = c_const.get_spline_curvature_constraint(control_points, 
        num_control_points, max_curvature);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_constraint, constraint, tolerance);
}

TEST(CurvatureConstraintTests, SplineIntervalCurvatureConstraints)
{
    CurvatureConstraints<2> c_const{};
    int num_control_points = 8;
    double max_curvature = 1;
    double control_points[] = {-0.96055144, -0.04452866, 1.13866608, 2.49667343, 
        3.80204877, 4.76675134, 5.11662433,  4.76675134, -0.13352168, 0.06676084,
        -0.13352168, -0.58215741, -0.93804997, -0.84450962, -0.10254957,  1.25470791};
    double* constraints = c_const.get_interval_curvature_constraints(control_points, 
        num_control_points, max_curvature);
    double tolerance = 0.00001;
    EXPECT_NEAR(constraints[0], -0.636405, tolerance);
    EXPECT_NEAR(constraints[1], -0.885224, tolerance);
    EXPECT_NEAR(constraints[2], -0.687979, tolerance);
    EXPECT_NEAR(constraints[3], 0.44596, tolerance);
    EXPECT_NEAR(constraints[4], 0.6, tolerance);
}
