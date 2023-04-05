#include "gtest/gtest.h"
#include "ObjectiveFunctions.hpp"


TEST(ObjectiveFunctionsTests, MinimizeAccelAndTime1)
{
    ObjectiveFunctions<2> objective_func{};
    double control_points[] = {-8.13473798e-01, -5.26818295e-02,  1.02420112e+00,  2.36108527e+00,
        3.76503405e+00,  5.03277942e+00,  6.02419156e+00,  6.87045435e+00,
       -6.93589629e-03,  3.46794815e-03, -6.93589629e-03, -3.22053557e-02,
       -4.36547416e-02, -2.65620089e-02,  1.32810044e-02, -2.65620089e-02};
    int num_control_points = 8;
    double scale_factor = 0.9188375660069013;
    double true_objective = 1.0489732788624528;
    double objective = objective_func.minimize_acceleration_and_time(control_points, num_control_points, scale_factor);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_objective, objective, tolerance);
}

TEST(ObjectiveFunctionsTests, MinimizeAccelAndTime2)
{
    ObjectiveFunctions<3> objective_func{};
    double control_points[] = {2, 3, 2, 7, 8, 5, 6, 2,
                                6, 8, 3, 0, 3, 8, 2, 4,
                                9, 2, 0, 9, 8, 2, 9, 3};
    int num_control_points = 8;
    double scale_factor = 1;
    double true_objective = 2455;
    double objective = objective_func.minimize_acceleration_and_time(control_points, num_control_points, scale_factor);
    double tolerance = 0.0000001;
    EXPECT_NEAR(true_objective, objective, tolerance);
}

TEST(ObjectiveFunctionsTests, MinimizeDistanceAndTime1)
{
    ObjectiveFunctions<2> objective_func{};
    double control_points[] = {-0.81165176, -0.05155, 1.01785177,  2.40103484,  3.7187882,   5.02253531, 6.03135648,  6.85203884,
                                0.01770916, -0.00885458,  0.01770916,  0.04368433,  0.06011326,  0.0669442, -0.0334721,   0.0669442};
    int num_control_points = 8;
    double scale_factor = 0.9147517645848163;
    double true_objective = 8.21631083225973;
    double objective = objective_func.minimize_distance_and_time(control_points, num_control_points, scale_factor);
    double tolerance = 0.00001;
    EXPECT_NEAR(true_objective, objective, tolerance);
}

TEST(ObjectiveFunctionsTests, MinimizeDistanceAndTime2)
{
    ObjectiveFunctions<3> objective_func{};
    double control_points[] = {7, 3, 1, 9, 8, 4, 0, 6,
                                6, 0, 3, 6, 2, 7, 8, 8,
                                0, 3, 1, 3, 8, 8, 9, 6};
    int num_control_points = 8;
    double scale_factor = 1;
    double true_objective = 93.88333333333334;
    double objective = objective_func.minimize_distance_and_time(control_points, num_control_points, scale_factor);
    double tolerance = 0.0000001;
    EXPECT_NEAR(true_objective, objective, tolerance);
}