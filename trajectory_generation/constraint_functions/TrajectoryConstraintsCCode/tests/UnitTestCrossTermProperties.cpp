#include "gtest/gtest.h"
#include "CrossTermProperties.hpp"

TEST(CrossTermPropertiesTest, CrossCoeficients2D)
{
    CrossTermProperties<2> c_eval{};
    double c_3 = 288.0;
    double c_2 = -1008.0;
    double c_1 = 1168.0;
    double c_0 = -448.0;
    Eigen::Matrix<double, 2, 4> control_points;
    control_points << 9, 4, 8, 6,
                      1, 5, 5, 5;
    Eigen::Vector4d coeficient_array = c_eval.get_2D_cross_coefficients(control_points);
    double true_c_3 = coeficient_array(0);
    double true_c_2 = coeficient_array(1);
    double true_c_1 = coeficient_array(2);
    double true_c_0 = coeficient_array(3);
    EXPECT_EQ(true_c_3, c_3);
    EXPECT_EQ(true_c_2, c_2);
    EXPECT_EQ(true_c_1, c_1);
    EXPECT_EQ(true_c_0, c_0);
}

TEST(CrossTermPropertiesTest, CrossCoeficients3D)
{
    CrossTermProperties<3> c_eval{};
    double c_3 = 830.5;
    double c_2 = -354.75;
    double c_1 = 789.25;
    double c_0 = -154.0;
    Eigen::Matrix<double, 3, 4> control_points;
    control_points << 2, 4, 4, 2,
                      1, 2, 8, 6,
                      2, 3, 5, 0;
    Eigen::Vector4d coeficient_array = c_eval.get_3D_cross_coefficients(control_points);
    double true_c_3 = coeficient_array(0);
    double true_c_2 = coeficient_array(1);
    double true_c_1 = coeficient_array(2);
    double true_c_0 = coeficient_array(3);
    EXPECT_EQ(true_c_3, c_3);
    EXPECT_EQ(true_c_2, c_2);
    EXPECT_EQ(true_c_1, c_1);
    EXPECT_EQ(true_c_0, c_0);
}
