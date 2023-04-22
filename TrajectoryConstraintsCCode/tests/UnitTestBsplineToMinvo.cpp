#include "gtest/gtest.h"
#include "BsplineToMinvo.hpp"

TEST(BsplineToMinvoTests, 2DFirstOrderConversion)
{
    Eigen::Matrix<double, 2,2> bspline_control_points;
    bspline_control_points << 0, 2,
                              2,-1;
    Eigen::Matrix<double, 2,2> true_minvo_control_points;
    true_minvo_control_points << 0, 2,
                                 2,-1;
    Eigen::Matrix<double, 2,2> minvo_control_points;
    BsplineToMinvo<2> converter{};
    minvo_control_points << converter.convert_1st_order_spline(bspline_control_points);
    EXPECT_TRUE(minvo_control_points.isApprox(true_minvo_control_points));
}

TEST(BsplineToMinvoTests, 2DSecondOrderConversion)
{
    Eigen::Matrix<double, 2, 3> bspline_control_points;
    bspline_control_points << 3, 0, 9,
                              1, 6, 2;
    Eigen::Matrix<double, 2,3> true_minvo_control_points;
    true_minvo_control_points << 1.26794922, 0.99999996, 4.73205078,
                                 3.46132487, 5.24999997, 4.03867513;
    Eigen::Matrix<double, 2,3> minvo_control_points;
    BsplineToMinvo<2> converter{};
    minvo_control_points << converter.convert_2nd_order_spline(bspline_control_points);
    EXPECT_TRUE(minvo_control_points.isApprox(true_minvo_control_points));
}

TEST(BsplineToMinvoTests, 2DThirdOrderConversion)
{
    Eigen::Matrix<double, 2, 4> bspline_control_points;
    bspline_control_points << 7, 8, 0, 1,
                              1, 4, 1, 4;
    Eigen::Matrix<double, 2, 4> true_minvo_control_points;
    true_minvo_control_points << 6.89483632, 5.71620216, 2.28379784, 1.10516368,
                                 3.0892793,  2.95335598, 2.04664402, 1.9107207;
    Eigen::Matrix<double, 2,4> minvo_control_points;
    BsplineToMinvo<2> converter{};
    minvo_control_points << converter.convert_3rd_order_spline(bspline_control_points);
    double tolerance = 0.000001;
    EXPECT_TRUE(minvo_control_points.isApprox(true_minvo_control_points,tolerance));
}


TEST(BsplineToMinvoTests, 2DFourthOrderConversion)
{
    Eigen::Matrix<double, 2, 5> bspline_control_points;
    bspline_control_points << 9, 5, 7, 8, 0,
                              2, 2, 1, 4, 4;
    Eigen::Matrix<double, 2, 5> true_minvo_control_points;
    true_minvo_control_points <<  6.13077557, 6.28454891, 6.82851097, 7.18711478, 7.13032121,
                                  1.59029616, 1.53531469, 1.71840252, 2.30466159, 2.63743928;
    Eigen::Matrix<double, 2, 5> minvo_control_points;
    BsplineToMinvo<2> converter{};
    minvo_control_points << converter.convert_4th_order_spline(bspline_control_points);
    double tolerance = 0.000001;
    EXPECT_TRUE(minvo_control_points.isApprox(true_minvo_control_points,tolerance));
}

TEST(BsplineToMinvoTests, 2DFifthOrderConversion)
{
    Eigen::Matrix<double, 2, 6> bspline_control_points;
    bspline_control_points << 3, 0, 9, 5, 7, 8,
                              1, 6, 2, 2, 1, 4;
    Eigen::Matrix<double, 2, 6> true_minvo_control_points;
    true_minvo_control_points << 6.06218264, 6.34587079, 6.66493977, 6.55522864, 6.32649018, 6.25785912,
                                 2.93333089, 2.74929314, 2.32209231, 2.01223232, 1.81847038, 1.78401508;
    Eigen::Matrix<double, 2, 6> minvo_control_points;
    BsplineToMinvo<2> converter{};
    minvo_control_points << converter.convert_5th_order_spline(bspline_control_points);
    double tolerance = 0.000001;
    EXPECT_TRUE(minvo_control_points.isApprox(true_minvo_control_points,tolerance));
}

TEST(BsplineToMinvoTests, 2DSixthOrderConversion)
{
    Eigen::Matrix<double, 2, 7> bspline_control_points;
    bspline_control_points << 1, 5, 1, 6, 5, 4, 2,
                              5, 4, 5, 6, 8, 7, 5;
    Eigen::Matrix<double, 2, 7> true_minvo_control_points;
    true_minvo_control_points << 3.63859065, 3.75563364, 4.16404993, 4.62239292, 4.9494005, 5.08393524, 5.07383627,
                                 5.50507974, 5.5790647, 5.8567374,  6.21328739, 6.5710469, 6.84266442, 6.90874509;
    Eigen::Matrix<double, 2, 7> minvo_control_points;
    BsplineToMinvo<2> converter{};
    minvo_control_points << converter.convert_6th_order_spline(bspline_control_points);
    double tolerance = 0.000001;
    EXPECT_TRUE(minvo_control_points.isApprox(true_minvo_control_points,tolerance));
}

TEST(BsplineToMinvoTests, 2DSeventhOrderConversion)
{
    Eigen::Matrix<double, 2, 8> bspline_control_points;
    bspline_control_points << 3, 0, 9, 5, 7, 8, 0, 1,
                              1, 6, 2, 2, 1, 4, 1, 4;
    Eigen::Matrix<double, 2, 8> true_minvo_control_points;
    true_minvo_control_points << 6.36171626, 6.35004406, 6.34582959, 6.3741547, 6.46486262, 6.57231754, 6.65070158, 6.66296694,
                                 1.91693037, 1.88340658, 1.81421594, 1.77023177, 1.78653576, 1.85938716, 1.95131615, 1.98847777;
    Eigen::Matrix<double, 2, 8> minvo_control_points;
    BsplineToMinvo<2> converter{};
    minvo_control_points << converter.convert_7th_order_spline(bspline_control_points);
    double tolerance = 0.000001;
    EXPECT_TRUE(minvo_control_points.isApprox(true_minvo_control_points,tolerance));
}

TEST(BsplineToMinvoTests, 3DThirdOrderConversion)
{
    Eigen::Matrix<double, 3, 4> bspline_control_points;
    bspline_control_points << 9, 0, 8, 5,
                              1, 5, 1, 6,
                              5, 4, 5, 6;
    Eigen::Matrix<double, 3, 4> true_minvo_control_points;
    true_minvo_control_points << 2.58561647, 2.77078449, 5.47854183, 6.48468216,
                                 3.78036693, 3.58901948, 2.45253491, 2.39801617,
                                 4.29289598, 4.31797103, 4.76513774, 5.06387023;
    Eigen::Matrix<double, 3,4> minvo_control_points;
    BsplineToMinvo<3> converter{};
    minvo_control_points << converter.convert_3rd_order_spline(bspline_control_points);
    double tolerance = 0.000001;
    EXPECT_TRUE(minvo_control_points.isApprox(true_minvo_control_points,tolerance));
}





