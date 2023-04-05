#include "MDMAlgorithmClass.hpp"
#include "gtest/gtest.h"
#include <iostream>
#include <array>
#include <chrono>
#include <ctime>  

TEST(MDMAlgorithmTests, MDMClassRandomCase1)
{
    int num_points = 6;
    const int dimension = 2;
    Eigen::MatrixXd points(dimension,num_points);
    points << 8, 5, 13, 10,  8, 14,
             13, 9, 10, 10, 12, 10;
    int max_iter = 5000;
    unsigned int init_index = 1;
    double true_min_norm = 10.295630140987;
    double tolerance = 0.000001;
    // auto start = std::chrono::system_clock::now();
    MDMAlgorithmClass<dimension> mdm{};
    double min_norm = mdm.min_norm(points,num_points, max_iter, init_index, tolerance);
    // auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end-start;
    // std::cout << "elapsed time class: " << elapsed_seconds.count() << "s" << std::endl;
    double error_tolerance = 0.00000001;
    EXPECT_NEAR(min_norm, true_min_norm, tolerance);
}

TEST(MDMAlgorithmTests, MDMRandomCase2)
{
    int num_points = 4;
    const int dimension = 2;
    Eigen::MatrixXd points(dimension,num_points);
    points << 10,  8, 14, 5,
               6, 10,  8, 9; 
    int max_iter = 5000;
    unsigned int init_index = 1;
    double true_min_norm = 10.289915108550531;
    double tolerance = 0.000001;
    MDMAlgorithmClass<dimension> mdm{};
    double min_norm = mdm.min_norm(points,num_points, max_iter, init_index, tolerance);
    double error_tolerance = 0.00000001;
    EXPECT_NEAR(true_min_norm, min_norm, error_tolerance);
}

TEST(MDMAlgorithmTests, MDMStraightLine)
{
    int num_points = 6;
    const int dimension = 2;
    Eigen::MatrixXd points(dimension,num_points);
    points << 1,2,3,4,5,6,
              1,2,3,4,5,6; 
    int max_iter = 5000;
    unsigned int init_index = 1;
    double true_min_norm = 1.4142135623730951;
    double tolerance = 0.000001;
    MDMAlgorithmClass<dimension> mdm{};
    double min_norm = mdm.min_norm(points,num_points, max_iter, init_index, tolerance);
    double error_tolerance = 0.00000001;
    EXPECT_NEAR(true_min_norm, min_norm, error_tolerance);
}

TEST(MDMAlgorithmTests, MDMStraightLine2)
{
    int num_points = 6;
    const int dimension = 2;
    Eigen::MatrixXd points(dimension,num_points); 
    points << 6,5,4,3,2,1,
              1,2,3,4,5,6; 
    int max_iter = 5000;
    unsigned int init_index = 1;
    double true_min_norm = 4.949747468305833;
    double tolerance = 0.000001;
    MDMAlgorithmClass<dimension> mdm{};
    double min_norm = mdm.min_norm(points,num_points, max_iter, init_index, tolerance);
    double error_tolerance = 0.00000001;
    EXPECT_NEAR(true_min_norm, min_norm, error_tolerance);
}

TEST(MDMAlgorithmTests, MDMSamePoints)
{
    int num_points = 6;
    const int dimension = 2;
    Eigen::MatrixXd points(dimension,num_points);
    points << 5, 5, 5, 5, 5, 5,
              6, 6, 6, 6, 6, 6;
    int max_iter = 5000;
    double tolerance = 0.000001;
    unsigned int init_index = 1;
    double true_min_norm = 7.810249675906654;
    MDMAlgorithmClass<dimension> mdm{};
    double min_norm = mdm.min_norm(points,num_points, max_iter, init_index, tolerance);
    double error_tolerance = 0.00000001;
    EXPECT_NEAR(true_min_norm, min_norm, error_tolerance);
}

TEST(MDMAlgorithmTests, RandomCase3)
{
    int num_points = 6;
    const int dimension = 2;
    Eigen::MatrixXd points(dimension,num_points);
    points << 9, 6, 12, 14, 13, 7,
              5, 13, 11, 11, 8, 8; 
    int max_iter = 5000;
    double tolerance = 0.000001;
    unsigned int init_index = 1;
    double true_min_norm = 10.261953630166738;
    MDMAlgorithmClass<dimension> mdm{};
    double min_norm = mdm.min_norm(points,num_points, max_iter, init_index, tolerance);
    double error_tolerance = 0.00000001;
    EXPECT_NEAR(true_min_norm, min_norm, error_tolerance);
}


TEST(MDMAlgorithmTests, RandomCase3D_1)
{
    int num_points = 10;
    const int dimension = 3;
    Eigen::MatrixXd points(dimension,num_points);
    points << 9,  6, 12, 14, 13,  7, 10,  8, 14,  5,
              5, 13, 11, 11,  8,  8,  6, 10,  8,  9,
              6,  5,  4,  3,  8,  5, 13, 10,  8, 14; 
    int max_iter = 5000;
    double tolerance = 0.000001;
    unsigned int init_index = 1;
    double true_min_norm = 11.671087598447691;
    MDMAlgorithmClass<dimension> mdm{};
    double min_norm = mdm.min_norm(points,num_points, max_iter, init_index, tolerance);
    double error_tolerance = 0.00000001;
    EXPECT_NEAR(true_min_norm, min_norm, error_tolerance);
}
