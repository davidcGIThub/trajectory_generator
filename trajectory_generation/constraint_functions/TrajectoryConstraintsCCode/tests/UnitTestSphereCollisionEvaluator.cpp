#include "gtest/gtest.h"
#include "SphereCollisionEvaluator.hpp"

TEST(SphereCollisionEvaluatorTests, NotColliding2D)
{
    const int D{2};
    SphereCollisionEvaluator<D> obstacle_dist_eval{};
    Eigen::Matrix<double,D,1> obstacle_center;
    obstacle_center << 7, 9;
    double obstacle_radius = 2; 
    Eigen::Matrix<double, D, 4> cont_pts; 
    cont_pts << 1, 3,  -1,  2,
                2, 5, 6.9, -4;
    double distance = obstacle_dist_eval.getPointsDistanceToSphere(obstacle_center, obstacle_radius, cont_pts);
    double true_distance = 3.6456131350136847;
    double tolerance = 0.0000000001;
    EXPECT_NEAR(true_distance, distance, tolerance);
}

TEST(SphereCollisionEvaluatorTests, Colliding2D)
{
    const int D{2};
    SphereCollisionEvaluator<D> obstacle_dist_eval{};
    Eigen::Matrix<double,D,1> obstacle_center;
    obstacle_center << 1, 3;
    double obstacle_radius = 2; 
    Eigen::Matrix<double, D, 4> cont_pts; 
    cont_pts << 1, 3,  -1,  2,
              2, 5, 6.9, -4;
    double distance = obstacle_dist_eval.getPointsDistanceToSphere(obstacle_center, obstacle_radius, cont_pts);
    double true_distance = -6.381023184198024;
    double tolerance = 0.0000000001;
    EXPECT_NEAR(true_distance, distance, tolerance);
}

TEST(SphereCollisionEvaluatorTests, NotColliding3D)
{
    const int D{3};
    SphereCollisionEvaluator<D> obstacle_dist_eval{};
    Eigen::Matrix<double,D,1> obstacle_center;
    obstacle_center << 7, 9, 12;
    double obstacle_radius = 2; 
    Eigen::Matrix<double, D, 4> cont_pts; 
    cont_pts << 1, 3,  -1,  2,
              2, 5, 6.9, -4,
              3, 4, 8, 2;
    double distance = obstacle_dist_eval.getPointsDistanceToSphere(obstacle_center, obstacle_radius, cont_pts);
    double true_distance = 5.786249815167853;
    double tolerance = 0.0000000001;
    EXPECT_NEAR(true_distance, distance, tolerance);
}

TEST(SphereCollisionEvaluatorTests, NotColliding3DUnder)
{
    const int D{3};
    SphereCollisionEvaluator<D> obstacle_dist_eval{};
    Eigen::Matrix<double,D,1> obstacle_center;
    obstacle_center << -7, -9, -12;
    double obstacle_radius = 2; 
    Eigen::Matrix<double, D, 4> cont_pts; 
    cont_pts << 1, 3,  -1,  2,
              2, 5, 6.9, -4,
              3, 4, 8, 2;
    double distance = obstacle_dist_eval.getPointsDistanceToSphere(obstacle_center, obstacle_radius, cont_pts);
    double true_distance = 14.675542601750479;
    double tolerance = 0.0000000001;
    EXPECT_NEAR(true_distance, distance, tolerance);
}

TEST(SphereCollisionEvaluatorTests, Colliding3D)
{
    const int D{3};
    SphereCollisionEvaluator<D> obstacle_dist_eval{};
    Eigen::Matrix<double,D,1> obstacle_center;
    obstacle_center << 1, 1, 5;
    double obstacle_radius = 2; 
    Eigen::Matrix<double, D, 4> cont_pts; 
    cont_pts << 1, 3,  -1,  2,
              2, 5, 6.9, -4,
              3, 4, 8, 2;
    double distance = obstacle_dist_eval.getPointsDistanceToSphere(obstacle_center, obstacle_radius, cont_pts);
    double true_distance = -4.9130446368042975;
    double tolerance = 0.0000000001;
    EXPECT_NEAR(true_distance, distance, tolerance);
}


TEST(SphereCollisionEvaluatorTests, AllIntervalsSplineNotColliding2D)
{
    const int D{2};
    SphereCollisionEvaluator<D> obstacle_dist_eval{};
    double obstacle_center[] = {7, 9};
    double obstacle_radius = 2; 
    double cont_pts[] = {1, 3,  -1,  2, 4,
                        2, 5, 6.9, -4, 7};
    int num_control_points = 5;
    double* distances = obstacle_dist_eval.getAllSplineIntervalDistancesToSphere(obstacle_center, 
            obstacle_radius, cont_pts, num_control_points);
    double true_distance_1 = 4.3100344057807334;
    double true_distance_2 = 5.3919541197808476;
    double tolerance = 0.0000000001;
    EXPECT_NEAR(true_distance_1, distances[0], tolerance);
    EXPECT_NEAR(true_distance_2, distances[1], tolerance);
}

TEST(SphereCollisionEvaluatorTests, AllIntervalsNotColliding3D)
{
    const int D{3};
    SphereCollisionEvaluator<D> obstacle_dist_eval{};
    double obstacle_center[] = {7, 9, 12};
    double obstacle_radius = 2; 
    double cont_pts[] = {1, 3,  -1,  2, 4,
                        2, 5, 6.9, -4, 7,
                        3, 4, 8, 2, 8.5};
    int num_control_points = 5;
    double* distances = obstacle_dist_eval.getAllSplineIntervalDistancesToSphere(obstacle_center, 
            obstacle_radius, cont_pts, num_control_points);
    double true_distance_1 = 7.1033445550584702;
    double true_distance_2 = 7.1877666609532103;
    double tolerance = 0.0000000001;
    EXPECT_NEAR(true_distance_1, distances[0], tolerance);
    EXPECT_NEAR(true_distance_2, distances[1], tolerance);
}

TEST(SphereCollisionEvaluatorTests, WholeSplineNotColliding3D)
{
    const int D{3};
    SphereCollisionEvaluator<D> obstacle_dist_eval{};
    double obstacle_center[] = {7, 9, 12};
    double obstacle_radius = 2; 
    double cont_pts[] = {1, 3,  -1,  2, 4,
                        2, 5, 6.9, -4, 7,
                        3, 4, 8, 2, 8.5};
    int num_control_points = 5;
    double distance = obstacle_dist_eval.getSplineDistanceToSphere(obstacle_center, 
            obstacle_radius, cont_pts, num_control_points);
    double true_distance = 7.1033445550584702;
    double tolerance = 0.0000000001;
    EXPECT_NEAR(true_distance, distance, tolerance);
}

TEST(SphereCollisionEvaluatorTests, WholeSplineNotColliding2D)
{
    const int D{2};
    SphereCollisionEvaluator<D> obstacle_dist_eval{};
    double obstacle_center[] = {7, 9};
    double obstacle_radius = 2; 
    double cont_pts[] = {1, 3,  -1,  2, 4,
                        2, 5, 6.9, -4, 7};
    int num_control_points = 5;
    double distance = obstacle_dist_eval.getSplineDistanceToSphere(obstacle_center, 
            obstacle_radius, cont_pts, num_control_points);
    double true_distance = 4.3100344057807334;
    double tolerance = 0.0000000001;
    EXPECT_NEAR(true_distance, distance, tolerance);
}

TEST(SphereCollisionEvaluatorTests, WholeSplineMultipleSpheres)
{
    const int D{3};
    SphereCollisionEvaluator<D> obstacle_dist_eval{};
    double obstacle_center[] = {7 ,  -7,
                                9 ,  -9,
                                12, -12};
    double obstacle_radii[] = {2, 2}; 
    int num_obstacles = 2;
    double cont_pts[] = {1, 3,  -1,  2, 4,
                        2, 5, 6.9, -4, 7,
                        3, 4, 8, 2, 8.5};
    int num_control_points = 5;
    double* distances = obstacle_dist_eval.getSplineDistancesToSpheres(obstacle_center, 
            obstacle_radii, num_obstacles, cont_pts, num_control_points);
    double true_distance_1 = 7.1033445550584702;
    double true_distance_2 = 17.900516160476471;
    double tolerance = 0.0000000001;
    EXPECT_NEAR(true_distance_1, distances[0], tolerance);
    EXPECT_NEAR(true_distance_2, distances[1], tolerance);
}