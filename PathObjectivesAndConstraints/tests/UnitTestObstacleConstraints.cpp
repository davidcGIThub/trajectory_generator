#include "gtest/gtest.h"
#include "ObstacleConstraints.hpp"

TEST(ObstacleConstraintTests, CenterInMinvoHull)
{
    const int D{2};
    ObstacleConstraints<D> obst_const{};
    double sphere_center[] = {4.84435679, 6.42836434};
    double sphere_radius = 1; 
    const int num_spline_points = 10;
    double control_points[] = {7.91705873, 9.88263331, 0.27303466, 7.50604049, 4.61073475, 5.98801717, 1.52432928, 3.8850049, 1.61195392, 8.22471529,
        5.22947263, 1.33282499, 3.51583204, 8.62435967, 3.03096953, 0.84672315, 0.54028843, 7.24686189, 4.79897482, 5.00498365};
    double distance = obst_const.getObstacleDistanceToSpline(control_points, num_spline_points, sphere_radius, sphere_center);
    double true_distance = 0.9999843316663152;
    double tolerance = 0.0001;
    EXPECT_NEAR(true_distance, distance, tolerance);
}

TEST(ObstacleConstraintTests, RadiusInMinvoHull)
{
    const int D{2};
    ObstacleConstraints<D> obst_const{};
    double sphere_center[] = {4.59242023, 2.80862623};
    double sphere_radius = 1; 
    const int num_spline_points = 10;
    double control_points[] = {3.51490868, 9.93890562, 3.18371347, 4.82774563, 1.53139681, 0.38652503, 6.0471705,  0.16208627, 4.40754293, 3.47274037,
        3.91808237, 8.35404157, 1.90443825, 8.44816466, 4.16849762, 3.4327385, 1.76263076, 5.31268297, 2.22008256, 2.8775095};
    double distance = obst_const.getObstacleDistanceToSpline(control_points, num_spline_points, sphere_radius, sphere_center);
    double true_distance = 0.6514747211179606;
    double tolerance = 0.000001;
    EXPECT_NEAR(true_distance, distance, tolerance);
}



TEST(ObstacleConstraintTests, RadiusInBsplineHull)
{
    const int D{2};
    ObstacleConstraints<D> obst_const{};
    double sphere_center[] = {5.51107017, 1.05467685};
    double sphere_radius = 1; 
    const int num_spline_points = 10;
    double control_points[] = {2.61021784, 5.11996892, 2.16685915, 2.01382634, 8.37519743, 8.03594715, 2.67299177, 0.73316133, 7.69503321, 6.12743855,
        1.76549201, 9.58027961, 2.61914916, 9.97791921, 1.69992255, 2.74704919, 8.39148742, 7.41307135, 6.71635575, 3.61696069};
    double distance = obst_const.getObstacleDistanceToSpline(control_points, num_spline_points, sphere_radius, sphere_center);
    double true_distance = -1.7486032163129486;
    double tolerance = 0.000001;
    EXPECT_NEAR(true_distance, distance, tolerance);
}

TEST(ObstacleConstraintTests, PointInBsplineHull)
{
    const int D{2};
    ObstacleConstraints<D> obst_const{};
    double sphere_center[] = {2.09598594 , 5.31302225};
    double sphere_radius = 1; 
    const int num_spline_points = 10;
    double control_points[] = {1.71441477, 3.6139517,  5.30094221, 8.37323056, 9.6235351,  3.90841661, 1.99899417, 8.67304047, 5.21592068, 2.55523126,
        8.09595843, 1.83482911, 7.77011536, 6.37508643, 5.37388355, 0.48746415, 4.62696183, 1.95456998, 7.39982416, 9.53329381};
    double distance = obst_const.getObstacleDistanceToSpline(control_points, num_spline_points, sphere_radius, sphere_center);
    double true_distance = -1.0435314502170652;
    double tolerance = 0.000001;
    EXPECT_NEAR(true_distance, distance, tolerance);
}

TEST(ObstacleConstraintTests, OutsideBsplineHull)
{
    const int D{2};
    ObstacleConstraints<D> obst_const{};
    double sphere_center[] = {-1.8820112, -0.61966279};
    double sphere_radius = 1; 
    const int num_spline_points = 10;
    double control_points[] = {5.40876234, 3.59385918, 8.64430584, 1.54778121, 3.28521738, 6.45670928, 2.4775997,  2.13578183, 5.73122222, 4.73792694,
        6.48052042, 6.70371021, 1.27331269, 0.54655433, 1.03721891, 0.41984464, 2.40628853, 3.16038393, 8.05965801, 9.55043722};
    double distance = obst_const.getObstacleDistanceToSpline(control_points, num_spline_points, sphere_radius, sphere_center);
    double true_distance = -8.219382383246899;
    double tolerance = 0.000001;
    EXPECT_NEAR(true_distance, distance, tolerance);
}

TEST(ObstacleConstraintTests, ThreeDimensionTest)
{
    const int D{3};
    ObstacleConstraints<D> obst_const{};
    double sphere_center[] = {1.97355228, 4.03641385, 2.56398793};
    double sphere_radius = 1; 
    const int num_spline_points = 10;
    double control_points[] = {0.76042949, 3.42851596, 7.12733203, 5.40578535, 6.19092292, 4.19655381, 7.35779894, 4.71393053, 8.11393068, 1.14161887,
        1.10422243, 1.74691534, 7.83946679, 6.62205838, 4.60485943, 2.11879405, 3.1969421,  6.9578278,  9.98252687, 7.9863814,
        4.56558915, 0.69220484, 4.01782909, 8.88922696, 3.95016982, 9.36658683, 2.52569157, 7.46387504, 3.91443386, 2.48021325};
    double distance = obst_const.getObstacleDistanceToSpline(control_points, num_spline_points, sphere_radius, sphere_center);
    double true_distance = -1.1481243240992596;
    double tolerance = 0.000001;
    EXPECT_NEAR(true_distance, distance, tolerance);
}

TEST(ObstacleConstraintTests, TestMultipleObstacles)
{
    const int D{2};
    ObstacleConstraints<D> obst_const{};
    const int num_spline_points = 10;
    double control_points[] = {8.33665694, 8.74600799, 0.15411698, 1.03944841, 2.71745767, 5.10167975, 0.49748729, 9.24934927, 2.09060337, 0.3531816,
        1.40865962, 5.09002261, 8.24619956, 0.06025564, 7.38338304, 6.72558684, 7.26305315, 8.93062143, 2.91758113, 6.65393249};
    double obstacle_radii[] = {1, 1, 1};
    int num_obstacles = 3;
    double obstacle_centers[] = {2.6662532, -1.14261837, 3.31630941, 
                                6.16021986, 0.02562916, 0.98248971};
    double* distances = obst_const.getObstacleDistancesToSpline(control_points, num_spline_points,
                                          obstacle_radii, obstacle_centers, num_obstacles);
    double true_distance_1 = 2.6063083134151492;
    double true_distance_2 = -7.605461310820617;
    double true_distance_3 = -1.4706446781371505;
    double tolerance = 0.000001;
    EXPECT_NEAR(true_distance_1, distances[0], tolerance);
    EXPECT_NEAR(true_distance_2, distances[1], tolerance);
    EXPECT_NEAR(true_distance_3, distances[2], tolerance);
}

TEST(ObstacleConstraintTests, CheckIfCollides)
{
    const int D{2};
    ObstacleConstraints<D> obst_const{};
    double sphere_center[] = {4.84435679, 6.42836434};
    double sphere_radius = 1; 
    const int num_spline_points = 10;
    double control_points[] = {7.91705873, 9.88263331, 0.27303466, 7.50604049, 4.61073475, 5.98801717, 1.52432928, 3.8850049, 1.61195392, 8.22471529,
        5.22947263, 1.33282499, 3.51583204, 8.62435967, 3.03096953, 0.84672315, 0.54028843, 7.24686189, 4.79897482, 5.00498365};
    bool collides = obst_const.checkIfObstacleCollides(control_points, num_spline_points, sphere_radius, sphere_center);
    double true_collides = true;
    EXPECT_EQ(true_collides, collides);
}

TEST(ObstacleConstraintTests, CheckIfCollidesDoesnt)
{
    const int D{2};
    ObstacleConstraints<D> obst_const{};
     double sphere_center[] = {5.51107017, 1.05467685};
    double sphere_radius = 1; 
    const int num_spline_points = 10;
    double control_points[] = {2.61021784, 5.11996892, 2.16685915, 2.01382634, 8.37519743, 8.03594715, 2.67299177, 0.73316133, 7.69503321, 6.12743855,
        1.76549201, 9.58027961, 2.61914916, 9.97791921, 1.69992255, 2.74704919, 8.39148742, 7.41307135, 6.71635575, 3.61696069};
    bool collides = obst_const.checkIfObstacleCollides(control_points, num_spline_points, sphere_radius, sphere_center);
    double true_collides = false;
    EXPECT_EQ(true_collides, collides);
}

TEST(ObstacleConstraintTests, CheckMultipleObstacles)
{
    const int D{2};
    ObstacleConstraints<D> obst_const{};
    const int num_spline_points = 10;
    double control_points[] = {8.33665694, 8.74600799, 0.15411698, 1.03944841, 2.71745767, 5.10167975, 0.49748729, 9.24934927, 2.09060337, 0.3531816,
        1.40865962, 5.09002261, 8.24619956, 0.06025564, 7.38338304, 6.72558684, 7.26305315, 8.93062143, 2.91758113, 6.65393249};
    double obstacle_radii[] = {1, 1, 1};
    int num_obstacles = 3;
    double obstacle_centers[] = {2.6662532, -1.14261837, 3.31630941, 
                                6.16021986, 0.02562916, 0.98248971};
    bool* collisions = obst_const.checkIfObstaclesCollide(control_points, num_spline_points,
                                          obstacle_radii, obstacle_centers, num_obstacles);
    EXPECT_EQ(true, collisions[0]);
    EXPECT_EQ(false, collisions[1]);
    EXPECT_EQ(false, collisions[2]);
}