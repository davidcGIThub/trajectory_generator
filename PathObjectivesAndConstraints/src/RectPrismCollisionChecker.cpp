#include "RectPrismCollisionChecker.hpp"
#include <iostream>

template <int D>
RectPrismCollisionChecker<D>::RectPrismCollisionChecker(){}

template <int D>
double RectPrismCollisionChecker<D>::getDistanceToObstacle(Eigen::Matrix<double,D,1> obstacle_center, 
                             double obstacle_radius, Eigen::MatrixXd points, int num_points)
{
    Eigen::Matrix<double,D,1> min_vector = getMinOfPoints(points, num_points);
    Eigen::Matrix<double,D,1> max_vector = getMaxOfPoints(points, num_points);
    double distanceToObstacle = getDistanceToObstacle(obstacle_center, 
                            obstacle_radius, min_vector, max_vector);
    return distanceToObstacle;
}

template <int D>
double RectPrismCollisionChecker<D>::getDistanceToObstacle(Eigen::Matrix<double,D,1> obstacle_center, 
                             double obstacle_radius, Eigen::Matrix<double,D,1> min_vector,
                             Eigen::Matrix<double,D,1> max_vector)
{
    double distance_to_center = getDistanceFromCenterToPrism(obstacle_center, 
                                                      max_vector, min_vector);
    double distance_to_obstacle = distance_to_center - obstacle_radius;
    return distance_to_obstacle;
}
 
template <int D>
std::vector<std::array<Eigen::Matrix<double,D,1>, 2>>
    RectPrismCollisionChecker<D>::convertPointsListToPrismList(
        std::vector<Eigen::MatrixXd> point_set_list, int num_point_sets, 
        int num_points_per_set)
{
    std::vector<std::array<Eigen::Matrix<double,D,1>,2>> prism_list;
    for (unsigned int i = 0; i < num_point_sets; i++)
    {
        Eigen::Matrix<double,D,1> max_vector = getMaxOfPoints(point_set_list[i], num_points_per_set);
        Eigen::Matrix<double,D,1> min_vector = getMinOfPoints(point_set_list[i], num_points_per_set);
        std::array<Eigen::Matrix<double,D,1>, 2> prism = {min_vector, max_vector};
        prism_list.push_back(prism);
    }
    return prism_list;
}

template <int D>
Eigen::Matrix<double,D,1> RectPrismCollisionChecker<D>::getMaxOfPoints(Eigen::MatrixXd points, int num_points)
{
    Eigen::Matrix<double,D,1> maxVector;
    if (D == 2)
    {
        maxVector << points.block(0,0,1,num_points).maxCoeff() , points.block(1,0,1,num_points).maxCoeff();
    }
    if (D == 3)
    {
        maxVector << points.block(0,0,1,num_points).maxCoeff(),
                     points.block(1,0,1,num_points).maxCoeff(),
                     points.block(2,0,1,num_points).maxCoeff();
    }
    return maxVector;
}

template <int D>
Eigen::Matrix<double,D,1> RectPrismCollisionChecker<D>::getMinOfPoints(Eigen::MatrixXd points, int num_points)
{
    Eigen::Matrix<double,D,1> minVector;
    if (D == 2)
    {
        minVector << points.block(0,0,1,num_points).minCoeff() , points.block(1,0,1,num_points).minCoeff();
    }
    if (D == 3)
    {
        minVector << points.block(0,0,1,num_points).minCoeff(),
                     points.block(1,0,1,num_points).minCoeff(),
                     points.block(2,0,1,num_points).minCoeff();
    }
    return minVector;
}

template <int D>
double RectPrismCollisionChecker<D>::getDistanceFromCenterToPrism(Eigen::Matrix<double,D,1> obstacle_center, 
                                                      Eigen::Matrix<double,D,1> max_vector, 
                                                      Eigen::Matrix<double,D,1> min_vector)
{
    Eigen::Matrix<double,D,3> temp_matrix;
    Eigen::Matrix<double,D,1> zero = Eigen::Matrix<double,D,1>::Zero();
    temp_matrix << obstacle_center - max_vector, min_vector - obstacle_center, zero;
    int cols = 3;
    Eigen::Matrix<double,D,1> distance_vector = getMaxOfPoints(temp_matrix, cols);
    double distance = distance_vector.norm();
    return distance;
}

// explicit instantiation
template class RectPrismCollisionChecker<2>;
template class RectPrismCollisionChecker<3>;