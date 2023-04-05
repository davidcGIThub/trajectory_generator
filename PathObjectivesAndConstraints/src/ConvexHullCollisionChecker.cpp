#include "ConvexHullCollisionChecker.hpp"
#include <iostream>

template<int D>
ConvexHullCollisionChecker<D>::ConvexHullCollisionChecker(){}

template<int D>
bool ConvexHullCollisionChecker<D>::checkIfCollides(Eigen::Matrix<double,D,1> &sphere_center, 
                                                double &sphere_radius, Eigen::MatrixXd &points, int &num_points)
{
    double distance_to_point = getDistanceToPoint(sphere_center, points, num_points);
    if (distance_to_point < sphere_radius + 0.000001)
    {
        return true;
    }
    else
    {
        return false;
    }
}

template<int D>
double ConvexHullCollisionChecker<D>::getDistanceToSphere(Eigen::Matrix<double,D,1> &sphere_center, 
                                                double &sphere_radius, Eigen::MatrixXd &points, int &num_points)
{
    double distance_to_point = getDistanceToPoint(sphere_center, points, num_points);
    double distance_to_obstacle;
    if(distance_to_point < 0.000001)
    {
        Eigen::Matrix<double,D,1> mean_of_points = points.rowwise().mean();
        double points_pseudo_radius = ((points.colwise() - mean_of_points).colwise().norm()).maxCoeff();
        double points_mean_to_sphere_center = (mean_of_points - sphere_center).norm();
        distance_to_obstacle = distance_to_point - sphere_radius + points_mean_to_sphere_center
            - points_pseudo_radius;
    }
    else
    {
        distance_to_obstacle = distance_to_point - sphere_radius;
    }
    return distance_to_obstacle;
}

template<int D>
double ConvexHullCollisionChecker<D>::getConservativeDistanceToSphere(Eigen::Matrix<double,D,1> &sphere_center, 
                                                double &sphere_radius, Eigen::MatrixXd &points, int &num_points)
{
    double distance_to_point = getDistanceToPoint(sphere_center, points, num_points);
    double distance_to_obstacle;
    if(distance_to_point < (sphere_radius + 0.000001))
    {
        distance_to_obstacle = -sphere_radius;
    }
    else
    {
        Eigen::Matrix<double,D,1> mean_of_points = points.rowwise().mean();
        double points_pseudo_radius = ((points.colwise() - mean_of_points).colwise().norm()).maxCoeff();
        distance_to_obstacle = distance_to_point - sphere_radius + points_pseudo_radius;
    }
    return distance_to_obstacle;
}

template<int D>
double ConvexHullCollisionChecker<D>::getDistanceToPoint(Eigen::Matrix<double,D,1> &sphere_center, 
                          Eigen::MatrixXd &points, int &num_points)
{
    Eigen::MatrixXd translated_points = points.colwise() - sphere_center;
    int max_iter = 5000;
    unsigned int init_index = 1;
    double tolerance = 0.000001;
    double distance_to_point = mdm.min_norm(translated_points, num_points, max_iter, 
                                            init_index, tolerance);
    return distance_to_point;
}

// explicit instantiation
template class ConvexHullCollisionChecker<2>;
template class ConvexHullCollisionChecker<3>;