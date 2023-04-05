#ifndef OBSTACLECONSTRAINTS_HPP
#define OBSTACLECONSTRAINTS_HPP
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "ConvexHullCollisionChecker.hpp"
#include "CBindingHelper.hpp"
#include "BsplineToMinvo.hpp"

template <int D>
class ObstacleConstraints
{
    public:
        ObstacleConstraints();
        double* getObstacleDistancesToSpline(double cont_pts[], int num_control_points,
                                          double obstacle_radii[], double obstacle_centers[],
                                          unsigned int num_obstacles);
        double getObstacleDistanceToSpline(double cont_pts[], int num_control_points,
                                double obstacle_radius, double obstacle_center[]);
        bool* checkIfObstaclesCollide(double cont_pts[], int num_control_points,
                                          double obstacle_radii[], double obstacle_centers[],
                                          unsigned int num_obstacles);
        bool checkIfObstacleCollides(double cont_pts[], int num_control_points,
                                double obstacle_radius, double obstacle_center[]);
    private:
        ConvexHullCollisionChecker<D> collision_checker{};
        BsplineToMinvo<D> cp_converter{};
        CBindingHelper<D> helper{};
        double getDistanceToClosestInterval(Eigen::MatrixXd &control_points, int &num_control_points,
                                double &obstacle_radius, Eigen::Matrix<double,D,1> &obstacle_center);
        Eigen::Matrix<double,D,1> getObstacleCenterFromArray(double obstacle_centers[], unsigned int &obstacle_num,
                                                            unsigned int &num_obstacles);
};

extern "C"
{
    ObstacleConstraints<2>* ObstacleConstraints_2(){return new ObstacleConstraints<2>();}
    bool checkIfObstacleCollides_2(ObstacleConstraints<2>* ex, double cont_pts[], int num_control_points,
                                double obstacle_radius, double obstacle_center[]){return ex->checkIfObstacleCollides(cont_pts,
                                num_control_points,obstacle_radius,obstacle_center);}
    double getObstacleDistanceToSpline_2(ObstacleConstraints<2>* ex, double cont_pts[], int num_control_points,
                                double obstacle_radius, double obstacle_center[]){return ex->getObstacleDistanceToSpline(cont_pts,
                                num_control_points,obstacle_radius,obstacle_center);}
    double* getObstacleDistancesToSpline_2(ObstacleConstraints<2>* ex, double cont_pts[], int num_control_points,
                                double obstacle_radii[], double obstacle_centers[], unsigned int num_obstacles)
                                {return ex->getObstacleDistancesToSpline(cont_pts,num_control_points,obstacle_radii,
                                obstacle_centers, num_obstacles);}

    ObstacleConstraints<3>* ObstacleConstraints_3(){return new ObstacleConstraints<3>();}
    bool checkIfObstacleCollides_3(ObstacleConstraints<3>* ex, double cont_pts[], int num_control_points,
                                double obstacle_radius, double obstacle_center[]){return ex->checkIfObstacleCollides(cont_pts,
                                num_control_points,obstacle_radius,obstacle_center);}
    double getObstacleDistanceToSpline_3(ObstacleConstraints<3>* ex, double cont_pts[], int num_control_points,
                                double obstacle_radius, double obstacle_center[]){return ex->getObstacleDistanceToSpline(cont_pts,
                                num_control_points,obstacle_radius,obstacle_center);}
    double* getObstacleDistancesToSpline_3(ObstacleConstraints<3>* ex, double cont_pts[], int num_control_points,
                                double obstacle_radii[], double obstacle_centers[], unsigned int num_obstacles)
                                {return ex->getObstacleDistancesToSpline(cont_pts,num_control_points,obstacle_radii,
                                obstacle_centers, num_obstacles);}
}

#endif