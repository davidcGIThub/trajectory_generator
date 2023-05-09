#ifndef OBSTACLECONSTRAINTS_HPP
#define OBSTACLECONSTRAINTS_HPP
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "CBindHelperFunctions.hpp"
#include "SphereCollisionEvaluator.hpp"
#include <iostream>

template <int D>
class ObstacleConstraints
{
    public:
        ObstacleConstraints();
        double* getObstaclesConstraintsForSpline(double obstacle_centers[], double obstacle_radii[], 
            int num_obstacles, double cont_pts[], int num_cont_points);
        double* getObstacleConstraintsForIntervals(double cont_pts[], int num_control_points,
                                double obstacle_radius, double obstacle_center[]);
        double getObstacleConstraintForSpline(double cont_pts[], int num_control_points,
                                double obstacle_radius, double obstacle_center[]);
    private:
        CBindHelperFunctions<D> helper{};
        SphereCollisionEvaluator<D> colision_eval{};
};

extern "C"
{
    ObstacleConstraints<2>* ObstacleConstraints_2(){return new ObstacleConstraints<2>();}
    double* getObstaclesConstraintsForSpline_2(ObstacleConstraints<2>* obj, double obstacle_centers[], double obstacle_radii[], 
            int num_obstacles, double cont_pts[], int num_cont_points){
                return obj->getObstaclesConstraintsForSpline(
        obstacle_centers, obstacle_radii, num_obstacles, cont_pts, num_cont_points);}
    double* getObstacleConstraintsForIntervals_2(ObstacleConstraints<2>* obj, double cont_pts[], int num_control_points,
        double obstacle_radius, double obstacle_center[]){return obj->getObstacleConstraintsForIntervals(
        cont_pts, num_control_points, obstacle_radius, obstacle_center);}
    double getObstacleConstraintForSpline_2(ObstacleConstraints<2>* obj, double cont_pts[], int num_control_points,
        double obstacle_radius, double obstacle_center[]){return obj->getObstacleConstraintForSpline(
        cont_pts, num_control_points, obstacle_radius, obstacle_center);}

    ObstacleConstraints<3>* ObstacleConstraints_3(){return new ObstacleConstraints<3>();}
    double* getObstaclesConstraintsForSpline_3(ObstacleConstraints<3>* obj, double obstacle_centers[], double obstacle_radii[], 
            int num_obstacles, double cont_pts[], int num_cont_points){return obj->getObstaclesConstraintsForSpline(
        obstacle_centers, obstacle_radii, num_obstacles, cont_pts, num_cont_points);}
    double* getObstacleConstraintsForIntervals_3(ObstacleConstraints<3>* obj, double cont_pts[], int num_control_points,
        double obstacle_radius, double obstacle_center[]){return obj->getObstacleConstraintsForIntervals(
        cont_pts, num_control_points, obstacle_radius, obstacle_center);}
    double getObstacleConstraintForSpline_3(ObstacleConstraints<3>* obj, double cont_pts[], int num_control_points,
        double obstacle_radius, double obstacle_center[]){return obj->getObstacleConstraintForSpline(
        cont_pts, num_control_points, obstacle_radius, obstacle_center);}
}

#endif