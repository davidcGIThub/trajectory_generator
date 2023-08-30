#ifndef SPHERECOLLISIONEVALUATOR_HPP
#define SPHERECOLLISIONEVALUATOR_HPP
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "CBindHelperFunctions.hpp"
#include "RotationHelperFunctions.hpp"
#include "BsplineToMinvo.hpp"

template<int D>
class SphereCollisionEvaluator
{
    public:
        SphereCollisionEvaluator();
        double* getSplineDistancesToSpheres(double obstacle_centers[], double obstacle_radii[], 
            int &num_obstacles, double cont_pts[], int &num_cont_points);
        double getSplineDistanceToSphere(double obstacle_center[], 
            double &obstacle_radius, double cont_pts[], int &num_cont_points);
        double* getAllSplineIntervalDistancesToSphere(double obstacle_center[], 
            double &obstacle_radius, double cont_pts[], int &num_cont_points);
        double getPointsDistanceToSphere(Eigen::Matrix<double,D,1> &obstacle_center, 
            double &obstacle_radius, Eigen::Matrix<double,D,4> &cont_pts);
        
    private:
        CBindHelperFunctions<D> cbind_help{};
        BsplineToMinvo<D> cp_converter{};
        RotationHelperFunctions<D> rot_helper{};
        Eigen::Matrix<double,D,D> getRotationForSphereToIntervalVector(
            Eigen::Matrix<double,D,1> &obstacle_center, Eigen::Matrix<double,D,4> &cont_pts);
        Eigen::Matrix<double,D,1> ArrayToEigenArray(double obstacle_center[]);
        Eigen::Matrix<double,D,D> get2DVectorToXDirRotation(Eigen::Matrix<double, D, 1> &vector);
        Eigen::Matrix<double,D,D> get3DVectorToXDirRotation(Eigen::Matrix<double, D, 1> &vector);
        Eigen::Matrix<double,D,1> getObstacleCenterFromArray(double obstacle_centers[], int &obstacle_num,
                                                            int &num_obstacles);
};      

#endif