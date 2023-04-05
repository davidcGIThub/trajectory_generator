#ifndef CONVEXHULLCOLLISIONCHECKER_HPP
#define CONVEXHULLCOLLISIONCHECKER_HPP
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "MDMAlgorithmClass.hpp"

template <int D>
class ConvexHullCollisionChecker
{
    public:
        ConvexHullCollisionChecker();
        double getDistanceToSphere(Eigen::Matrix<double,D,1> &sphere_center, 
                                    double &sphere_radius, Eigen::MatrixXd &spline_points, int &num_spline_points);
        double getConservativeDistanceToSphere(Eigen::Matrix<double,D,1> &sphere_center, 
                                    double &sphere_radius, Eigen::MatrixXd &points, int &num_points);
        bool checkIfCollides(Eigen::Matrix<double,D,1> &sphere_center, 
                                    double &sphere_radius, Eigen::MatrixXd &points, int &num_points);
    private:
        double getDistanceToPoint(Eigen::Matrix<double,D,1> &point, 
                                   Eigen::MatrixXd &spline_points, int &num_spline_points);
        MDMAlgorithmClass<D> mdm{};
};
#endif