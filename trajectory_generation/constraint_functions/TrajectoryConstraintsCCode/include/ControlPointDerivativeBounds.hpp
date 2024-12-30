#ifndef DERIVATIVEBOUNDS_HPP
#define DERIVATIVEBOUNDS_HPP
#include <array>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "gtest/gtest_prod.h"
#include "CBindHelperFunctions.hpp"
#include "MDMAlgorithmClass.hpp"

template <int D> // D is the dimension of the spline
class ControlPointDerivativeBounds
{
    public:
        ControlPointDerivativeBounds();
        double find_min_velocity_of_bez_vel_cont_pts(double bez_vel_cont_pts[], int num_bez_vel_cont_pts);
        double find_min_velocity_hull_method(Eigen::Matrix<double,D,3> &interval_control_points);
    private:
        CBindHelperFunctions<D> cbind_help{};
        std::array<double,3> get_velocity_roots(Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        MDMAlgorithmClass<D,3> mdm_obj{};
};


extern "C"
{
    ControlPointDerivativeBounds<2>* ControlPointDerivativeBounds_2(){return new ControlPointDerivativeBounds<2>();}
    double find_min_velocity_of_bez_vel_cont_pts_2(ControlPointDerivativeBounds<2>* obj, 
                    double bez_vel_cont_pts[], int num_bez_vel_cont_pts){
        return obj->find_min_velocity_of_bez_vel_cont_pts(bez_vel_cont_pts, num_bez_vel_cont_pts);}
    ControlPointDerivativeBounds<3>* ControlPointDerivativeBounds_3(){return new ControlPointDerivativeBounds<3>();}
    double find_min_velocity_of_bez_vel_cont_pts_3(ControlPointDerivativeBounds<3>* obj, 
                    double bez_vel_cont_pts[], int num_bez_vel_cont_pts){
        return obj->find_min_velocity_of_bez_vel_cont_pts(bez_vel_cont_pts, num_bez_vel_cont_pts);}
}


#endif
