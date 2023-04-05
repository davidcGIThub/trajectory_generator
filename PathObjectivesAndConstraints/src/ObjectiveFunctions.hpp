#ifndef OBJECTIVEFUNCTIONS_HPP
#define OBJECTIVEFUNCTIONS_HPP
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "CBindingHelper.hpp"

template <int D>
class ObjectiveFunctions
{
    public:
        ObjectiveFunctions();
        double minimize_acceleration_and_time(double cont_pts[], int num_control_points, double scale_factor);
        double minimize_distance_and_time(double cont_pts[], int num_control_points, double scale_factor);
    private:
        CBindingHelper<D> cbind_help{};
        double minimize_acceleration(double cont_pts[], int &num_control_points);
        double minimize_distance(double cont_pts[], int &num_control_points);
};

extern "C"
{
    ObjectiveFunctions<2>* ObjectiveFunctions_2(){return new ObjectiveFunctions<2>();}
    double minimize_acceleration_and_time_2(ObjectiveFunctions<2>* obj, double cont_pts[], 
        int num_control_points, double scale_factor){return obj->minimize_acceleration_and_time(
            cont_pts, num_control_points, scale_factor);}
    double minimize_distance_and_time_2(ObjectiveFunctions<2>* obj, double cont_pts[], 
        int num_control_points, double scale_factor){return obj->minimize_distance_and_time(
            cont_pts, num_control_points, scale_factor);}

    ObjectiveFunctions<3>* ObjectiveFunctions_3(){return new ObjectiveFunctions<3>();}
    double minimize_acceleration_and_time_3(ObjectiveFunctions<3>* obj, double cont_pts[], 
        int num_control_points, double scale_factor){return obj->minimize_acceleration_and_time(
            cont_pts, num_control_points, scale_factor);}
    double minimize_distance_and_time_3(ObjectiveFunctions<3>* obj, double cont_pts[], 
        int num_control_points, double scale_factor){return obj->minimize_distance_and_time(
            cont_pts, num_control_points, scale_factor);}
}

#endif