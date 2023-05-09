#ifndef CROSSTERMBOUNDS_HPP
#define CROSSTERMBOUNDS_HPP
#include <array>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "gtest/gtest_prod.h"
#include "CBindHelperFunctions.hpp"
#include "DerivativeBounds.hpp"
#include "DerivativeEvaluator.hpp"
#include "CrossTermProperties.hpp"
#include "CrossTermEvaluator.hpp"

struct DerivativeBoundsData
{
    double min_velocity;
    double max_acceleration;
    double max_cross_term;
    double time_at_min_velocity;
};

template <int D> // D is the dimension of the spline
class CrossTermBounds
{
    public:
        CrossTermBounds();
        double get_spline_curvature_bound(double cont_pts[], int num_control_points);
        double evaluate_interval_curvature_bound(Eigen::Matrix<double,D,4> &control_points);
        double get_spline_angular_rate_bound(double cont_pts[], int num_control_points, double scale_factor);
        double evaluate_interval_angular_rate_bound(Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        double get_spline_centripetal_acceleration_bound(double cont_pts[], int num_control_points, double scale_factor);
        double evaluate_interval_centripetal_acceleration_bound(Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
    private:
        CBindHelperFunctions<D> cbind_help{};
        DerivativeBounds<D> d_dt_bounds{};
        DerivativeEvaluator<D> d_dt_eval{};
        CrossTermProperties<D> c_prop{};
        CrossTermEvaluator<D> c_eval{};
        DerivativeBoundsData get_derivative_bound_data(Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        double find_maximum_cross_term(Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
    FRIEND_TEST(CrossTermTest, MaxCrossTerm);
};

extern "C"
{
    CrossTermBounds<2>* CrossTermBounds_2(){return new CrossTermBounds<2>();}
    double get_spline_curvature_bound_2(CrossTermBounds<2>* obj, double cont_pts[], int num_control_points){
        return obj->get_spline_curvature_bound(cont_pts, num_control_points);}
    double get_spline_angular_rate_bound_2(CrossTermBounds<2>* obj, double cont_pts[], int num_control_points,
        double scale_factor){return obj->get_spline_angular_rate_bound(cont_pts, num_control_points,scale_factor);}
    double get_spline_centripetal_acceleration_bound_2(CrossTermBounds<2>* obj, double cont_pts[], int num_control_points,
        double scale_factor){return obj->get_spline_centripetal_acceleration_bound(cont_pts, num_control_points,scale_factor);}
    CrossTermBounds<3>* CrossTermBounds_3(){return new CrossTermBounds<3>();}
    double get_spline_curvature_bound_3(CrossTermBounds<3>* obj, double cont_pts[], int num_control_points){
        return obj->get_spline_curvature_bound(cont_pts, num_control_points);}
    double get_spline_angular_rate_bound_3(CrossTermBounds<3>* obj, double cont_pts[], int num_control_points,
        double scale_factor){return obj->get_spline_angular_rate_bound(cont_pts, num_control_points,scale_factor);}
    double get_spline_centripetal_acceleration_bound_3(CrossTermBounds<3>* obj, double cont_pts[], int num_control_points,
        double scale_factor){return obj->get_spline_centripetal_acceleration_bound(cont_pts, num_control_points,scale_factor);}
}
#endif

