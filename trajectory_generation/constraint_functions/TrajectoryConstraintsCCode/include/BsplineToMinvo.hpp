#ifndef BSPLINETOMINVO_HPP
#define BSPLINETOMINVO_HPP
#include <array>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

template <int D>
class BsplineToMinvo
{
    public:
        BsplineToMinvo();
        Eigen::Matrix<double, D,2> convert_1st_order_spline(Eigen::Matrix<double, D,2> &bspline_control_points);
        Eigen::Matrix<double, D,3> convert_2nd_order_spline(Eigen::Matrix<double, D,3> &bspline_control_points);
        Eigen::Matrix<double, D,4> convert_3rd_order_spline(Eigen::Matrix<double, D,4> &bspline_control_points);
        Eigen::Matrix<double, D,5> convert_4th_order_spline(Eigen::Matrix<double, D,5> &bspline_control_points);
        Eigen::Matrix<double, D,6> convert_5th_order_spline(Eigen::Matrix<double, D,6> &bspline_control_points);
        Eigen::Matrix<double, D,7> convert_6th_order_spline(Eigen::Matrix<double, D,7> &bspline_control_points);
        Eigen::Matrix<double, D,8> convert_7th_order_spline(Eigen::Matrix<double, D,8> &bspline_control_points);
    
    private:
        Eigen::Matrix<double,2,2> get_first_order_bspline_to_minvo_conversion_matrix();
        Eigen::Matrix<double,3,3> get_second_order_bspline_to_minvo_conversion_matrix();
        Eigen::Matrix<double,4,4> get_third_order_bspline_to_minvo_conversion_matrix();
        Eigen::Matrix<double,5,5> get_fourth_order_bspline_to_minvo_conversion_matrix();
        Eigen::Matrix<double,6,6> get_fifth_order_bspline_to_minvo_conversion_matrix();
        Eigen::Matrix<double,7,7> get_sixth_order_bspline_to_minvo_conversion_matrix();
        Eigen::Matrix<double,8,8> get_seventh_order_bspline_to_minvo_conversion_matrix();
};

#endif