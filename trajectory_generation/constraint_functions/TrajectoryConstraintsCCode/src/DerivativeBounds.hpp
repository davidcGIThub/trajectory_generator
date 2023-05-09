#ifndef DERIVATIVEBOUNDS_HPP
#define DERIVATIVEBOUNDS_HPP
#include <array>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "gtest/gtest_prod.h"
#include "CBindHelperFunctions.hpp"
#include "DerivativeEvaluator.hpp"
#include "CubicEquationSolver.hpp"

template <int D> // D is the dimension of the spline
class DerivativeBounds
{
    public:
        DerivativeBounds();
        double find_min_velocity_of_spline(double cont_pts[], int num_control_points, double scale_factor);
        double find_max_acceleration_of_spline(double cont_pts[], int num_control_points, double scale_factor);
        std::array<double,2> find_max_velocity_and_time(Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        std::array<double,2> find_min_velocity_and_time(Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        std::array<double,2> find_max_acceleration_and_time(Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
        double find_max_velocity_magnitude_in_single_dimension(Eigen::Matrix<double,D,4> &control_points, double &scale_factor, unsigned int &dimension);
    private:
        CBindHelperFunctions<D> cbind_help{};
        DerivativeEvaluator<D> d_eval{};
        DerivativeEvaluator<1> d_eval_single{};
        std::array<double,3> get_velocity_roots(Eigen::Matrix<double,D,4> &control_points, double &scale_factor);
};
#endif

