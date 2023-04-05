#include "CurvatureConstraints.hpp"

template <int D>
CurvatureConstraints<D>::CurvatureConstraints()
{

}

template <int D>
double CurvatureConstraints<D>::get_spline_curvature_constraint(double cont_pts[], 
    int num_cont_pts, double max_curvature)
{
    double constraint;
    constraint = curv_bound.get_spline_curvature_bound(cont_pts, num_cont_pts) - max_curvature;
    return constraint;
}

template <int D>
double* CurvatureConstraints<D>::get_interval_curvature_constraints(double cont_pts[], int num_cont_pts, 
    double max_curvature)
{
    int order = 3;
    int num_intervals = num_cont_pts - order;
    double* constraints = new double[num_intervals];
    Eigen::VectorXd curvature_bounds(num_intervals);
    curvature_bounds = curv_bound.get_interval_curvature_bounds(cont_pts, num_cont_pts);
    for(unsigned int i = 0; i < num_intervals; i++)
    {
        constraints[i] = curvature_bounds.coeff(i) - max_curvature;
    }
    return constraints;
}

// explicit instantiation
template class CurvatureConstraints<2>;
template class CurvatureConstraints<3>;