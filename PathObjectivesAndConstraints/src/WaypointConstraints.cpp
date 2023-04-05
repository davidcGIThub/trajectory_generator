#include "WaypointConstraints.hpp"
#include <iostream>

template<int D>
WaypointConstraints<D>::WaypointConstraints()
{

}

template<int D>
double* WaypointConstraints<D>::velocity_at_waypoints_constraints(double cont_pts[], int num_cont_pts, 
            double scale_factor, double desired_velocities[], bool switches[])
{
    int order = 3;
    double* constraints = new double[D*2];
    DerivativeEvaluator<D> d_dt_eval{};
    CBindingHelper<D> helper{};
    double t0 = 0;
    double tf = scale_factor;
    unsigned int start_index = 0;
    unsigned int end_index = num_cont_pts - order - 1;
    if (switches[0] == true)
    {
        Eigen::Matrix<double,D,4> initial_control_points = helper.array_section_to_eigen(cont_pts, num_cont_pts, start_index);
        Eigen::Matrix<double,D,1> velocity_vector_0 = d_dt_eval.calculate_velocity_vector(t0, initial_control_points, scale_factor);
        for(unsigned int i = 0; i < D; i++)
        {
            constraints[2*i] = velocity_vector_0.coeff(i) - desired_velocities[2*i];
        }
    }
    else
    {
        for(unsigned int i = 0; i < D; i++)
        {
            constraints[2*i] = 0;
        }
    }
    if (switches[1] == true)
    {
        Eigen::Matrix<double,D,4> final_control_points = helper.array_section_to_eigen(cont_pts, num_cont_pts, end_index);
        Eigen::Matrix<double,D,1> velocity_vector_f = d_dt_eval.calculate_velocity_vector(tf, final_control_points, scale_factor);
        for(unsigned int i = 0; i < D; i++)
        {
            constraints[2*i+1] = velocity_vector_f.coeff(i) - desired_velocities[2*i+1];
        }
    }
    else
    {
        for(unsigned int i = 0; i < D; i++)
        {
            constraints[2*i+1] = 0;
        }

    }
    return constraints;
}

template<int D>
double* WaypointConstraints<D>::acceleration_at_waypoints_constraints(double cont_pts[], 
    int num_cont_pts, double scale_factor, double desired_accelerations[])
{
    int order = 3;
    double* constraints = new double[D*2];
    DerivativeEvaluator<D> d_dt_eval{};
    CBindingHelper<D> helper{};
    double t0 = 0;
    double tf = scale_factor;
    unsigned int start_index = 0;
    unsigned int end_index = num_cont_pts - order - 1;
    Eigen::Matrix<double,D,4> initial_control_points = helper.array_section_to_eigen(cont_pts, num_cont_pts, start_index);
    Eigen::Matrix<double,D,4> final_control_points = helper.array_section_to_eigen(cont_pts, num_cont_pts, end_index);
    Eigen::Matrix<double,D,1> acceleration_vector_0 = d_dt_eval.calculate_acceleration_vector(t0, initial_control_points, scale_factor);
    Eigen::Matrix<double,D,1> acceleration_vector_f = d_dt_eval.calculate_acceleration_vector(tf, final_control_points, scale_factor);
    for(unsigned int i = 0; i < D; i++)
    {
        constraints[2*i] = acceleration_vector_0.coeff(i) - desired_accelerations[2*i];
        constraints[2*i+1] = acceleration_vector_f.coeff(i) - desired_accelerations[2*i+1];
    }
    return constraints;
}

template<int D>
double* WaypointConstraints<D>::curvature_at_waypoints_constraints(double cont_pts[], 
            int num_cont_pts, double scale_factor, double desired_curvatures[], bool switches[])
{
    int order = 3;
    double* constraints = new double[2];
    DerivativeEvaluator<D> d_dt_eval{};
    CBindingHelper<D> helper{};
    double t0 = 0;
    double tf = scale_factor;
    unsigned int start_index = 0;
    unsigned int end_index = num_cont_pts - order - 1;
    if (switches[0] == true)
    {
        Eigen::Matrix<double,D,4> initial_control_points = helper.array_section_to_eigen(cont_pts, num_cont_pts, start_index);
        Eigen::Matrix<double,D,1> acceleration_vector_0 = d_dt_eval.calculate_acceleration_vector(t0, initial_control_points, scale_factor);
        Eigen::Matrix<double,D,1> velocity_vector_0 = d_dt_eval.calculate_velocity_vector(t0, initial_control_points, scale_factor);
        double curvature_0 = helper.curvature_calculation(velocity_vector_0, acceleration_vector_0);
        constraints[0] = curvature_0 - desired_curvatures[0];
    }
    else
    {
        constraints[0] = 0;
    }
    if (switches[1] == true)
    {
        Eigen::Matrix<double,D,4> final_control_points = helper.array_section_to_eigen(cont_pts, num_cont_pts, end_index);
        Eigen::Matrix<double,D,1> acceleration_vector_f = d_dt_eval.calculate_acceleration_vector(tf, final_control_points, scale_factor);
        Eigen::Matrix<double,D,1> velocity_vector_f = d_dt_eval.calculate_velocity_vector(tf, final_control_points, scale_factor);
        double curvature_f =  helper.curvature_calculation(velocity_vector_f, acceleration_vector_f);
        constraints[1] = curvature_f - desired_curvatures[1];
    }
    else
    {
        constraints[1] = 0;
    }
    return constraints;
}

template<int D>
double* WaypointConstraints<D>::direction_at_waypoints_constraints(double cont_pts[], 
            int num_cont_pts, double scale_factor, double desired_directions[])
{
    int order = 3;
    double* constraints = new double[D*2];
    DerivativeEvaluator<D> d_dt_eval{};
    CBindingHelper<D> helper{};
    double t0 = 0;
    double tf = scale_factor;
    unsigned int start_index = 0;
    unsigned int end_index = num_cont_pts - order - 1;
    Eigen::Matrix<double,D,4> initial_control_points = helper.array_section_to_eigen(cont_pts, num_cont_pts, start_index);
    Eigen::Matrix<double,D,1> velocity_vector_0 = d_dt_eval.calculate_velocity_vector(t0, initial_control_points, scale_factor);
    Eigen::Matrix<double,D,1> direction_vector_0 = velocity_vector_0/velocity_vector_0.norm();
    Eigen::Matrix<double,D,4> final_control_points = helper.array_section_to_eigen(cont_pts, num_cont_pts, end_index);
    Eigen::Matrix<double,D,1> velocity_vector_f = d_dt_eval.calculate_velocity_vector(tf, final_control_points, scale_factor);
    Eigen::Matrix<double,D,1> direction_vector_f = velocity_vector_f/velocity_vector_f.norm();
    for(unsigned int i = 0; i < D; i++)
    {
        constraints[2*i] = direction_vector_0.coeff(i) - desired_directions[2*i];
        constraints[2*i+1] = direction_vector_f.coeff(i) - desired_directions[2*i+1];
    }
    return constraints;
}

//explicit instantiation
template class WaypointConstraints<2>;
template class WaypointConstraints<3>;
