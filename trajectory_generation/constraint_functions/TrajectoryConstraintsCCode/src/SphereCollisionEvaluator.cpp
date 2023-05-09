#include "SphereCollisionEvaluator.hpp"
#include <math.h>  
#include <iostream>
#include <list>

template <int D>
SphereCollisionEvaluator<D>::SphereCollisionEvaluator()
{

}

template <int D>
double* SphereCollisionEvaluator<D>::getSplineDistancesToSpheres(double obstacle_centers[], double obstacle_radii[], 
    int &num_obstacles, double cont_pts[], int &num_cont_points)
{
    const int order = 3;
    int num_intervals = num_cont_points - order;
    double distance_to_obstacle{0};
    double* distances_array = new double[num_obstacles];
    std::list<Eigen::Matrix<double,D,4>> minvo_list;
    
    for(unsigned int j = 0; j < num_intervals; j++)
    {
        Eigen::Matrix<double,D,4> interval_control_points = cbind_help.array_section_to_eigen(cont_pts, num_cont_points, j);
        Eigen::Matrix<double,D,4> minvo_control_points = cp_converter.convert_3rd_order_spline(interval_control_points);
        minvo_list.push_back(minvo_control_points);
    }
    for (int i = 0; i < num_obstacles; i++)
    {
        double obstacle_radius = obstacle_radii[i];
        Eigen::Matrix<double,D,1> obstacle_center = getObstacleCenterFromArray(
                obstacle_centers, i, num_obstacles);
        double min_distance = std::numeric_limits<double>::max(); 
        for (auto minvo_cont_pts : minvo_list) 
        {
            double distance = getPointsDistanceToSphere(obstacle_center, obstacle_radius, minvo_cont_pts);
            if (min_distance > distance)
            {
                min_distance = distance;
            }
        }
        distances_array[i] = min_distance;
    }
    return distances_array;
}

template <int D>
double SphereCollisionEvaluator<D>::getSplineDistanceToSphere(double obstacle_center[], 
            double &obstacle_radius, double cont_pts[], int &num_cont_points)
{
    int order = 3;
    int num_intervals = num_cont_points - order;
    double min_distance = std::numeric_limits<double>::max();
    Eigen::Matrix<double,D,1> obstacle_center_ = ArrayToEigenArray(obstacle_center);
    
    for (unsigned int i = 0; i < num_intervals; i++)
    {
        Eigen::Matrix<double,D,4> interval_control_points = cbind_help.array_section_to_eigen(cont_pts, num_cont_points, i);
        Eigen::Matrix<double,D,4> minvo_control_points = cp_converter.convert_3rd_order_spline(interval_control_points);
        double distance = getPointsDistanceToSphere(obstacle_center_, obstacle_radius, minvo_control_points);
        if (min_distance > distance)
        {
            min_distance = distance;
        }
    }
    return min_distance;
}

template <int D>
double*  SphereCollisionEvaluator<D>::getAllSplineIntervalDistancesToSphere(double obstacle_center[], 
    double &obstacle_radius, double cont_pts[], int &num_cont_points)
{
    int order = 3;
    int num_intervals = num_cont_points - order;
    double* distances_to_obstacle = new double[num_intervals];
    Eigen::Matrix<double,D,1> obstacle_center_ = ArrayToEigenArray(obstacle_center);
    
    for (unsigned int i = 0; i < num_intervals; i++)
    {
        Eigen::Matrix<double,D,4> interval_control_points = cbind_help.array_section_to_eigen(cont_pts, num_cont_points, i);
        Eigen::Matrix<double,D,4> minvo_control_points = cp_converter.convert_3rd_order_spline(interval_control_points);
        distances_to_obstacle[i] = getPointsDistanceToSphere(obstacle_center_, obstacle_radius, minvo_control_points);
    }
    return distances_to_obstacle;
}

template <int D>
double SphereCollisionEvaluator<D>::getPointsDistanceToSphere(Eigen::Matrix<double,D,1> &obstacle_center, 
    double &obstacle_radius, Eigen::Matrix<double,D,4> &cont_pts)
{
    Eigen::Matrix<double,D,D> rotation = getRotationForSphereToIntervalVector(obstacle_center, cont_pts);
    Eigen::Matrix<double,D,4> rotated_points = rotation*cont_pts;
    double points_min_x = rotated_points.block(0,0,1,4).minCoeff();
    Eigen::Matrix<double,D,1> rotated_sphere_center = rotation*obstacle_center;
    double sphere_max_x = rotated_sphere_center.coeff(0) + obstacle_radius;
    double distance = points_min_x - sphere_max_x;
    return distance;
}

template <int D>
Eigen::Matrix<double,D,D> SphereCollisionEvaluator<D>::getRotationForSphereToIntervalVector(Eigen::Matrix<double,D,1> &obstacle_center,
    Eigen::Matrix<double,D,4> &cont_pts)
{
    Eigen::Matrix<double, D, 1> mean_pts = cont_pts.rowwise().mean();
    Eigen::Matrix<double, D, 1> vector = mean_pts - obstacle_center;
    Eigen::Matrix<double, D, D> rotation;
    if (D == 2)
    {
        if (vector(0) == 0 && vector(1) == 0)
        {
            rotation << 1 , 0, 
                        0 , 1;
        }
        else
        {
            rotation = get2DVectorToXDirRotation(vector);
        }
    }
    else //(D == 3)
    {
        if (vector(0) == 0 && vector(1) == 0 && vector(2) == 0)
        {
            rotation << 1 , 0 , 0, 
                        0, 1, 0, 
                        0, 0, 1;
        }
        else
        {
            rotation = get3DVectorToXDirRotation(vector);
        }
    }
    return rotation;
}

template <int D>
Eigen::Matrix<double,D,D> SphereCollisionEvaluator<D>::get2DVectorToXDirRotation(Eigen::Matrix<double, D, 1> &vector)
{
    double psi = -rot_helper.get2DXDirToVectorAngle(vector);
    Eigen::Matrix<double,D,D> rotation = rot_helper.get_2D_Z_rotation(psi);
    return rotation;
}

template <int D>
Eigen::Matrix<double,D,D> SphereCollisionEvaluator<D>::get3DVectorToXDirRotation(Eigen::Matrix<double, D, 1> &vector)
{
    Eigen::Matrix<double, 2, 1> angles = rot_helper.get3DXDirToVectorAngles(vector);
    double theta = -angles.coeff(1);
    double psi = -angles.coeff(0);
    Eigen::Matrix<double,D,D> R_y = rot_helper.get_3D_Y_rotation(theta);
    Eigen::Matrix<double,D,D> R_z = rot_helper.get_3D_Z_rotation(psi);
    Eigen::Matrix<double,D,D> rotation = R_z * R_y;
    return rotation;
}

template <int D>
Eigen::Matrix<double,D,1> SphereCollisionEvaluator<D>::ArrayToEigenArray(double obstacle_center[])
{
    Eigen::Matrix<double,D,1> obstacle_center_;
    if (D == 2)
    {
        obstacle_center_ << obstacle_center[0], obstacle_center[1];
    }
    else //D == 3
    {
        obstacle_center_ << obstacle_center[0], obstacle_center[1], obstacle_center[2];
    }
    return obstacle_center_;
}

template <int D>
Eigen::Matrix<double,D,1> SphereCollisionEvaluator<D>::getObstacleCenterFromArray(double obstacle_centers[], int &obstacle_num,
                                                            int &num_obstacles)
{
    Eigen::Matrix<double,D,1> obstacle_center;
    if (D == 2)
    {
        obstacle_center << obstacle_centers[obstacle_num], obstacle_centers[obstacle_num+num_obstacles];
    }
    else //D == 3
    {
        obstacle_center << obstacle_centers[obstacle_num], obstacle_centers[obstacle_num+num_obstacles], 
            obstacle_centers[obstacle_num+num_obstacles*2];
    }
    return obstacle_center;
}

// explicit instantiation
template class SphereCollisionEvaluator<2>;
template class SphereCollisionEvaluator<3>;