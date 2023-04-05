#include "ObstacleConstraints.hpp"
#include <iostream>

template <int D>
ObstacleConstraints<D>::ObstacleConstraints()
{

}

template <int D>
double* ObstacleConstraints<D>::getObstacleDistancesToSpline(double cont_pts[], int num_control_points,
                                double obstacle_radii[], double obstacle_centers[], unsigned int num_obstacles)
{
    const int order = 3;
    double distance_to_obstacle{0};
    double* distances_array = new double[num_obstacles];
    Eigen::MatrixXd control_points(D,num_control_points);
    control_points = helper.array_to_eigen(cont_pts, num_control_points);
    for (unsigned int i = 0; i < num_obstacles; i++)
    {
        double obstacle_radius = obstacle_radii[i];
        Eigen::Matrix<double,D,1> obstacle_center = getObstacleCenterFromArray(
                obstacle_centers, i, num_obstacles);
        double distance_to_obstacle = collision_checker.getConservativeDistanceToSphere(obstacle_center, obstacle_radius,
                control_points, num_control_points); 
        
        if (distance_to_obstacle < 0)
        {
            distance_to_obstacle = getDistanceToClosestInterval(control_points, num_control_points,
                                obstacle_radius, obstacle_center);
        }
        distances_array[i] = -distance_to_obstacle;
    }
    return distances_array;
}

template <int D>
double ObstacleConstraints<D>::getObstacleDistanceToSpline(double cont_pts[], int num_control_points,
                                double obstacle_radius, double obstacle_center[])
{
    const int order = 3;
    Eigen::MatrixXd control_points(D,num_control_points);
    control_points = helper.array_to_eigen(cont_pts, num_control_points);
    unsigned int obstacle_index = 0;
    unsigned int num_obstacles = 1;
    Eigen::Matrix<double,D,1> obstacle_center_ = getObstacleCenterFromArray(obstacle_center, obstacle_index,
                                                num_obstacles);
    double distance_to_obstacle = collision_checker.getConservativeDistanceToSphere(obstacle_center_, obstacle_radius,
                control_points, num_control_points); 
    if (distance_to_obstacle < 0)
    {
        distance_to_obstacle = getDistanceToClosestInterval(control_points, num_control_points,
                                obstacle_radius, obstacle_center_);
    }
    return -distance_to_obstacle;
}

template <int D>
bool* ObstacleConstraints<D>::checkIfObstaclesCollide(double cont_pts[], int num_control_points,
        double obstacle_radii[], double obstacle_centers[], unsigned int num_obstacles)
{
    const int order = 3;
    const int NUM_POINTS = order+1;
    int num_points = NUM_POINTS;
    bool collision = false;
    Eigen::MatrixXd bspline_points(D,num_control_points);
    bspline_points = helper.array_to_eigen(cont_pts,num_control_points);
    Eigen::MatrixXd minvo_points;
    Eigen::Matrix<double,D,1> obstacle_center;
    bool* collides = new bool[num_obstacles];
    Eigen::Matrix<double,D,NUM_POINTS> bspline_point_section;
    double obstacle_radius;
    for(unsigned int j = 0; j < num_obstacles; j++)
    {
        obstacle_center = getObstacleCenterFromArray(obstacle_centers, j, num_obstacles);
        obstacle_radius = obstacle_radii[j];
        collides[j] = false;
        for (unsigned int i = 0; i < num_control_points-order; i++)
        {
            bspline_point_section = bspline_points.block(0,i,D,num_points);
            //faster if dont have to compute minvo points every interation
            minvo_points = cp_converter.convert_3rd_order_spline(bspline_point_section);
            if (collision_checker.checkIfCollides(obstacle_center, obstacle_radius, minvo_points, num_points))
            {
                collides[j] = true;
                break;
            }
        }
    }
    return collides;

}

template <int D>
bool ObstacleConstraints<D>::checkIfObstacleCollides(double cont_pts[], int num_control_points,
        double obstacle_radius, double obstacle_center[])
{
    int order = 3;
    int num_points = order+1;
    bool collision = false;
    Eigen::Matrix<double,D,4> bspline_points;
    Eigen::MatrixXd minvo_points;
    unsigned int obstacle_index = 0;
    unsigned int num_obstacles = 1;
    Eigen::Matrix<double,D,1> obstacle_center_ = getObstacleCenterFromArray(obstacle_center, obstacle_index, num_obstacles);
    bool collides = false;
    for (unsigned int i = 0; i < num_control_points-order; i++)
    {
        bspline_points = helper.array_section_to_eigen(cont_pts, num_control_points, i);
        minvo_points = cp_converter.convert_3rd_order_spline(bspline_points);
        if (collision_checker.checkIfCollides(obstacle_center_, obstacle_radius, minvo_points, num_points))
        {
            collides = true;
            break;
        }
    }
    return collides;
}

template <int D>
double ObstacleConstraints<D>::getDistanceToClosestInterval(Eigen::MatrixXd &control_points, int &num_control_points,
                                double &obstacle_radius, Eigen::Matrix<double,D,1> &obstacle_center)
{ 
    const int order = 3;
    const int NUM_POINTS = order+1;
    int num_points = NUM_POINTS;
    double min_distance{std::numeric_limits<double>::max()};
    double distance = 0;

    for (unsigned int i = 0; i < num_control_points-order-1; i++)
    {
        Eigen::Matrix<double, D,NUM_POINTS> bspline_points = control_points.block(0,i,D,order+1);
        Eigen::MatrixXd minvo_points(D,NUM_POINTS);
        minvo_points = cp_converter.convert_3rd_order_spline(bspline_points);
        distance = collision_checker.getDistanceToSphere(obstacle_center, obstacle_radius,
                minvo_points, num_points);
        if (distance < min_distance)
        {
            min_distance = distance;
        }
    }
    return min_distance;
}

template <int D>
Eigen::Matrix<double,D,1> ObstacleConstraints<D>::getObstacleCenterFromArray(double obstacle_centers[], unsigned int &obstacle_num,
                                                            unsigned int &num_obstacles)
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
template class ObstacleConstraints<2>;
template class ObstacleConstraints<3>;
