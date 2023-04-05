#ifndef RECTPRISMCOLLISIONCHECKER_HPP
#define RECTPRISMCOLLISIONCHECKER_HPP
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

template<int D>
class RectPrismCollisionChecker
{
    public:
        RectPrismCollisionChecker();
        double getDistancesToObstacle();
        double getDistanceToObstacle(Eigen::Matrix<double,D,1> obstacle_center, 
                                    double obstacle_radius, Eigen::MatrixXd points, 
                                    int num_points);
        double getDistanceToObstacle(Eigen::Matrix<double,D,1> obstacle_center, 
                             double obstacle_radius, Eigen::Matrix<double,D,1> min_vector,
                             Eigen::Matrix<double,D,1> max_vector);
        std::vector<std::array<Eigen::Matrix<double,D,1>, 2>>
            convertPointsListToPrismList(std::vector<Eigen::MatrixXd> point_set_list, 
            int num_point_sets, int num_points_per_set);
    private:
        Eigen::Matrix<double,D,1> getMaxOfPoints(Eigen::MatrixXd points, int num_points);
        Eigen::Matrix<double,D,1> getMinOfPoints(Eigen::MatrixXd points, int num_points);
        double getDistanceFromCenterToPrism(Eigen::Matrix<double,D,1> obstacle_center, 
                                            Eigen::Matrix<double,D,1> max_vector, 
                                            Eigen::Matrix<double,D,1> min_vector);
};      

#endif