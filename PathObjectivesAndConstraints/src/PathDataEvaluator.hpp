#ifndef PATHDATAEVALUATOR_HPP
#define PATHDATAEVALUATOR_HPP
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

template <int D>
class PathDataEvaluator
{
    public:
        PathDataEvaluator();
        float* getClosestPositionAndDirectionVector(float control_points[], 
            int num_control_points, float point[]);

    private:
        Eigen::Matrix<float,D,4> getControlPointsOfClosestInterval(Eigen::MatrixXd &control_points, 
            Eigen::Matrix<float,D,1> point);
        float getTParameterOfClosestPoint(Eigen::Matrix<float,D,4> &interval_control_points, 
            Eigen::Matrix<float,D,1> point);
        Eigen::Matrix<float,D,1> getPositionVector(float &t, Eigen::Matrix<float,D,4> &interval_control_points);
        Eigen::Matrix<float,D,1> getDirectionVector(float &t, Eigen::Matrix<float,D,4> &interval_control_points);
        Eigen::Matrix<float,D,1> getVelocityVector(float &t, Eigen::Matrix<float,D,4> &interval_control_points);
        FRIEND_TEST(PathDataEvaluatorTest, ControlPointsOfClosestInterval);
        FRIEND_TEST(PathDataEvaluatorTest, TParameterOfClosestPoint);
};

#endif