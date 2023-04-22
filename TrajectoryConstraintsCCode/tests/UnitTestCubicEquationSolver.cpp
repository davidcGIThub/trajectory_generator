#include "gtest/gtest.h"
#include "CubicEquationSolver.hpp"
#include <chrono>
#include <ctime>  

TEST(CubicSolverTests, LinearProblem1)
{
    double c_term = 2;
    double d_term = 3.2;
    double true_answer = -1.6;
    double answer = CubicEquationSolver::solve_linear_equation(c_term, d_term);
    EXPECT_EQ(answer , true_answer);
}

TEST(CubicSolverTests, LinearProblem2)
{
    double c_term = 0;
    double d_term = 3.2;
    double true_answer = CubicEquationSolver::DOUBLEMAX;
    double answer = CubicEquationSolver::solve_linear_equation(c_term, d_term);
    EXPECT_EQ(answer , true_answer);
}

TEST(CubicSolverTests, QuadraticProblem1)
{
    double b_term = 3;
    double c_term = 6;
    double d_term = 3;
    std::array<double,2> true_answer = {-1.0,CubicEquationSolver::DOUBLEMAX};
    std::array<double,2> answer = CubicEquationSolver::solve_quadratic_equation(b_term,c_term,d_term);
    EXPECT_EQ(answer, true_answer);
}

TEST(CubicSolverTests, QuadraticProblem2)
{
    double b_term = 3;
    double c_term = 6;
    double d_term = 8;
    std::array<double,2> true_answer = {CubicEquationSolver::DOUBLEMAX,CubicEquationSolver::DOUBLEMAX};
    std::array<double,2> answer = CubicEquationSolver::solve_quadratic_equation(b_term,c_term,d_term);
    EXPECT_EQ(answer, true_answer);
}

TEST(CubicSolverTests, QuadraticProblem3)
{
    double b_term = 3;
    double c_term = 6;
    double d_term = 2;
    std::array<double,2> true_answer = {-0.42264973081037427, -1.5773502691896255};
    std::array<double,2> answer = CubicEquationSolver::solve_quadratic_equation(b_term,c_term,d_term);
    EXPECT_EQ(answer, true_answer);
}

TEST(CubicSolverTests, CubicProblem1)
{
    double a_term = 4;
    double b_term = 3;
    double c_term = -6;
    double d_term = -2;
    std::array<double,3> true_answer = {1.075972408704097, -1.520314680974536, -0.3056577277295605};
    // auto start = std::chrono::system_clock::now();
    std::array<double,3> answer = CubicEquationSolver::solve_cubic_equation(a_term,b_term,c_term,d_term);
    // auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end-start;
    // std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
    EXPECT_NEAR(answer[0], true_answer[0],0.000001);
    EXPECT_NEAR(answer[1], true_answer[1],0.000001);
    EXPECT_NEAR(answer[2], true_answer[2],0.000001);
}

TEST(CubicSolverTests, CubicProblem2)
{
    double a_term = 4;
    double b_term = 3;
    double c_term = 6;
    double d_term = 2;
    std::array<double,3> true_answer = {-0.3678020738373728, CubicEquationSolver::DOUBLEMAX,CubicEquationSolver::DOUBLEMAX};
    std::array<double,3> answer = CubicEquationSolver::solve_cubic_equation(a_term,b_term,c_term,d_term);
    EXPECT_NEAR(answer[0], true_answer[0], 0.0000000001);
    EXPECT_NEAR(answer[1], true_answer[1], 0.0000000001);
    EXPECT_NEAR(answer[2], true_answer[2], 0.0000000001);
}

TEST(CubicSolverTests, CubicProblem3)
{
    double a_term = 4;
    double b_term = 4;
    double c_term = 1;
    double d_term = 0;
    std::array<double,3> true_answer = {-0.5, 0.0, CubicEquationSolver::DOUBLEMAX};
    std::array<double,3> answer = CubicEquationSolver::solve_cubic_equation(a_term,b_term,c_term,d_term);
    EXPECT_NEAR(answer[0], true_answer[0],0.0000001);
    EXPECT_NEAR(answer[1], true_answer[1],0.0000001);
    EXPECT_NEAR(answer[2], true_answer[2],0.0000001);
}

TEST(CubicSolverTests, CubicProblem4)
{
    double a_term = 3;
    double b_term = 6;
    double c_term = 4;
    double d_term = 0.8888888888888888;
    std::array<double,3> true_answer = {-0.6666666666666666, CubicEquationSolver::DOUBLEMAX, CubicEquationSolver::DOUBLEMAX};
    std::array<double,3> answer = CubicEquationSolver::solve_cubic_equation(a_term,b_term,c_term,d_term);
    EXPECT_EQ(answer, true_answer);
}


TEST(CubicSolverTests, SolveEquation1)
{
    double a_term = 0;
    double b_term = 0;
    double c_term = 0;
    double d_term = 0;
    std::array<double,3> true_answer = {CubicEquationSolver::DOUBLEMAX, CubicEquationSolver::DOUBLEMAX, CubicEquationSolver::DOUBLEMAX};
    std::array<double,3> answer = CubicEquationSolver::solve_equation(a_term,b_term,c_term,d_term);
    EXPECT_EQ(answer, true_answer);
}

TEST(CubicSolverTests, SolveEquation2)
{
    double a_term = 0;
    double b_term = 0;
    double c_term = 2;
    double d_term = 3.2;
    std::array<double,3> true_answer = {-1.6, CubicEquationSolver::DOUBLEMAX, CubicEquationSolver::DOUBLEMAX};
    std::array<double,3> answer = CubicEquationSolver::solve_equation(a_term,b_term,c_term,d_term);
    EXPECT_EQ(answer, true_answer);
}

TEST(CubicSolverTests, SolveEquation3)
{
    double a_term = 0;
    double b_term = 3;
    double c_term = 6;
    double d_term = 3;
    std::array<double,3> true_answer = {-1.0, CubicEquationSolver::DOUBLEMAX, CubicEquationSolver::DOUBLEMAX};
    std::array<double,3> answer = CubicEquationSolver::solve_equation(a_term,b_term,c_term,d_term);
    EXPECT_EQ(answer, true_answer);
}

TEST(CubicSolverTests, SolveEquation4)
{
    double a_term = 4;
    double b_term = 3;
    double c_term = -6;
    double d_term = -2;
    std::array<double,3> true_answer = {1.075972408704097, -1.520314680974536, -0.3056577277295605};
    std::array<double,3> answer = CubicEquationSolver::solve_equation(a_term,b_term,c_term,d_term);
    EXPECT_NEAR(answer[0], true_answer[0],0.0000001);
    EXPECT_NEAR(answer[1], true_answer[1],0.0000001);
    EXPECT_NEAR(answer[2], true_answer[2],0.0000001);
}

