#ifndef CUBICEQUATIONSOLVER_HPP
#define CUBICEQUATIONSOLVER_HPP
#include <array>
#include <list>
#include <limits>

namespace CubicEquationSolver
{
    std::array<double,3> solve_equation(double &a_term, double &b_term, double &c_term, double &d_term);

    std::array<double,3> solve_cubic_equation(double &a_term, double &b_term, double &c_term, double &d_term);

    std::array<double,2> solve_quadratic_equation(double &b_term, double &c_term, double &d_term);

    double solve_linear_equation(double &c_term, double &d_term);

    static const double DOUBLEMAX = std::numeric_limits<double>::max();
}

#endif