#include "CubicEquationSolver.hpp"
#include <cmath>
#include <iostream>

namespace CubicEquationSolver
{

    std::array<double,3> solve_equation(double &a_term, double &b_term, double &c_term, double &d_term)
    {
        std::array<double, 3> roots;
        if (a_term == 0)
        {
            if (b_term == 0)
            {
                if (c_term == 0)
                {
                    roots = {DOUBLEMAX, DOUBLEMAX, DOUBLEMAX};
                }
                else
                {
                    double linear_root = solve_linear_equation(c_term, d_term);
                    roots = {linear_root, DOUBLEMAX, DOUBLEMAX};
                }
            }
            else
            {
                std::array<double,2> quadratic_roots = solve_quadratic_equation(b_term, c_term, d_term);
                roots = {quadratic_roots[0], quadratic_roots[1], DOUBLEMAX};
            }
        }
        else
        {
            roots = solve_cubic_equation(a_term, b_term, c_term, d_term);
        }
        return roots;
    }


    double solve_linear_equation(double &c_term, double &d_term)
    {
        double root;
        if (c_term == 0)
        {
            root = DOUBLEMAX;
        }
        else
        {
            root = -d_term/c_term; 
        }  
        return root;
    }

    std::array<double,2> solve_quadratic_equation(double &b_term, double &c_term, double &d_term)
    {
        std::array<double,2> roots;
        if (c_term*c_term - 4*b_term*d_term == 0)
        {
            double root_1 = -c_term/(2*b_term);
            roots = {root_1, DOUBLEMAX};
        }
        else if (c_term*c_term - 4*b_term*d_term < 0)
        {
            roots = {DOUBLEMAX, DOUBLEMAX};
        }
        else
        {
            double root_1 = (-c_term + sqrt(c_term*c_term - 4*b_term*d_term))/(2*b_term);
            double root_2 = (-c_term - sqrt(c_term*c_term - 4*b_term*d_term))/(2*b_term);
            roots = {root_1, root_2};
        }
        return roots;
    }

    std::array<double,3> solve_cubic_equation(double &a_term, double &b_term, double &c_term, double &d_term)
    {
        std::array<double,3> roots;
        double d = 18*a_term*b_term*c_term*d_term - 4*(b_term*b_term*b_term)*d_term + 
                    (b_term*b_term)*(c_term*c_term) - 4*a_term*(c_term*c_term*c_term) - 
                    27*(a_term*a_term)*(d_term*d_term);
        if (d > 0)
        {
            double Q_ = (3*(c_term/a_term) - pow((b_term/a_term),2))/9;
            double R_ = (9*(b_term/a_term)*(c_term/a_term) - 27*(d_term/a_term) - 2*pow(b_term/a_term,3))/54;
            double term = R_/pow(-Q_,3.0/2.0);
            double theta_ = acos(term);
            double t1 = 2*sqrt(-Q_)*cos(theta_/3) - (b_term/a_term)/3;
            double t2 = 2*sqrt(-Q_)*cos((theta_ + 2*M_PI)/3) - (b_term/a_term)/3;
            double t3 = 2*sqrt(-Q_)*cos((theta_ + 4*M_PI)/3) - (b_term/a_term)/3;
            roots = {t1,t2,t3}; 
        }
        else if (d < 0)
        {
            double P = b_term*b_term - 3*a_term*c_term;
            double Q = 9*a_term*b_term*c_term - 2*(b_term*b_term*b_term) - 27*(a_term*a_term)*d_term;
            double term_1 = Q/2 + sqrt((Q*Q)/4 - pow(P,3));
            double term_2 = Q/2 - sqrt((Q*Q)/4 - pow(P,3));
            double N = cbrt(term_1) + cbrt(term_2);
            double t1 = -b_term/(3*a_term) + N/(3*a_term);
            roots = {t1,DOUBLEMAX,DOUBLEMAX};
        }
        else
        {
            double P = b_term*b_term - 3*a_term*c_term;
            if(P == 0)
            {
                double t1 = -b_term/(3*a_term);
                roots = {t1,DOUBLEMAX,DOUBLEMAX};
            }
            else
            {
                double t1 = (9*a_term*d_term - b_term*c_term)/(2*P);
                double t2 = (4*a_term*b_term*c_term - 9*a_term*a_term*d_term - b_term*b_term*b_term)/(a_term*P);
                roots = {t1,t2,DOUBLEMAX};
            }
        }
        return roots;
    }
}