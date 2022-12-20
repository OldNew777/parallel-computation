//
// Created by ChenXin on 2022/12/20.
//

#include <omp.h>

#include "log.h"
#include "timer.h"
#include "SparseMatrix.h"
#include "cxxopts.hpp"

int n, m, n_non_zero;
Real epsilon = 1e-6;
Real learning_rate = 1e-4;

int parse_args(int argc, char *argv[]) {
    cxxopts::Options options(
            "Sparse Matrix Conjugate Gradient",
            "Calculate the solution of Ax = b by conjugate gradient method, with A being a sparse matrix.");
    options.add_options()
            ("n", "Dimension n of matrix A", cxxopts::value<int>(n)->default_value("1000"))
            ("m", "Dimension m of matrix A", cxxopts::value<int>(m)->default_value("1000"))
            ("n_non_zero", "Number of non-zero values in A", cxxopts::value<int>(n_non_zero)->default_value("1000"))
            ("h,help", "Print help");
    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    return 0;
}

int main(int argc, char **argv) {
    parse_args(argc, argv);

    Timer::Tik();
    SparseMatrix A(n, m);
    A.random(n_non_zero);
    LOG_INFO("Init matrix A in %.4f seconds", Timer::Toc());

    SparseVector x(m);
    x.random();
    LOG_INFO("Init vector x in %.4f seconds", Timer::Toc());

    SparseVector b = A * x;
    LOG_INFO("Calculate vector b in %.4f seconds", Timer::Toc());

    LOG_INFO("A: %s", A.to_string().c_str());
    LOG_INFO("x: %s", x.to_string().c_str());
    LOG_INFO("b: %s", b.to_string().c_str());

    SparseVector x_star = x + 1e-3;
    SparseVector r = b - A * x_star;
    LOG_INFO("r: %s", r.to_string().c_str());
    SparseVector p = r;
    SparseVector Ap(m);
    Real r_dot_r = r.dot(r);
    Real alpha, beta;
    int iter = 0;
    while (r_dot_r > epsilon) {
        Ap = A * p;
        alpha = r_dot_r / p.dot(Ap);
//        x_star += p * (alpha * learning_rate);
        x_star += p * alpha;
        LOG_INFO("x_star = %s", x_star.to_string().c_str());
        r -= Ap * alpha;
        Real r_dot_r_new = r.dot(r);
        beta = r_dot_r_new / r_dot_r;
        p = r + p * beta;
        r_dot_r = r_dot_r_new;
        ++iter;
    }
    LOG_INFO("Conjugate gradient finished in %d iterations in %.4f seconds", iter, Timer::Toc());
    LOG_INFO("x = %s", x.to_string().c_str());
    LOG_INFO("x_star = %s", x_star.to_string().c_str());

    return 0;
}