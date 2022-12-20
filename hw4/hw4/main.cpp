//
// Created by ChenXin on 2022/12/20.
//

#include <omp.h>

#include "log.h"
#include "timer.h"
#include "SparseMatrix.h"
#include "cxxopts.hpp"

int n, m, n_non_zero;

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

//    SparseVector x_star = x + 0.1;
    SparseVector x_star(A.m);
    SparseVector r = b - A * x_star;
    SparseVector p = r;
    SparseVector Ap(m);
    Real rr = r.dot(r);
    Real alpha, beta;
    int iter = 0;
    while (sqrt(rr) > Config::epsilon * sqrt(x_star.dot(x_star))) {
//        LOG_DEBUG("p: %s", p.to_string().c_str());
//        LOG_DEBUG("rr: %f", rr);
        Ap = A * p;
//        LOG_DEBUG("Ap: %s", Ap.to_string().c_str());
        alpha = rr / p.dot(Ap);
//        LOG_DEBUG("alpha: %f", alpha);
//        x_star += p * (alpha * Config::lr);
        x_star += p * alpha;
        LOG_DEBUG("x_star = %s", x_star.to_string().c_str());
        r -= Ap * alpha;
//        LOG_DEBUG("r: %s", r.to_string().c_str());
        Real rr_new = r.dot(r);
//        LOG_DEBUG("rr_new: %f", rr_new);
        beta = rr_new / rr;
//        LOG_DEBUG("beta: %f", beta);
//        LOG_DEBUG("p * beta: %s", (p * beta).to_string().c_str());
        p = r + p * beta;
        rr = rr_new;
        ++iter;
    }
    LOG_INFO("Conjugate gradient finished in %d iterations in %.4f seconds", iter, Timer::Toc());
    LOG_INFO("x = %s", x.to_string().c_str());
    LOG_INFO("x_star = %s", x_star.to_string().c_str());

    return 0;
}