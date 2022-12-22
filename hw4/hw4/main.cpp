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
    SparseMatrix A(n, m, n_non_zero);
    LOG_INFO("Init matrix A in %.4f seconds", Timer::Toc());

    Timer::Tik();
    SparseVector x(m);
    x.random();
    LOG_INFO("Init vector x in %.4f seconds", Timer::Toc());

    Timer::Tik();
    SparseVector b = A.dot(x);
    LOG_INFO("Calculate vector b in %.4f seconds", Timer::Toc());

    LOG_INFO("A: %s", A.to_string().c_str());
    LOG_INFO("x: %s", x.to_string().c_str());
    LOG_INFO("b: %s", b.to_string().c_str());

    SparseVector x_star(A.m);           // (m * 1)
    x_star.random();
    LOG_INFO("x0: %s", x_star.to_string().c_str());
    SparseVector r = b - A.dot(x_star); // (n * 1)
    SparseVector p = r;                 // (n * 1)
    SparseVector Ap(m);                 // (m * 1)
    Real rr = r.dot(r);
    Real alpha, beta;
    int iter = 0;
    Timer::Tik();
    auto iter_timer = Timer();
    while (sqrt(rr) > Config::epsilon) {
        iter_timer.tik();
//        LOG_DEBUG("Iter %04d: p: %s", iter, p.to_string().c_str());
//        LOG_DEBUG("Iter %04d: rr: %f", iter, rr);
        Ap = A.dot(p);          // (n * m) * (m * 1) = n * 1
//        LOG_DEBUG("Iter %04d: Ap: %s", iter, Ap.to_string().c_str());
        alpha = rr / p.dot(Ap); // (1 * 1) / ((1 * n) * (n * 1)) = 1
        // TODO: barrier after p.dot(Ap)
//        LOG_DEBUG("Iter %04d: alpha: %f", iter, alpha);
        x_star += p * alpha;    // (n * 1) * 1
        LOG_DEBUG("Iter %04d: x_star = %s", iter, x_star.to_string().c_str());
        r -= Ap * alpha;        // (n * 1) * 1
//        LOG_DEBUG("Iter %04d: r: %s", iter, r.to_string().c_str());
        // TODO: barrier
        Real rr_new = r.dot(r); // (1 * n) * (n * 1) = 1
//        LOG_DEBUG("Iter %04d: rr_new: %f", iter, rr_new);
        beta = rr_new / rr;
//        LOG_DEBUG("Iter %04d: beta: %f", iter, beta);
//        LOG_DEBUG("Iter %04d: p * beta: %s", iter, (p * beta).to_string().c_str());
        p = r + p * beta;       // (n * 1) * 1
        rr = rr_new;
        LOG_INFO("Iter %04d: %.4fs", iter, iter_timer.toc());
        ++iter;
    }
    LOG_INFO("Conjugate gradient finished in %d iterations in %.4f seconds", iter, Timer::Toc());
    LOG_INFO("x = %s", x.to_string().c_str());
    LOG_INFO("x_star = %s", x_star.to_string().c_str());

    return 0;
}