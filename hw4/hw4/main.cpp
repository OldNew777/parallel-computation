//
// Created by ChenXin on 2022/12/20.
//

#include <omp.h>

#include "log.h"
#include "timer.h"
#include "SparseMatrix.h"
#include "cxxopts.hpp"

int n_thread;
int n, m, n_non_zero;

int parse_args(int argc, char *argv[]) {
    cxxopts::Options options(
            "Sparse Matrix Conjugate Gradient",
            "Calculate the solution of Ax = b by conjugate gradient method, with A being a sparse matrix.");
    options.add_options()
            ("n", "Dimension n of matrix A", cxxopts::value<int>(n)->default_value("1000"))
            ("m", "Dimension m of matrix A", cxxopts::value<int>(m)->default_value("1000"))
            ("n_non_zero", "Number of non-zero values in A", cxxopts::value<int>(n_non_zero)->default_value("1000"))
            ("n_thread", "Number of threads", cxxopts::value<int>(n_thread)->default_value("1"))
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
    LOG_INFO("Init matrix A in %.4f seconds", Timer::Tok());

    SparseVector x(n);
    x.random();
    LOG_INFO("Init vector x in %.4f seconds", Timer::Tok());

    auto b = A * x;
    LOG_INFO("Calculate b=Ax in %.4f seconds", Timer::Tok());

//    LOG_INFO("A: %s", A.to_string().c_str());
//    LOG_INFO("x: %s", x.to_string().c_str());
//    LOG_INFO("b: %s", b.to_string().c_str());

    return 0;
}