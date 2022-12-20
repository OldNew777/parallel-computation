//
// Created by ChenXin on 2022/12/20.
//

#include "SparseMatrix.h"

SparseVector SparseVector::operator*(const SparseMatrix &A) const {
    if (A.n != this->length) {
        throw std::runtime_error("A.n != x.length in x * A");
    }
    SparseVector y(A.m);
#pragma omp parallel
    {
        SparseVector y_private(A.m);
        int index = 0;
        for (auto& iter_vec : A.col_head) {
            if (index++ % omp_get_num_threads() != omp_get_thread_num()) {
                continue;
            }
            for (auto iter_mult = iter_vec.second; iter_mult != nullptr; iter_mult = iter_mult->down) {
                y_private.val[iter_vec.first] += this->val[iter_mult->row] * iter_mult->value;
            }
        }
#pragma omp critical
        y += y_private;
    }
    return y;
}

SparseVector SparseMatrix::operator*(const SparseVector &x) const {
    if (this->m != x.length) {
        throw std::runtime_error("A.m != x.length in A * x");
    }
    SparseVector y(this->n);
#pragma omp parallel
    {
        SparseVector y_private(this->n);
        int index = 0;
        for (auto& iter_vec : this->row_head) {
            if ((index++) % omp_get_num_threads() != omp_get_thread_num()) {
                continue;
            }
            for (auto iter_mult = iter_vec.second; iter_mult != nullptr; iter_mult = iter_mult->right) {
                y_private.val[iter_vec.first] += iter_mult->value * x.val[iter_mult->col];
            }
        }
#pragma omp critical
        y += y_private;
    }
    return y;
}

Real Config::epsilon = 1e-4;
