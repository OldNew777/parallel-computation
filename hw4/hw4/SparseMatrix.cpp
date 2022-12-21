//
// Created by ChenXin on 2022/12/20.
//

#include "SparseMatrix.h"

SparseVector SparseMatrix::dot(const SparseVector &x) const {
    if (this->m != x.length) {
        throw std::runtime_error("A.m != x.length in A * x");
    }

    SparseVector y(this->n);
#pragma omp parallel shared(y)
    {
#pragma omp single
        if (n_thread != omp_get_num_threads()) {
            auto allocate_threads = [this]() {
                int num_threads = omp_get_num_threads();
                LOG_INFO("%d rows allocated for %d threads", n, num_threads);
                vector<unordered_set<int>> row_set(num_threads);
                Real n_row_avg = (Real) rindex.size() / num_threads;
                int thread_id_now = 0;
                int n_row_now = 0.;
                int row_id_allocated = -1;
                for (auto node : rdata) {
                    if (node.row == row_id_allocated)
                        continue;
                    row_id_allocated = node.row;
                    row_set[thread_id_now].insert(row_id_allocated);
                    n_row_now += 1;
                    if (n_row_now > n_row_avg and thread_id_now < num_threads - 1) {
                        thread_id_now++;
                        n_row_now = 0;
                    }
                }
                return row_set;
            };
            rindex_threads = allocate_threads();
            n_thread = omp_get_num_threads();
        }
#pragma omp barrier
        SparseVector y_private(this->n);
        for (const auto& [row, row_range] : this->rindex) {
            if (rindex_threads.at(omp_get_thread_num()).count(row) == 0) {
                continue;
            }
            for (auto j = row_range.start; j < row_range.end; ++j) {
                y_private.val[row] += rdata[j].value * x.val[rdata[j].col];
            }
        }
#pragma omp critical
        y += y_private;
    }
    return y;
}

Real Config::epsilon = 1e-3;
