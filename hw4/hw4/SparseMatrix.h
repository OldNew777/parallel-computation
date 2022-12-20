//
// Created by ChenXin on 2022/12/20.
//

#pragma once

#include <random>
#include <unordered_map>
#include <map>
#include <set>
#include <string>
#include <omp.h>
#include <format>

#include "rand_sampler.h"
#include "pool.h"
#include "type.h"

using namespace std;

struct OLNode {
public:
    int row = 0, col = 0;
    Real value = 0.;
    OLNode *right = nullptr, *down = nullptr;

public:
    [[nodiscard]] OLNode() = default;

    [[nodiscard]] string to_string() const {
        return format("[{}, {}, {:.4f}]", row, col, value);
    }
};

class SparseMatrix;

struct SparseVector {
public:
    int length = 0;
    vector<Real> val;

    [[nodiscard]] explicit SparseVector(int length) : length(length) {
        val.resize(length, 0.);
    }
    [[nodiscard]] SparseVector(const SparseVector &other) = default;
    [[nodiscard]] SparseVector& operator=(const SparseVector &other) = default;

    [[nodiscard]] SparseVector operator*(const SparseMatrix& A) const;

    void random() {
        for (int i = 0; i < length; ++i) {
            val[i] = RandSampler::Global().rand_real(-1., 1.);
        }
    }

#define _SPARSE_VECTOR_OPERATOR(op)                                                         \
    SparseVector operator op(const SparseVector &other) const {                             \
        if (length != other.length) {                                                       \
            throw runtime_error("length != other.length in SparseVector + SparseVector");   \
        }                                                                                   \
        SparseVector res(length);                                                           \
        for (int i = 0; i < length; ++i) {                                                  \
            res.val[i] = val[i] op other.val[i];                                            \
        }                                                                                   \
        return res;                                                                         \
    }                                                                                       \
    SparseVector& operator op##=(const SparseVector &other) {                               \
        if (length != other.length) {                                                       \
            throw runtime_error("length != other.length in SparseVector + SparseVector");   \
        }                                                                                   \
        for (int i = 0; i < length; ++i) {                                                  \
            val[i] op##= other.val[i];                                                      \
        }                                                                                   \
        return *this;                                                                       \
    }
    _SPARSE_VECTOR_OPERATOR(+)
    _SPARSE_VECTOR_OPERATOR(-)
    _SPARSE_VECTOR_OPERATOR(*)
    _SPARSE_VECTOR_OPERATOR(/)
#undef _SPARSE_VECTOR_OPERATOR

    [[nodiscard]] string to_string() const {
        string res = "[";
        for (int i = 0; i < length - 1; ++i) {
            res += format("{:.4f}, ", val[i]);
        }
        if (length > 0) {
            res += format("{:.4f}", val[length - 1]);
        }
        res += "]";
        return res;
    }
};

struct SparseMatrix {
public:
    unordered_map<int, OLNode *> row_head, col_head, row_tail, col_tail;
    int n, m, n_non_zero;

public:
    SparseMatrix(const SparseMatrix &other) = delete;
    SparseMatrix &operator=(const SparseMatrix &other) = delete;

    [[nodiscard]] SparseMatrix(int n, int m) : n(n), m(m) {}

    [[nodiscard]] SparseVector operator*(const SparseVector& x) const;

    void random(int n_non_zero) {
        this->n_non_zero = n_non_zero;

        auto &sampler = RandSampler::Global();
        auto &pool = Pool<OLNode>::Global();

        map<int, set<int>> data;
        for (auto i = 0; i < n_non_zero; ++i) {
            int row, col;
            do {
                row = sampler.rand_int(0, n - 1);
                col = sampler.rand_int(0, m - 1);
            } while (data[row].count(col));
            data[row].insert(col);
        }

        for (auto i = 0; i < n; ++i) {
            // save non-zero elements of each row
            set<int> &j_allocated = data[i];

            // allocate memory for each row
            OLNode *node_last = nullptr;
            for (const auto j: j_allocated) {
                auto *node = pool.allocate(1);
                node->row = i;
                node->col = j;
                node->value = sampler.rand_real(-1., 1.);
                // insert into row
                if (node_last == nullptr) {
                    row_head[i] = node;
                } else {
                    node_last->right = node;
                }
                // insert into col
                if (col_tail.find(j) == col_tail.end()) {
                    col_head[j] = node_last;
                } else {
                    col_tail[j]->down = node_last;
                }
                col_tail[j] = node;
                node_last = node;
            }
            if (node_last != nullptr) {
                row_tail[i] = node_last;
            }
        }
    }

    [[nodiscard]] string to_string() const {
        string s;
        for (const auto &iter_x: row_head) {
            string line = "Row " + std::to_string(iter_x.first) + ": ";
            for (auto iter_y = iter_x.second; iter_y != nullptr; iter_y = iter_y->right) {
                line += iter_y->to_string() + " ";
            }
            line += "\n";
            s += line;
        }
        return s;
    }
};
