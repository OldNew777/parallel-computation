//
// Created by ChenXin on 2022/12/20.
//

#pragma once

#include <random>
#include <unordered_map>
#include <set>
#include <string>
#include <omp.h>
#include <format>

#include "rand_sampler.h"
#include "pool.h"
#include "type.h"
#include "log.h"

//#define DEBUG_FULL_PRINT

using namespace std;

struct OLNode {
public:
    int row = 0, col = 0;
    Real value = 0.;
    OLNode *right = nullptr, *down = nullptr;

public:
    [[nodiscard]] OLNode() = default;

    [[nodiscard]] string to_string() const {
        return std::format("[{}, {}, {:.4f}]", row, col, value);
    }
};

class SparseMatrix;

struct SparseVector {
public:
    int length = 0;
    vector<Real> val;

    [[nodiscard]] explicit SparseVector(int length, Real value = 0.) : length(length) {
        val.resize(length, value);
    }
    [[nodiscard]] SparseVector(const SparseVector &other) = default;
    [[nodiscard]] SparseVector& operator=(const SparseVector &other) = default;

    [[nodiscard]] SparseVector operator*(const SparseMatrix& A) const;

    void random() {
        for (int i = 0; i < length; ++i) {
            val[i] = RandSampler::Global().rand_real(-1., 1.);
        }
    }

#define _SPARSE_VECTOR_OPERATOR(op)                                                                 \
    [[nodiscard]] SparseVector operator op(const SparseVector &other) const {                       \
        if (length != other.length) {                                                               \
            throw runtime_error("length != other.length in SparseVector " #op " SparseVector");     \
        }                                                                                           \
        SparseVector res(length);                                                                   \
        for (int i = 0; i < length; ++i) {                                                          \
            res.val[i] = val[i] op other.val[i];                                                    \
        }                                                                                           \
        return res;                                                                                 \
    }                                                                                               \
    SparseVector& operator op##=(const SparseVector &other) {                                       \
        if (length != other.length) {                                                               \
            throw runtime_error("length != other.length in SparseVector " #op " SparseVector");     \
        }                                                                                           \
        for (int i = 0; i < length; ++i) {                                                          \
            val[i] op##= other.val[i];                                                              \
        }                                                                                           \
        return *this;                                                                               \
    }                                                                                               \
    [[nodiscard]] SparseVector operator op(Real a) const {                                          \
        SparseVector res(length);                                                                   \
        for (int i = 0; i < length; ++i) {                                                          \
            res.val[i] = val[i] op a;                                                               \
        }                                                                                           \
        return res;                                                                                 \
    }                                                                                               \
    SparseVector& operator op##=(Real a) {                                                          \
        for (int i = 0; i < length; ++i) {                                                          \
            val[i] op##= a;                                                                         \
        }                                                                                           \
        return *this;                                                                               \
    }
    _SPARSE_VECTOR_OPERATOR(+)
    _SPARSE_VECTOR_OPERATOR(-)
    _SPARSE_VECTOR_OPERATOR(*)
    _SPARSE_VECTOR_OPERATOR(/)
#undef _SPARSE_VECTOR_OPERATOR

    [[nodiscard]] Real dot(const SparseVector &other) const {
        if (length != other.length) {
            throw runtime_error("length != other.length in SparseVector.dot(SparseVector)");
        }
        Real res = 0.;
        for (int i = 0; i < length; ++i) {
            res += val[i] * other.val[i];
        }
        return res;
    }

    [[nodiscard]] string to_string() const {
        string res = "[";
        for (int i = 0; i < length - 1; ++i) {
            res += std::format("{:.4f}, ", val[i]);
#ifndef DEBUG_FULL_PRINT
            if (i + 1 == 3) {
                res += "... ";
                break;
            }
#endif
        }
        if (length > 0) {
            res += std::format("{:.4f}", val[length - 1]);
        }
        res += "]";
        return res;
    }
};

class Config {
public:
    static Real epsilon;
    static Real lr;
};

struct SparseMatrix {
public:
    unordered_map<int, OLNode *> row_head, col_head, row_tail, col_tail;
    int n, m;

public:
    SparseMatrix(const SparseMatrix &other) = delete;
    SparseMatrix &operator=(const SparseMatrix &other) = delete;

    [[nodiscard]] SparseMatrix(int n, int m) : n(n), m(m) {}

    [[nodiscard]] SparseVector operator*(const SparseVector& x) const;

    void random(int n_non_zero) {
        if (n_non_zero > int64_t(n) * int64_t(m)) {
            throw runtime_error("n_non_zero > n * m in SparseMatrix::random");
        }

        auto &sampler = RandSampler::Global();
        auto &pool = Pool<OLNode>::Global();

        unordered_map<int, set<int>> data;
        for (auto i = 0; i < n_non_zero; ++i) {
            int row, col;

            // Method 1: add 1 if full
            row = sampler.rand_int(0, n - 1);
            while (data.count(row) && data[row].size() == m)
                row = (row + 1) % n;
            col = sampler.rand_int(0, m - 1);
            while (data.count(row) && data[row].count(col))
                col = (col + 1) % m;
//            // Method 2: reject sampling
//            do {
//                row = sampler.rand_int(0, n - 1);
//            } while (data.count(row) && data[row].size() == m);
//            do {
//                col = sampler.rand_int(0, m - 1);
//            } while (data.count(row) && data[row].count(col));

            data[row].insert(col);
        }

        for (auto i = 0; i < n; ++i) {
            if (!data.count(i)) {
                continue;
            }

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
                if (col_tail.count(j) == 0) {
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
        static auto print_line = [this](const unordered_map<int, OLNode *>::const_iterator &iter_x) {
            string line = "\nRow " + std::to_string(iter_x->first) + ": ";
#ifndef DEBUG_FULL_PRINT
            int n_print_col = 0;
            bool break_col = false;
#endif
            for (auto iter_y = iter_x->second; iter_y != nullptr; iter_y = iter_y->right) {
                line += iter_y->to_string() + " ";
#ifndef DEBUG_FULL_PRINT
                if (++n_print_col >= 3) {
                    break_col = true;
                    break;
                }
#endif
            }
#ifndef DEBUG_FULL_PRINT
            // FIXME: maybe we have printed all cols before
            if (break_col && n_print_col < m) {
                auto iter_y = row_tail.at(iter_x->first);
                line += "... " + iter_y->to_string();
            }
#endif
            return line;
        };

        string s;
#ifndef DEBUG_FULL_PRINT
        int n_print_row = 0;
        bool break_row = false;
#endif
        for (auto iter_x = row_head.begin(); iter_x != row_head.end(); ++iter_x) {
            s += print_line(iter_x);
#ifndef DEBUG_FULL_PRINT
            if (++n_print_row >= 3) {
                break_row = true;
                break;
            }
#endif
        }
#ifndef DEBUG_FULL_PRINT
        // FIXME: maybe we have printed all rows before
        if (break_row && n_print_row < n) {
            auto iter_x = row_head.begin();
            std::advance(iter_x, n - 1);
            s += "\n..." + print_line(iter_x);
        }
#endif
        return s;
    }
};
