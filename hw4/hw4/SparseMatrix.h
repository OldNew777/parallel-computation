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
    vector<Real> col_abs_sum;

public:
    SparseMatrix(const SparseMatrix &other) = delete;
    SparseMatrix &operator=(const SparseMatrix &other) = delete;

    [[nodiscard]] SparseMatrix(int n, int m) : n(n), m(m) {
        col_abs_sum.resize(m, 0.);
    }

    [[nodiscard]] SparseVector operator*(const SparseVector& x) const;

    bool insert(int i, int j, Real value) {
        if (i < 0 || i >= n || j < 0 || j >= m) {
            throw runtime_error("index out of range in SparseMatrix.insert");
        }
        auto &pool = Pool<OLNode>::Global();

        auto create_node = [&]() {
            auto *node = pool.allocate(1);
            node->row = i;
            node->col = j;
            node->value = value;
            return node;
        };
        auto *node = create_node();
        col_abs_sum[j] += fabs(value);

        // insert into row
        if (row_head.count(i) == 0) {
            row_head[i] = node;
            row_tail[i] = node;
        } else if (OLNode *head = row_head[i]; head->col > j) {
            node->right = head;
            row_head[i] = node;
        } else {
            OLNode *ptr = row_head[i];
            while (ptr->right != nullptr && ptr->right->col <= j) {
                ptr = ptr->right;
            }
            // ptr->col <= j
            if (ptr->col == j) {
                ptr->value = value;
                pool.deallocate(node, 1);
                return false;
            } else {
                node->right = ptr->right;
                ptr->right = node;
                if (row_tail[i]->col < j) {
                    row_tail[i] = node;
                }
            }
        }

        // insert into col
        if (col_head.count(j) == 0) {
            col_head[j] = node;
            col_tail[j] = node;
        } else if (OLNode *head = col_head[j]; head->row > i) {
            node->down = head;
            col_head[j] = node;
        } else {
            OLNode *ptr = col_head[j];
            while (ptr->down != nullptr && ptr->down->row <= i) {
                ptr = ptr->down;
            }
            // ptr->row <= i
            if (ptr->row == i) {
                ptr->value = value;
                pool.deallocate(node, 1);
                return false;
            } else {
                node->down = ptr->down;
                ptr->down = node;
                if (col_tail[j]->row < i) {
                    col_tail[j] = node;
                }
            }
        }

        return true;
    }

    void random(int n_non_zero) {
        if (n_non_zero > int64_t(n) * int64_t(m)) {
            throw runtime_error("n_non_zero > n * m in SparseMatrix::random");
        }

        auto &sampler = RandSampler::Global();
        auto &pool = Pool<OLNode>::Global();

        unordered_map<int, set<int>> data;
        int n_diag = min(min(n, m), n_non_zero);
        for (int i = 0; i < n_diag; ++i) {
            data[i].insert(i);
        }
        for (auto i = 0; i < n_non_zero - n_diag; ++i) {
            int row, col;

            // Method 1: add 1 if full
            row = sampler.rand_int(0, n - 1);
            while (data.count(row) && data[row].size() == m)
                row = (row + 1) % n;
            col = sampler.rand_int(0, m - 1);
            while (data[row].count(col))
                col = (col + 1) % m;

            data[row].insert(col);
        }

        // allocate memory
        for (auto i = 0; i < n; ++i) {
            if (!data.count(i)) {
                continue;
            }
            for (const auto j: data[i]) {
                insert(i, j, sampler.rand_real(-1., 1.));
            }
        }

        // refresh diagonal to make sure it is diagonal dominant
        Real col_abs_sum_max = *std::max_element(col_abs_sum.begin(), col_abs_sum.end());
        for (auto &iter_y: col_head) {
            OLNode *node = iter_y.second;
            while (node != nullptr) {
                if (node->col == node->row) {
                    node->value = sampler.rand_real(2.0, 5.0) * col_abs_sum[node->col];
                    node->value *= sampler.rand_int(0, 1) ? 1 : -1;
                    break;
                } else if (node->row > node->col) {
                    break;
                }
                node = node->down;
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
