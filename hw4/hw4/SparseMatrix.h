//
// Created by ChenXin on 2022/12/20.
//

#pragma once

#include <random>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <omp.h>
#include <format>

#include "rand_sampler.h"
#include "type.h"
#include "log.h"

//#define DEBUG_FULL_PRINT

using namespace std;


struct Range {
    int start = 0, end = 0;
};

class Config {
public:
    static Real epsilon;
    static int n_threads;
    static vector<Range> for_parallel;

    static void init_for_parallel(int n) {
        Real n_avg = (Real) n / n_threads;
        for_parallel.resize(n_threads);
        for_parallel[0].start = 0;
        for_parallel[0].end = 0;
        int thread_id_now = 0;
        int n_now = 0;
        for (int i = 0; i < n; ++i) {
            if (n_now > n_avg and thread_id_now < n_threads - 1) {
                thread_id_now++;
                n_now = 0;
                for_parallel[thread_id_now].start = i;
                for_parallel[thread_id_now].end = i;
            }
            for_parallel[thread_id_now].end++;
            n_now++;
        }
    }
};

class SparseMatrix;

class Vector {
public:
    int length = 0;
    vector<Real> val;

public:
    [[nodiscard]] Vector(int length, Real value = 0.) : length(length) {
        val.resize(length, value);
    }
    [[nodiscard]] Vector(const Vector &other) = default;
    [[nodiscard]] Vector& operator=(const Vector &other) = default;

    void random() {
        for (int i = 0; i < length; ++i) {
            val[i] = RandSampler::Global().rand_real(-1., 1.);
        }
    }

    [[nodiscard]] Vector operator +(const Vector &other) const {
        Vector res(length);
#pragma omp parallel shared(res) num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                res.val[i] = val[i] + other.val[i];
            }
        }
        return res;
    }
    [[nodiscard]] Vector operator -(const Vector &other) const {
        Vector res(length);
#pragma omp parallel shared(res) num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                res.val[i] = val[i] - other.val[i];
            }
        }
        return res;
    }
    [[nodiscard]] Vector operator *(const Vector &other) const {
        Vector res(length);
#pragma omp parallel shared(res) num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                res.val[i] = val[i] * other.val[i];
            }
        }
        return res;
    }
    [[nodiscard]] Vector operator /(const Vector &other) const {
        Vector res(length);
#pragma omp parallel shared(res) num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                res.val[i] = val[i] / other.val[i];
            }
        }
        return res;
    }
    Vector& operator +=(const Vector &other) {
#pragma omp parallel num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                val[i] += other.val[i];
            }
        }
        return *this;
    }
    Vector& operator -=(const Vector &other) {
#pragma omp parallel num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                val[i] -= other.val[i];
            }
        }
        return *this;
    }
    Vector& operator *=(const Vector &other) {
#pragma omp parallel num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                val[i] *= other.val[i];
            }
        }
        return *this;
    }
    Vector& operator /=(const Vector &other) {
#pragma omp parallel num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                val[i] /= other.val[i];
            }
        }
        return *this;
    }
    [[nodiscard]] Vector operator +(Real a) const {
        Vector res(length);
#pragma omp parallel shared(res) num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                res.val[i] = val[i] + a;
            }
        }
        return res;
    }
    [[nodiscard]] Vector operator -(Real a) const {
        Vector res(length);
#pragma omp parallel shared(res) num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                res.val[i] = val[i] - a;
            }
        }
        return res;
    }
    [[nodiscard]] Vector operator *(Real a) const {
        Vector res(length);
#pragma omp parallel shared(res) num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                res.val[i] = val[i] * a;
            }
        }
        return res;
    }
    [[nodiscard]] Vector operator /(Real a) const {
        Vector res(length);
#pragma omp parallel shared(res) num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                res.val[i] = val[i] / a;
            }
        }
        return res;
    }
    Vector& operator +=(Real a) {
#pragma omp parallel num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                val[i] += a;
            }
        }
        return *this;
    }
    Vector& operator -=(Real a) {
#pragma omp parallel num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                val[i] -= a;
            }
        }
        return *this;
    }
    Vector& operator *=(Real a) {
#pragma omp parallel num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                val[i] *= a;
            }
        }
        return *this;
    }
    Vector& operator /=(Real a) {
#pragma omp parallel num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                val[i] /= a;
            }
        }
        return *this;
    }

    [[nodiscard]] Real dot(const Vector &other) const {
        Real res = 0.;
#pragma omp parallel num_threads(Config::n_threads) reduction(+:res)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                res += val[i] * other.val[i];
            }
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

struct OLNode {
public:
    int row = 0, col = 0;
    Real value = 0.;

public:
    [[nodiscard]] OLNode() = default;
    [[nodiscard]] OLNode(const OLNode &other) = default;
    [[nodiscard]] OLNode(OLNode &&other) noexcept = default;
    [[nodiscard]] OLNode(int row, int col, Real value) : row(row), col(col), value(value) {}
    OLNode& operator=(const OLNode &other) = default;

    [[nodiscard]] string to_string() const {
        return std::format("[{}, {}, {:.4f}]", row, col, value);
    }
};

struct SparseMatrix {
public:
    vector<OLNode> rdata, cdata;
    unordered_map<int, Range> rindex, cindex;
    int n, m;
    vector<Real> col_abs_sum;

    mutable vector<unordered_set<int>> rindex_threads;
    mutable int n_thread = -1;

public:
    SparseMatrix(const SparseMatrix &other) = delete;
    SparseMatrix &operator=(const SparseMatrix &other) = delete;

    [[nodiscard]] SparseMatrix(int n, int m, int n_non_zero) : n(n), m(m) {
        if (n_non_zero > int64_t(n) * int64_t(m)) {
            throw runtime_error("n_non_zero > n * m in SparseMatrix::random");
        }

        col_abs_sum.resize(m, 0.);
        rdata.reserve(n_non_zero);

        auto &sampler = RandSampler::Global();

        unordered_map<int, unordered_set<int>> data;
        int n_diag = min(min(n, m), n_non_zero);
        for (int i = 0; i < n_diag; ++i) {
            data[i].insert(i);
            rdata.emplace_back(OLNode{i, i, sampler.rand_real(-1., 1.)});
            col_abs_sum[i] += fabs(rdata.back().value);
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
            rdata.emplace_back(OLNode{row, col, sampler.rand_real(-1., 1.)});
            col_abs_sum[col] += fabs(rdata.back().value);
        }
        unordered_map<int, unordered_set<int>>().swap(data);
        cdata = rdata;

        // sort by row
        sort(rdata.begin(), rdata.end(), [](const OLNode &a, const OLNode &b) {
            if (a.row != b.row) {
                return a.row < b.row;
            }
            return a.col < b.col;
        });
        // sort by col
        sort(cdata.begin(), cdata.end(), [](const OLNode &a, const OLNode &b) {
            if (a.col != b.col) {
                return a.col < b.col;
            }
            return a.row < b.row;
        });

        // build index and refresh diagonal to make sure it is diagonal dominant
        vector<Real> diag(min(n, m), 0.);
        for (int i = 0; i < min(n, m); ++i) {
            diag[i] = sampler.rand_real(2.0, 5.0) * col_abs_sum[i];
        }
        for (int i = 0; i < n_non_zero; ++i) {
            if (rindex.count(rdata[i].row) == 0) {
                rindex[rdata[i].row] = Range{i, i + 1};
            } else {
                rindex[rdata[i].row].end = i + 1;
            }
            if (cindex.count(cdata[i].col) == 0) {
                cindex[cdata[i].col] = Range{i, i + 1};
            } else {
                cindex[cdata[i].col].end = i + 1;
            }

            if (rdata[i].row == rdata[i].col) {
                rdata[i].value = diag[rdata[i].col];
            }
            if (cdata[i].row == cdata[i].col) {
                cdata[i].value = diag[cdata[i].col];
            }
        }
        vector<Real>().swap(diag);
    }

    [[nodiscard]] Vector dot(const Vector& x) const;

    [[nodiscard]] string to_string() const {
        static auto print_line = [this](int row) {
            string line = "\nRow " + std::to_string(row) + ": ";
#ifndef DEBUG_FULL_PRINT
            int n_print_col = 0;
#endif
            Range row_range = rindex.at(row);
            for (int i = row_range.start; i < row_range.end; ++i) {
                line += rdata.at(i).to_string() + " ";
#ifndef DEBUG_FULL_PRINT
                if (++n_print_col >= 3)
                    break;
#endif
            }
#ifndef DEBUG_FULL_PRINT
            if (n_print_col < row_range.end - row_range.start) {
                line += "... " + rdata.at(row_range.end - 1).to_string();
            }
#endif
            return line;
        };

        string s;
#ifndef DEBUG_FULL_PRINT
        int n_print_row = 0;
#endif
        for (const auto & [row, row_range] : rindex) {
            s += print_line(row);
#ifndef DEBUG_FULL_PRINT
            if (++n_print_row >= 3)
                break;
#endif
        }
#ifndef DEBUG_FULL_PRINT
        if (n_print_row < rindex.size()) {
            s += "\n...";
        }
#endif

        return s;
    }
};
