//
// Created by ChenXin on 2022/12/20.
//

#pragma once

#include <random>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <format>
#include <fstream>

#ifdef ENABLE_PARALLEL
#include <omp.h>
#endif

#include "rand_sampler.h"
#include "type.h"
#include "log.h"
#include "timer.h"

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

    static void init_for_parallel(size_t n, vector<Range> &rindex) {
        Real n_avg = (Real) n / n_threads;
        rindex.resize(n_threads);
        rindex[0].start = 0;
        rindex[0].end = 0;
        int thread_id_now = 0;
        int n_now = 0;
        for (auto i = 0; i < n; ++i) {
            if (n_now >= n_avg and thread_id_now < n_threads - 1) {
                thread_id_now++;
                n_now = 0;
                rindex[thread_id_now].start = i;
                rindex[thread_id_now].end = i;
            }
            rindex[thread_id_now].end++;
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
    [[nodiscard]] explicit Vector(int length, Real value = 0.) : length(length) {
        val.resize(length, value);
    }
    [[nodiscard]] Vector(const Vector &other) {
        *this = other;
    }
#ifdef ENABLE_PARALLEL
    [[nodiscard]] Vector& operator=(const Vector &other) {
        length = other.length;
        val.resize(length);
#pragma omp parallel num_threads(Config::n_threads)
        {
            int id = omp_get_thread_num();
            const auto &range = Config::for_parallel[id];
            for (int i = range.start; i < range.end; ++i) {
                val[i] = other.val[i];
            }
        }
        return *this;
    }
#else
    [[nodiscard]] Vector& operator=(const Vector &other) = default;
#endif
    [[nodiscard]] Vector(Vector &&other) noexcept = default;
    [[nodiscard]] Vector& operator=(Vector &&other) noexcept = default;

    void random() {
        for (int i = 0; i < length; ++i) {
            val[i] = RandSampler::Global().rand_real(-1., 1.);
        }
    }

#ifdef ENABLE_PARALLEL
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
#else
#define _VECTOR_OPERATOR(op)                                                                 \
    [[nodiscard]] Vector operator op(const Vector &other) const {                       \
        if (length != other.length) {                                                               \
            throw runtime_error("length != other.length in Vector " #op " Vector");     \
        }                                                                                           \
        Vector res(length);                                                                   \
        for (int i = 0; i < length; ++i) {                                                          \
            res.val[i] = val[i] op other.val[i];                                                    \
        }                                                                                           \
        return res;                                                                                 \
    }                                                                                               \
    Vector& operator op##=(const Vector &other) {                                       \
        if (length != other.length) {                                                               \
            throw runtime_error("length != other.length in Vector " #op " Vector");     \
        }                                                                                           \
        for (int i = 0; i < length; ++i) {                                                          \
            val[i] op##= other.val[i];                                                              \
        }                                                                                           \
        return *this;                                                                               \
    }                                                                                               \
    [[nodiscard]] Vector operator op(Real a) const {                                          \
        Vector res(length);                                                                   \
        for (int i = 0; i < length; ++i) {                                                          \
            res.val[i] = val[i] op a;                                                               \
        }                                                                                           \
        return res;                                                                                 \
    }                                                                                               \
    Vector& operator op##=(Real a) {                                                          \
        for (int i = 0; i < length; ++i) {                                                          \
            val[i] op##= a;                                                                         \
        }                                                                                           \
        return *this;                                                                               \
    }
    _VECTOR_OPERATOR(+)
    _VECTOR_OPERATOR(-)
    _VECTOR_OPERATOR(*)
    _VECTOR_OPERATOR(/)
#undef _VECTOR_OPERATOR

    [[nodiscard]] Real dot(const Vector &other) const {
        Real res = 0.;
        for (int i = 0; i < length; ++i) {
            res += val[i] * other.val[i];
        }
        return res;
    }
#endif

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

class SparseMatrix {
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

    [[nodiscard]] SparseMatrix() = default;
    [[nodiscard]] SparseMatrix(const string &filename) {
        load_from_file(filename);
    }

    [[nodiscard]] SparseMatrix(int n, int m, size_t n_non_zero) : n(n), m(m) {
        if (n_non_zero > int64_t(n) * int64_t(m)) {
            throw runtime_error("n_non_zero > n * m in SparseMatrix::random");
        }

        Timer::Tik();
        LOG_INFO("Generating random sparse matrix (%d*%d) with %d non-zero elements...", n, m, n_non_zero);
        col_abs_sum.resize(m, 0.);
        rdata.resize(n_non_zero);

        auto &sampler = RandSampler::Global();

        unordered_map<int, unordered_set<int>> data;
        int n_diag = min(size_t(min(n, m)), n_non_zero);
        for (int i = 0; i < n_diag; ++i) {
            data[i].insert(i);
            auto &&node = OLNode{i, i, sampler.rand_real(-1., 1.)};
            col_abs_sum[i] += fabs(node.value);
            rdata[i] = node;
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

//            // Method 2: reject sampling
//            do {
//                row = sampler.rand_int(0, n - 1);
//                col = sampler.rand_int(0, m - 1);
//            } while (data[row].count(col));

            data[row].insert(col);
            auto &&node = OLNode{row, col, sampler.rand_real(-1., 1.)};
            col_abs_sum[col] += fabs(node.value);
            rdata[i + n_diag] = node;
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
        LOG_INFO("SparseMatrix::random: %.4fs", Timer::Toc());
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

    void save_to_file(const string &filename) const {
        Timer::Tik();
        LOG_INFO("Start to save matrix A(%d*%d) with %d non-zero elements", n, m, rdata.size());
        size_t n_non_zero = rdata.size();
        ofstream fout(filename, ios::out | ios::binary);
        fout.write((const char *)&n, sizeof(n));
        fout.write((const char *)&m, sizeof(m));
        fout.write((const char *)&n_non_zero, sizeof(n_non_zero));
        fout.write((const char *)rdata.data(), sizeof(OLNode) * n_non_zero);
        fout.write((const char *)cdata.data(), sizeof(OLNode) * n_non_zero);
        size_t rindex_size = rindex.size(), cindex_size = cindex.size();
        fout.write((const char *)&rindex_size, sizeof(rindex_size));
        for (const auto & [row, range] : rindex) {
            fout.write((const char *)&row, sizeof(row));
            fout.write((const char *)&range, sizeof(range));
        }
        fout.write((const char *)&cindex_size, sizeof(cindex_size));
        for (const auto & [col, range] : cindex) {
            fout.write((const char *)&col, sizeof(col));
            fout.write((const char *)&range, sizeof(range));
        }
        fout.close();
        LOG_INFO("Save matrix (%d*%d) in %.4fs", n, m, Timer::Toc());
    }

    void load_from_file(const string &filename) {
        Timer::Tik();
        ifstream fin(filename, ios::in | ios::binary);
        if (!fin.is_open()) {
            LOG_ERROR("Cannot open file %s", filename.c_str());
            exit(1);
        }
        size_t n_non_zero;
        fin.read((char *)&n, sizeof(n));
        fin.read((char *)&m, sizeof(m));
        fin.read((char *)&n_non_zero, sizeof(n_non_zero));
        LOG_INFO("Start to load matrix (%d*%d) with %d non-zero elements", n, m, n_non_zero);
        rdata.resize(n_non_zero);
        cdata.resize(n_non_zero);
        fin.read((char *)rdata.data(), sizeof(OLNode) * n_non_zero);
        fin.read((char *)cdata.data(), sizeof(OLNode) * n_non_zero);
        size_t rindex_size, cindex_size;
        fin.read((char *)&rindex_size, sizeof(rindex_size));
        for (int i = 0; i < rindex_size; ++i) {
            int row;
            Range range;
            fin.read((char *)&row, sizeof(row));
            fin.read((char *)&range, sizeof(range));
            rindex[row] = range;
        }
        fin.read((char *)&cindex_size, sizeof(cindex_size));
        for (int i = 0; i < cindex_size; ++i) {
            int col;
            Range range;
            fin.read((char *)&col, sizeof(col));
            fin.read((char *)&range, sizeof(range));
            cindex[col] = range;
        }
        fin.close();
        LOG_INFO("Load matrix (%d*%d) in %.4fs", n, m, Timer::Toc());
    }
};
