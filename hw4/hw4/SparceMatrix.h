//
// Created by ChenXin on 2022/12/20.
//

#pragma once

#include <random>
#include <unordered_map>
#include <map>
#include <set>
#include <string>

#include "rand_sampler.h"
#include "pool.h"

using namespace std;

struct OLNode {
public:
    int row = 0, col = 0;
    double value = 0.L;
    OLNode *right = nullptr, *down = nullptr;

public:
    [[nodiscard]] OLNode() = default;

    [[nodiscard]] string to_string() const {
        return format("({}, {}, {:.4f})", row, col, value);
    }
};

class CrossList {
private:
    unordered_map<int, OLNode *> row_head, col_head, row_tail, col_tail;
    int n, m, n_non_zero;

public:
    [[nodiscard]] CrossList(int n, int m, int n_non_zero) : n(n), m(m), n_non_zero(n_non_zero) {
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
            for (auto j: j_allocated) {
                auto *node = pool.allocate(1);
                node->row = i;
                node->col = j;
                node->value = sampler.rand_double(-1., 1.);
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

    [[nodiscard]] string to_string() {
        string s;
        for (auto iter_i : row_head) {
            string line = "Row " + std::to_string(iter_i.first) + ": ";
            for (auto iter_j = iter_i.second; iter_j != nullptr; iter_j = iter_j->right) {
                line += iter_j->to_string() + " ";
            }
            line += "\n";
            s += line;
        }
        return s;
    }
};