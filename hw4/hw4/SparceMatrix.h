//
// Created by ChenXin on 2022/12/20.
//

#pragma once

#include <random>
#include <unordered_map>
#include <unordered_set>

using namespace std;

struct OLNode {
	int row = 0, col = 0;
	double value = 0.L;
	OLNode* right = nullptr, * down = nullptr;

	~OLNode() {
        delete right;
        delete down;
	}
};

struct CrossList
{
	unordered_map<int, OLNode*> row_head, col_head;
	int n, m, n_nozero;

	CrossList(int n, int m, int n_nozero) : n(n), m(m), n_nozero(n_nozero) {
		auto rand_01 = []() {
			return double(rand()) / RAND_MAX;
		};

		int n_allocated = 0;
		int64_t nm = int64_t(n) * int64_t(m);
		for (int i = 0; i < n; ++i) {
			unordered_set<int> j_allocated;
			while (true) {
				double p = double(m - j_allocated.size()) / (nm - n_allocated);
				if (rand_01() <= p) {
					++n_allocated;
					int index = rand() % m;
					while (j_allocated.find(index) != j_allocated.end())
						index = rand() % m;
					j_allocated.emplace(index);
				}
				else {
					break;
				}
			}
			OLNode* node_last = nullptr;
			for (auto j : j_allocated) {
				OLNode* node = new OLNode();
			}
		}
	}
	~CrossList() {
		for (auto iter : row_head)
			delete iter.second;
		for (auto iter : col_head)
			delete iter.second;
	}
};