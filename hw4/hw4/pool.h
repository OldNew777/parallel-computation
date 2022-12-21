//
// Created by ChenXin on 2022/12/20.
//

#pragma once

#include <vector>
#include <list>

using namespace std;

template<typename T>
class Pool {
private:
    size_t default_size = 32768;

    vector<T *> data;
    vector<size_t> occupied;
    vector<size_t> capacity;
    list<size_t> q;

    [[nodiscard]] explicit Pool() = default;

public:
    Pool(Pool const &) = delete;
    Pool &operator=(Pool const &) = delete;
    ~Pool() {
        for (auto &p: data)
            delete[] p;
    }

    [[nodiscard]] static Pool &Global() {
        static Pool instance;
        return instance;
    }

    void deallocate(T *ptr, size_t n) {
        // TODO
    }

    [[nodiscard]] T *allocate(size_t n) {
        size_t new_size = max(n, default_size);

        static auto new_block = [this](size_t needed, size_t new_capacity) {
            data.emplace_back(new T[new_capacity]);
            occupied.emplace_back(needed);
            capacity.emplace_back(new_capacity);
            q.emplace_back(data.size() - 1);
            return data.back();
        };

        // no available memory
        if (q.empty()) {
            return new_block(n, new_size);
        }

        // no enough memory
        auto iter = q.begin();
        while (iter != q.end() and n + occupied[*iter] > capacity[*iter])
            ++iter;
        if (iter == q.end()) {
            return new_block(n, new_size);
        }

        // enough memory
        auto index = *iter;
        occupied[index] += n;
        if (occupied[index] == capacity[index])
            q.remove(index);
        return data[index] + occupied[index] - n;
    }
};