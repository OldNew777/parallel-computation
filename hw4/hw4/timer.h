#pragma once

#include <chrono>

class Timer {
public:
    static void Tik() {
        time_point = std::chrono::high_resolution_clock::now();
    }

    static double Toc() {
        auto time_point_new = std::chrono::high_resolution_clock::now();
        auto res = (double) (time_point_new - time_point).count() /
                   std::chrono::nanoseconds::period::den;
        time_point = time_point_new;
        return res;
    }

private:
    static std::chrono::time_point<std::chrono::steady_clock> time_point;
};