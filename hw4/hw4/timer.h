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
        return res;
    }

    void tik() {
        time_point_private = std::chrono::high_resolution_clock::now();
    }
    double toc() {
        auto time_point_new = std::chrono::high_resolution_clock::now();
        auto res = (double) (time_point_new - time_point_private).count() /
                   std::chrono::nanoseconds::period::den;
        return res;
    }

private:
    static std::chrono::time_point<std::chrono::steady_clock> time_point;

    std::chrono::time_point<std::chrono::steady_clock> time_point_private;
};