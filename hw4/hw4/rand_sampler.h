//
// Created by ChenXin on 2022/12/20.
//

#include <random>

#pragma once

class RandSampler {
private:
    std::default_random_engine generator;

    RandSampler() {
        generator.seed(124867);
    }

public:
    RandSampler(RandSampler const &) = delete;
    RandSampler &operator=(RandSampler const &) = delete;

    static RandSampler &Global() {
        static RandSampler instance;
        return instance;
    }

    double rand_double(double min, double max) {
        std::uniform_real_distribution<double> distribution(min, max);
        return distribution(generator);
    }

    int rand_int(int min, int max) {
        std::uniform_int_distribution<int> distribution(min, max);
        return distribution(generator);
    }
};