#include "timer.h"

std::chrono::time_point<std::chrono::steady_clock> Timer::time_point = std::chrono::high_resolution_clock::now();