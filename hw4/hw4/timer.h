#pragma once

#include <chrono>

class Timer {
public:
	static void Tik() {
		time_point = std::chrono::high_resolution_clock::now();
	}
	static double Tok() {
		return (double)(std::chrono::high_resolution_clock::now() - time_point).count() / std::chrono::nanoseconds::period::den;
	}

private:
	static std::chrono::time_point<std::chrono::steady_clock> time_point;
};