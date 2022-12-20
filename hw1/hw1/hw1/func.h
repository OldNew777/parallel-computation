#include "cxxopts.hpp"

inline char generate_array(int i) {
    return char(i * i + 3 * i + 1);
}

template<typename T>
void fill_buffer(T buff, int size) {
    for (int i = 0; i < size; ++i) {
        buff[i] = generate_array(i);
    }
}

template<typename T>
void reset_buffer(T buff, int size) {
    for (int i = 0; i < size; ++i) {
        buff[i] = 0;
    }
}

template<typename T>
void reset_2d_buffer(T buff, int x, int y) {
    for (int i = 0; i < x; ++i) {
        reset_buffer(&buff[i * y], y);
    }
}

template<typename T>
void reset_3d_buffer(T buff, int x, int y, int z) {
    for (int i = 0; i < x; ++i) {
        reset_2d_buffer(&buff[i * y * z], y, z);
    }
}

template<typename T>
void assert_buffer(T buff, int size) {
    for (int i = 0; i < size; ++i) {
        assert(buff[i] == generate_array(i));
    }
}