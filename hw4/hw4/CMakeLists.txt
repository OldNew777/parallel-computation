cmake_minimum_required(VERSION 3.21)
project(hw4)

set(CMAKE_CXX_STANDARD 23)

FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(MAIN_SOURCES
        main.cpp
        log.h log.cpp
        SparseMatrix.h SparseMatrix.cpp
        timer.h timer.cpp
        pool.h
        rand_sampler.h
        type.h
        cxxopts.hpp)
add_executable(hw4 ${MAIN_SOURCES})
