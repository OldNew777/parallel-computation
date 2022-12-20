cmake_minimum_required(VERSION 3.21)
project(hw4)

set(CMAKE_CXX_STANDARD 23)

set(MAIN_SOURCES
        main.cpp
        log.h log.cpp
        SparceMatrix.h
        timer.h timer.cpp
        pool.h)
add_executable(hw4 ${MAIN_SOURCES})
