cmake_minimum_required(VERSION 3.16)
project(system)

set(CMAKE_CXX_STANDARD 14)

add_executable(system src/main.cpp src/data.h src/data.cpp)