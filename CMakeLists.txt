cmake_minimum_required(VERSION 2.8)
SET(CMAKE_C_COMPILER "/usr/bin/gcc")
SET(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++11 -Wall -ggdb -fPIC -pg")

add_executable(main problem2/main.cpp problem2/order_worker.cpp)
target_link_libraries(main pthread)
