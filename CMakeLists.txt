cmake_minimum_required(VERSION 3.20)
project(MonteCarlo VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /GL")
    set(CMAKE_CXX_FLAGS_DEBUG   "/Zi /Od /W4")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -Wall -Wextra")
endif()

find_package(OpenMP REQUIRED)

add_executable(montecarlo
    src/MonteCarloSimulation.cpp
    src/options.cpp
)

target_link_libraries(montecarlo
    PRIVATE
    OpenMP::OpenMP_CXX
)

target_include_directories(montecarlo PRIVATE include)

if(MSVC)
    target_compile_options(montecarlo PRIVATE /openmp)
endif()