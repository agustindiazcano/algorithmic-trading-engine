cmake_minimum_required(VERSION 3.10)
project(WBenchmark)

set(CMAKE_CXX_STANDARD 17)

# Performance flags
if(MSVC)
    # Windows/MSVC
    add_compile_options(/O2 /arch:AVX2 /fp:fast)
    # OpenMP for MSVC
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        link_libraries(OpenMP::OpenMP_CXX)
    endif()
else()
    # GCC/Clang
    add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra)
    # OpenMP
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        link_libraries(OpenMP::OpenMP_CXX)
        add_compile_options(-fopenmp)
    endif()
endif()

add_executable(bench main.cpp)
