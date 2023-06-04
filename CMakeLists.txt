cmake_minimum_required(VERSION 3.17)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_C_STANDARD 11)

set(CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "-O3")

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    set(GCC_NEON_FLAGS "-mfpu=neon")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_NEON_FLAGS}")
endif()

project(MatrixMultiplication C)
add_executable(MatrixMultiplication main.c)

