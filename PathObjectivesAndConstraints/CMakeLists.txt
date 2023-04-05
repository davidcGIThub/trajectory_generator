cmake_minimum_required(VERSION 3.16.3)
project(PathObjectivesAndConstraints
        VERSION 0.0.1  #[[Specifies the version of the project]] 
        LANGUAGES CXX) #[[Selects which programming languages are needed to build project
                          these include C, CXX, CUDA, OBJC, OBJCXX, Fortran, HIP, ISPC, ASM]]

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED 1)
SET(CMAKE_INCLUDE_CURRENT_DIR ON)

FIND_PACKAGE(GTest REQUIRED)
FIND_PACKAGE(Eigen3 3.3 REQUIRED)

link_directories(${Eigen_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(tests)