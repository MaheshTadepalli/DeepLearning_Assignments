cmake_minimum_required(VERSION 3.10)
project(mydl_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Source files
set(SOURCES
    ops.cpp
    conv.cpp
    bindings.cpp
)

# Build Python module
pybind11_add_module(mydl_cpp ${SOURCES})

# Include current directory
target_include_directories(mydl_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
