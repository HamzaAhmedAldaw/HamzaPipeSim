# AI_GENERATED: Root CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(PipelineSim VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Find dependencies
find_package(Threads REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
endif()

# Set compile options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
endif()

# Add subdirectories
add_subdirectory(core)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install rules
install(EXPORT PipelineSimTargets
    FILE PipelineSimTargets.cmake
    NAMESPACE PipelineSim::
    DESTINATION lib/cmake/PipelineSim
)

# Package configuration
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PipelineSimConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/PipelineSimConfig.cmake"
    INSTALL_DESTINATION lib/cmake/PipelineSim
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/PipelineSimConfig.cmake"
    DESTINATION lib/cmake/PipelineSim
)
