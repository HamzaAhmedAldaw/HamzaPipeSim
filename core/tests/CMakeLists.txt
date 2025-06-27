# ===== core/tests/CMakeLists.txt =====
# AI_GENERATED: Test configuration
find_package(GTest REQUIRED)

set(TEST_SOURCES
    test_network.cpp
    test_correlations.cpp
    test_solver.cpp
    test_equipment.cpp
)

add_executable(pipeline_sim_tests ${TEST_SOURCES})

target_link_libraries(pipeline_sim_tests
    PRIVATE
        pipeline_sim_core
        GTest::GTest
        GTest::Main
)

# Add tests
include(GoogleTest)
gtest_discover_tests(pipeline_sim_tests)

# Coverage if enabled
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pipeline_sim_tests PRIVATE --coverage)
    target_link_options(pipeline_sim_tests PRIVATE --coverage)
endif()


if __name__ == "__main__":
    cli()