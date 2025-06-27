# AI_GENERATED: Core library CMakeLists.txt
add_library(pipeline_sim_core
    src/network.cpp
    src/node.cpp
    src/pipe.cpp
    src/fluid_properties.cpp
    src/solver.cpp
    src/correlations.cpp
    src/utils.cpp
)

target_include_directories(pipeline_sim_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(pipeline_sim_core
    PUBLIC
        Eigen3::Eigen
        Threads::Threads
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
)

# Install library
install(TARGETS pipeline_sim_core
    EXPORT PipelineSimTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Build tests if enabled
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()