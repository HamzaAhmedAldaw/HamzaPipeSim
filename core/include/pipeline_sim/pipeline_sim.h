/// AI_GENERATED: Main header file
/// Generated on: 2025-06-27
#pragma once

#include "pipeline_sim/network.h"
#include "pipeline_sim/node.h"
#include "pipeline_sim/pipe.h"
#include "pipeline_sim/fluid_properties.h"
#include "pipeline_sim/solver.h"
#include "pipeline_sim/correlations.h"
#include "pipeline_sim/types.h"
#include "pipeline_sim/utils.h"

namespace pipeline_sim {

/// Library version information
constexpr const char* VERSION = "0.1.0";
constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 1;
constexpr int VERSION_PATCH = 0;

/// Initialize the library (optional)
void initialize();

/// Get version string
const char* get_version();

} // namespace pipeline_sim