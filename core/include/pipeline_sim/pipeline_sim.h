#pragma once

#include "pipeline_sim/types.h"
#include "pipeline_sim/network.h"
#include "pipeline_sim/node.h"
#include "pipeline_sim/pipe.h"
#include "pipeline_sim/fluid_properties.h"
#include "pipeline_sim/solver.h"
#include "pipeline_sim/correlations.h"
#include "pipeline_sim/transient_solver.h"
#include "pipeline_sim/equipment.h"
#include "pipeline_sim/ml_integration.h"

namespace pipeline_sim {

/// Library version information
constexpr const char* VERSION = "0.1.0";
constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 1;
constexpr int VERSION_PATCH = 0;

/// Initialize the library (optional)
inline void initialize() {
    // Initialization code if needed
}

/// Get version string
inline const char* get_version() {
    return VERSION;
}

} // namespace pipeline_sim