/// AI_GENERATED: Fluid properties definition
/// Generated on: 2025-06-27
#pragma once

#include "pipeline_sim/types.h"
#include <nlohmann/json.hpp>

namespace pipeline_sim {

struct FluidProperties {
    // Densities (kg/m³)
    Real oil_density{850.0};
    Real gas_density{0.85};  // Relative to air at standard conditions
    Real water_density{1025.0};
    
    // Viscosities (Pa·s)
    Real oil_viscosity{0.02};
    Real gas_viscosity{1.8e-5};
    Real water_viscosity{0.001};
    
    // Phase fractions
    Real oil_fraction{1.0};
    Real gas_fraction{0.0};
    Real water_fraction{0.0};
    
    // PVT properties
    Real gas_oil_ratio{0.0};  // sm³/sm³
    Real water_cut{0.0};  // fraction
    Real api_gravity{30.0};
    
    // Surface tension (N/m)
    Real oil_gas_tension{0.03};
    Real oil_water_tension{0.025};
    
    // Compressibility (1/Pa)
    Real oil_compressibility{1e-9};
    Real gas_compressibility{1e-6};
    Real water_compressibility{4.5e-10};
    
    // Temperature dependence
    Real reference_temperature{288.15};  // K
    Real thermal_expansion_oil{0.0007};  // 1/K
    Real thermal_expansion_water{0.0002};  // 1/K
    
    // Computed properties
    Real mixture_density() const;
    Real mixture_viscosity() const;
    Real liquid_holdup(Real flow_pattern) const;
    
    // Load from JSON
    static FluidProperties from_json(const nlohmann::json& j);
    nlohmann::json to_json() const;
};

} // namespace pipeline_sim