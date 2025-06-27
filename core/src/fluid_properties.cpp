
// ===== fluid_properties.cpp =====
#include "pipeline_sim/fluid_properties.h"
#include <cmath>

namespace pipeline_sim {

Real FluidProperties::mixture_density() const {
    // Volume-weighted average density
    Real total_volume_fraction = oil_fraction + gas_fraction + water_fraction;
    if (total_volume_fraction <= 0) return oil_density;
    
    Real density = (oil_fraction * oil_density + 
                   gas_fraction * gas_density * 1.225 +  // Convert relative to absolute
                   water_fraction * water_density) / total_volume_fraction;
    
    return density;
}

Real FluidProperties::mixture_viscosity() const {
    // Use Dukler correlation for mixture viscosity
    Real liquid_fraction = oil_fraction + water_fraction;
    Real total_fraction = liquid_fraction + gas_fraction;
    
    if (total_fraction <= 0) return oil_viscosity;
    
    // Liquid phase viscosity (volume-weighted)
    Real liquid_viscosity = oil_viscosity;
    if (liquid_fraction > 0) {
        liquid_viscosity = (oil_fraction * oil_viscosity + 
                          water_fraction * water_viscosity) / liquid_fraction;
    }
    
    // No-slip mixture viscosity
    Real no_slip_viscosity = (liquid_fraction * liquid_viscosity + 
                            gas_fraction * gas_viscosity) / total_fraction;
    
    // Apply slip correction (simplified)
    Real slip_factor = 1.0 + 0.3 * gas_fraction;
    
    return no_slip_viscosity * slip_factor;
}

Real FluidProperties::liquid_holdup(Real flow_pattern) const {
    // Simplified liquid holdup calculation
    // flow_pattern: 0=segregated, 1=intermittent, 2=distributed, 3=annular
    
    Real liquid_fraction = oil_fraction + water_fraction;
    Real total_fraction = liquid_fraction + gas_fraction;
    
    if (gas_fraction <= 0 || total_fraction <= 0) return 1.0;
    
    Real no_slip_holdup = liquid_fraction / total_fraction;
    
    // Apply flow pattern correction
    Real correction = 1.0;
    if (flow_pattern < 1.0) {
        // Segregated flow - significant slip
        correction = 0.8;
    } else if (flow_pattern < 2.0) {
        // Intermittent flow - moderate slip
        correction = 0.9;
    } else {
        // Distributed/annular - minimal slip
        correction = 0.95;
    }
    
    return no_slip_holdup * correction;
}

FluidProperties FluidProperties::from_json(const nlohmann::json& j) {
    FluidProperties props;
    
    if (j.contains("oil_density")) props.oil_density = j["oil_density"];
    if (j.contains("gas_density")) props.gas_density = j["gas_density"];
    if (j.contains("water_density")) props.water_density = j["water_density"];
    
    if (j.contains("oil_viscosity")) props.oil_viscosity = j["oil_viscosity"];
    if (j.contains("gas_viscosity")) props.gas_viscosity = j["gas_viscosity"];
    if (j.contains("water_viscosity")) props.water_viscosity = j["water_viscosity"];
    
    if (j.contains("gas_oil_ratio")) props.gas_oil_ratio = j["gas_oil_ratio"];
    if (j.contains("water_cut")) props.water_cut = j["water_cut"];
    if (j.contains("api_gravity")) props.api_gravity = j["api_gravity"];
    
    // Calculate phase fractions from GOR and water cut
    Real total_liquid = 1.0 - props.water_cut;
    props.oil_fraction = total_liquid;
    props.water_fraction = props.water_cut;
    props.gas_fraction = props.gas_oil_ratio / 1000.0;  // Simplified conversion
    
    return props;
}

nlohmann::json FluidProperties::to_json() const {
    return nlohmann::json{
        {"oil_density", oil_density},
        {"gas_density", gas_density},
        {"water_density", water_density},
        {"oil_viscosity", oil_viscosity},
        {"gas_viscosity", gas_viscosity},
        {"water_viscosity", water_viscosity},
        {"oil_fraction", oil_fraction},
        {"gas_fraction", gas_fraction},
        {"water_fraction", water_fraction},
        {"gas_oil_ratio", gas_oil_ratio},
        {"water_cut", water_cut},
        {"api_gravity", api_gravity}
    };
}

} // namespace pipeline_sim
