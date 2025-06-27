#include "pipeline_sim/correlations.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace pipeline_sim {

// Helper function for friction factor calculation
static Real calculate_friction_factor(Real reynolds, Real roughness, Real diameter) {
    if (reynolds < 2300.0) {
        // Laminar flow
        return 64.0 / reynolds;
    } else {
        // Turbulent flow - Colebrook-White equation
        // Using Swamee-Jain explicit approximation
        Real A = -2.0 * std::log10(roughness / (3.7 * diameter) + 5.74 / std::pow(reynolds, 0.9));
        return 0.25 / (A * A);
    }
}

// SinglePhaseFlow implementation
FlowCorrelation::Results SinglePhaseFlow::calculate(
    const FluidProperties& fluid,
    const Pipe& pipe,
    Real flow_rate,
    Real inlet_pressure,
    Real inlet_temperature
) const {
    Results results;
    
    // Calculate flow properties
    Real area = pipe.area();
    Real velocity = flow_rate / area;
    Real density = fluid.mixture_density();
    Real viscosity = fluid.mixture_viscosity();
    
    // Reynolds number
    Real reynolds = density * std::abs(velocity) * pipe.diameter() / viscosity;
    
    // Friction factor
    Real f = calculate_friction_factor(reynolds, pipe.roughness(), pipe.diameter());
    
    // Pressure gradient components
    Real dp_friction = f * (density * velocity * std::abs(velocity)) / (2.0 * pipe.diameter());
    Real dp_elevation = density * constants::GRAVITY * std::sin(pipe.inclination());
    
    // Fill results
    results.pressure_gradient = dp_friction + dp_elevation;
    results.liquid_holdup = 1.0;  // Single phase
    results.flow_pattern = FlowPattern::SINGLE_PHASE;
    results.friction_factor = f;
    results.mixture_density = density;
    results.mixture_velocity = velocity;
    
    return results;
}

// CoreAnnularFlow implementation
FlowCorrelation::Results CoreAnnularFlow::calculate(
    const FluidProperties& fluid,
    const Pipe& pipe,
    Real flow_rate,
    Real inlet_pressure,
    Real inlet_temperature
) const {
    Results results;
    
    // Core-annular flow assumes oil core with water annulus
    Real area = pipe.area();
    Real oil_fraction = fluid.oil_fraction;
    Real water_fraction = fluid.water_fraction;
    
    // Estimate core and annulus radii
    Real r_pipe = pipe.diameter() / 2.0;
    Real r_core = r_pipe * std::sqrt(oil_fraction);
    Real delta = r_pipe - r_core;  // Annulus thickness
    
    // Velocities
    Real v_oil = flow_rate * oil_fraction / (constants::PI * r_core * r_core);
    Real v_water = flow_rate * water_fraction / (constants::PI * (r_pipe * r_pipe - r_core * r_core));
    
    // Use water properties for wall friction
    Real reynolds = fluid.water_density * v_water * 2 * delta / fluid.water_viscosity;
    Real f = calculate_friction_factor(reynolds, pipe.roughness(), 2 * delta);
    
    // Pressure gradient (simplified)
    Real mixture_density = fluid.mixture_density();
    Real dp_friction = f * fluid.water_density * v_water * v_water / (4.0 * delta);
    Real dp_elevation = mixture_density * constants::GRAVITY * std::sin(pipe.inclination());
    
    // Fill results
    results.pressure_gradient = dp_friction + dp_elevation;
    results.liquid_holdup = 1.0;  // Liquid flow
    results.flow_pattern = FlowPattern::ANNULAR;
    results.friction_factor = f;
    results.mixture_density = mixture_density;
    results.mixture_velocity = flow_rate / area;
    
    return results;
}

// BeggsBrill implementation
FlowCorrelation::Results BeggsBrill::calculate(
    const FluidProperties& fluid,
    const Pipe& pipe,
    Real flow_rate,
    Real inlet_pressure,
    Real inlet_temperature
) const {
    Results results;
    
    // Calculate superficial velocities
    Real area = pipe.area();
    Real vsl = flow_rate * fluid.liquid_fraction() / area;
    Real vsg = flow_rate * fluid.gas_fraction / area;
    Real vm = vsl + vsg;
    
    // No-slip holdup
    Real lambda_l = vsl / vm;
    
    // Froude number
    Real froude = vm * vm / (constants::GRAVITY * pipe.diameter());
    
    // Determine flow pattern
    FlowPattern pattern = determine_flow_pattern(vsl, vsg, pipe.diameter(), pipe.inclination());
    
    // Calculate liquid holdup
    Real holdup = calculate_liquid_holdup(lambda_l, froude, pipe.inclination(), pattern);
    
    // Mixture properties
    Real rho_l = fluid.oil_density * (fluid.oil_fraction / fluid.liquid_fraction()) +
                 fluid.water_density * (fluid.water_fraction / fluid.liquid_fraction());
    Real rho_g = fluid.gas_density;
    Real rho_m = rho_l * holdup + rho_g * (1 - holdup);
    
    // Two-phase Reynolds number
    Real mu_l = fluid.oil_viscosity * (fluid.oil_fraction / fluid.liquid_fraction()) +
                fluid.water_viscosity * (fluid.water_fraction / fluid.liquid_fraction());
    Real reynolds = rho_m * vm * pipe.diameter() / mu_l;
    
    // Friction factor
    Real f = calculate_friction_factor(reynolds, pipe.roughness(), pipe.diameter());
    
    // Pressure gradient
    Real dp_friction = f * rho_m * vm * vm / (2.0 * pipe.diameter());
    Real dp_elevation = rho_m * constants::GRAVITY * std::sin(pipe.inclination());
    
    // Fill results
    results.pressure_gradient = dp_friction + dp_elevation;
    results.liquid_holdup = holdup;
    results.flow_pattern = pattern;
    results.friction_factor = f;
    results.mixture_density = rho_m;
    results.mixture_velocity = vm;
    
    return results;
}

FlowPattern BeggsBrill::determine_flow_pattern(
    Real liquid_velocity,
    Real gas_velocity,
    Real pipe_diameter,
    Real inclination
) const {
    // Simplified Beggs-Brill flow pattern map
    Real vsl = liquid_velocity;
    Real vsg = gas_velocity;
    Real lambda_l = vsl / (vsl + vsg);
    
    // Flow pattern boundaries (simplified)
    if (lambda_l < 0.01 && vsg > 3.0) {
        return FlowPattern::ANNULAR;
    } else if (lambda_l < 0.4 && vsg > 0.5) {
        return FlowPattern::INTERMITTENT;
    } else {
        return FlowPattern::SEGREGATED;
    }
}

Real BeggsBrill::calculate_liquid_holdup(
    Real no_slip_holdup,
    Real froude_number,
    Real inclination,
    FlowPattern pattern
) const {
    // Simplified Beggs-Brill holdup correlation
    Real holdup = no_slip_holdup;
    
    // Correction factor based on flow pattern
    Real C = 0.0;
    switch (pattern) {
        case FlowPattern::SEGREGATED:
            C = 0.98;
            break;
        case FlowPattern::INTERMITTENT:
            C = 0.845;
            break;
        case FlowPattern::DISTRIBUTED:
        case FlowPattern::ANNULAR:
            C = 1.065;
            break;
        default:
            C = 1.0;
    }
    
    // Apply correction
    if (froude_number > 0) {
        holdup = C * no_slip_holdup / std::pow(froude_number, 0.1);
    }
    
    // Limit holdup
    holdup = std::max(no_slip_holdup, std::min(1.0, holdup));
    
    return holdup;
}

// HagedornBrownCorrelation implementation
FlowCorrelation::Results HagedornBrownCorrelation::calculate(
    const FluidProperties& fluid,
    const Pipe& pipe,
    Real flow_rate,
    Real inlet_pressure,
    Real inlet_temperature
) const {
    Results results;
    
    // For vertical flow, use simplified correlation
    Real area = pipe.area();
    Real velocity = flow_rate / area;
    Real density = fluid.mixture_density();
    Real viscosity = fluid.mixture_viscosity();
    
    // Reynolds number
    Real reynolds = density * velocity * pipe.diameter() / viscosity;
    
    // Friction factor
    Real f = calculate_friction_factor(reynolds, pipe.roughness(), pipe.diameter());
    
    // Pressure gradient (vertical flow)
    Real dp_friction = f * density * velocity * velocity / (2.0 * pipe.diameter());
    Real dp_gravity = density * constants::GRAVITY * std::cos(pipe.inclination());
    
    // Fill results
    results.pressure_gradient = dp_friction + dp_gravity;
    results.liquid_holdup = 0.8;  // Simplified
    results.flow_pattern = FlowPattern::SLUG;
    results.friction_factor = f;
    results.mixture_density = density;
    results.mixture_velocity = velocity;
    
    return results;
}

Real HagedornBrownCorrelation::griffith_wallis_correlation(
    Real liquid_velocity,
    Real gas_velocity,
    Real pipe_diameter
) const {
    // Simplified implementation
    return 0.5;
}

Real HagedornBrownCorrelation::cnu_correlation_number(
    Real liquid_viscosity,
    Real liquid_density,
    Real surface_tension,
    Real pipe_diameter
) const {
    // CNu = (liquid_viscosity^4 * g) / (liquid_density * surface_tension^3)
    return std::pow(liquid_viscosity, 4) * constants::GRAVITY / 
           (liquid_density * std::pow(surface_tension, 3));
}

// GrayCorrelation implementation
FlowCorrelation::Results GrayCorrelation::calculate(
    const FluidProperties& fluid,
    const Pipe& pipe,
    Real flow_rate,
    Real inlet_pressure,
    Real inlet_temperature
) const {
    Results results;
    
    // Gray correlation for wet gas
    Real area = pipe.area();
    Real velocity = flow_rate / area;
    
    // Use gas properties primarily
    Real reynolds = fluid.gas_density * velocity * pipe.diameter() / fluid.gas_viscosity;
    Real f = calculate_friction_factor(reynolds, pipe.roughness(), pipe.diameter());
    
    // Pressure gradient
    Real dp_friction = f * fluid.gas_density * velocity * velocity / (2.0 * pipe.diameter());
    Real dp_elevation = fluid.mixture_density() * constants::GRAVITY * std::sin(pipe.inclination());
    
    // Fill results
    results.pressure_gradient = dp_friction + dp_elevation;
    results.liquid_holdup = 0.05;  // Low for wet gas
    results.flow_pattern = FlowPattern::ANNULAR;
    results.friction_factor = f;
    results.mixture_density = fluid.mixture_density();
    results.mixture_velocity = velocity;
    
    return results;
}

// MechanisticModel implementation
FlowCorrelation::Results MechanisticModel::calculate(
    const FluidProperties& fluid,
    const Pipe& pipe,
    Real flow_rate,
    Real inlet_pressure,
    Real inlet_temperature
) const {
    Results results;
    
    // Calculate superficial velocities
    Real area = pipe.area();
    Real vsl = flow_rate * fluid.liquid_fraction() / area;
    Real vsg = flow_rate * fluid.gas_fraction / area;
    
    // Determine flow regime
    FlowRegime regime = determine_flow_regime(fluid, pipe, vsl, vsg);
    
    // Calculate based on flow pattern
    Real density = fluid.mixture_density();
    Real velocity = (vsl + vsg);
    Real reynolds = density * velocity * pipe.diameter() / fluid.mixture_viscosity();
    Real f = calculate_friction_factor(reynolds, pipe.roughness(), pipe.diameter());
    
    // Pressure gradient
    Real dp_friction = f * density * velocity * velocity / (2.0 * pipe.diameter());
    Real dp_elevation = density * constants::GRAVITY * std::sin(pipe.inclination());
    
    // Fill results
    results.pressure_gradient = dp_friction + dp_elevation;
    results.liquid_holdup = regime.holdup;
    results.flow_pattern = regime.pattern;
    results.friction_factor = f;
    results.mixture_density = density;
    results.mixture_velocity = velocity;
    
    return results;
}

MechanisticModel::FlowRegime MechanisticModel::determine_flow_regime(
    const FluidProperties& fluid,
    const Pipe& pipe,
    Real liquid_velocity,
    Real gas_velocity
) const {
    FlowRegime regime;
    
    // Simplified flow regime determination
    Real vsl = liquid_velocity;
    Real vsg = gas_velocity;
    Real vm = vsl + vsg;
    
    if (vsg / vm > 0.9) {
        regime.pattern = FlowPattern::ANNULAR;
        regime.holdup = 0.1;
    } else if (vsl < 0.1 && vsg < 1.0) {
        regime.pattern = FlowPattern::SEGREGATED;
        regime.holdup = stratified_holdup(vsl, vsg, pipe.diameter(), pipe.inclination());
    } else {
        regime.pattern = FlowPattern::SLUG;
        regime.holdup = 0.5;
    }
    
    regime.interfacial_friction = 0.01;
    
    return regime;
}

Real MechanisticModel::stratified_holdup(
    Real liquid_flow,
    Real gas_flow,
    Real pipe_diameter,
    Real inclination
) const {
    // Simplified stratified holdup calculation
    Real ql = liquid_flow;
    Real qg = gas_flow;
    Real qt = ql + qg;
    
    return ql / qt;  // No-slip holdup as approximation
}

// CorrelationFactory implementation
std::map<std::string, std::function<std::unique_ptr<FlowCorrelation>()>>& 
CorrelationFactory::registry() {
    static std::map<std::string, std::function<std::unique_ptr<FlowCorrelation>()>> reg;
    
    // Register default correlations
    static bool initialized = false;
    if (!initialized) {
        reg["single-phase"] = []() { return std::make_unique<SinglePhaseFlow>(); };
        reg["beggs-brill"] = []() { return std::make_unique<BeggsBrill>(); };
        reg["hagedorn-brown"] = []() { return std::make_unique<HagedornBrownCorrelation>(); };
        reg["gray"] = []() { return std::make_unique<GrayCorrelation>(); };
        reg["mechanistic"] = []() { return std::make_unique<MechanisticModel>(); };
        reg["core-annular"] = []() { return std::make_unique<CoreAnnularFlow>(); };
        initialized = true;
    }
    
    return reg;
}

std::unique_ptr<FlowCorrelation> CorrelationFactory::create(const std::string& name) {
    auto& reg = registry();
    auto it = reg.find(name);
    if (it != reg.end()) {
        return it->second();
    }
    throw std::runtime_error("Unknown correlation: " + name);
}

std::vector<std::string> CorrelationFactory::available_correlations() {
    auto& reg = registry();
    std::vector<std::string> names;
    for (const auto& [name, creator] : reg) {
        names.push_back(name);
    }
    return names;
}

void CorrelationFactory::register_correlation(
    const std::string& name,
    std::function<std::unique_ptr<FlowCorrelation>()> creator
) {
    registry()[name] = creator;
}

} // namespace pipeline_sim
