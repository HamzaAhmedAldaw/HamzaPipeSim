// ===== src/correlations.cpp =====
#include "pipeline_sim/correlations.h"
#include <cmath>
#include <algorithm>

namespace pipeline_sim {

// Beggs-Brill Implementation
FlowCorrelation::Results BeggsBrillCorrelation::calculate(
    const FluidProperties& fluid,
    const Pipe& pipe,
    Real flow_rate,
    Real inlet_pressure,
    Real inlet_temperature
) const {
    Results results;
    
    // Calculate phase velocities
    Real area = pipe.area();
    Real liquid_frac = fluid.oil_fraction + fluid.water_fraction;
    Real vsl = flow_rate * liquid_frac / area;
    Real vsg = flow_rate * fluid.gas_fraction / area;
    Real vm = vsl + vsg;
    
    // No-slip holdup
    Real lambda = vsl / vm;
    
    // Froude number
    Real froude = vm * vm / (constants::GRAVITY * pipe.diameter());
    
    // Determine flow pattern
    results.flow_pattern = determine_flow_pattern(vsl, vsg, 
                                                 pipe.diameter(), 
                                                 pipe.inclination());
    
    // Calculate liquid holdup
    results.liquid_holdup = calculate_liquid_holdup(lambda, froude,
                                                    pipe.inclination(),
                                                    results.flow_pattern);
    
    // Mixture properties
    Real rho_l = fluid.oil_density * fluid.oil_fraction / liquid_frac +
                 fluid.water_density * fluid.water_fraction / liquid_frac;
    Real rho_g = fluid.gas_density * 1.225;
    
    results.mixture_density = rho_l * results.liquid_holdup + 
                             rho_g * (1.0 - results.liquid_holdup);
    results.mixture_velocity = vm;
    
    // Friction factor calculation
    Real mu_l = fluid.oil_viscosity * fluid.oil_fraction / liquid_frac +
                fluid.water_viscosity * fluid.water_fraction / liquid_frac;
    Real mu_n = mu_l * lambda + fluid.gas_viscosity * (1.0 - lambda);
    
    Real Re = results.mixture_density * vm * pipe.diameter() / mu_n;
    
    // Two-phase friction factor multiplier
    Real fn = 16.0 / Re;  // Laminar single-phase
    if (Re > 2300) {
        // Turbulent - use Colebrook-White
        Real eps = pipe.roughness() / pipe.diameter();
        Real a = -2.0 * std::log10(eps/3.7 + 2.51/(Re*std::sqrt(fn)));
        fn = 1.0 / (a * a);
    }
    
    // Two-phase multiplier
    Real y = lambda / (results.liquid_holdup * results.liquid_holdup);
    Real S = std::log(y) / (-0.0523 + 3.182*std::log(y) - 
                            0.8725*std::pow(std::log(y), 2) +
                            0.01853*std::pow(std::log(y), 4));
    
    results.friction_factor = fn * std::exp(S);
    
    // Pressure gradient
    Real dp_friction = 2.0 * results.friction_factor * 
                      results.mixture_density * vm * vm / pipe.diameter();
    
    Real dp_gravity = results.mixture_density * constants::GRAVITY * 
                     std::sin(pipe.inclination());
    
    results.pressure_gradient = dp_friction + dp_gravity;
    
    // Additional data
    results.additional_data["froude_number"] = froude;
    results.additional_data["reynolds_number"] = Re;
    results.additional_data["no_slip_holdup"] = lambda;
    
    return results;
}

FlowPattern BeggsBrillCorrelation::determine_flow_pattern(
    Real vsl, Real vsg, Real diameter, Real inclination
) const {
    Real vm = vsl + vsg;
    Real lambda = vsl / vm;
    Real froude = vm * vm / (constants::GRAVITY * diameter);
    
    // Flow pattern boundaries
    Real L1 = 316.0 * std::pow(lambda, 0.302);
    Real L2 = 0.0009252 * std::pow(lambda, -2.4684);
    Real L3 = 0.10 * std::pow(lambda, -1.4516);
    Real L4 = 0.5 * std::pow(lambda, -6.738);
    
    // Horizontal or downward flow
    if (inclination <= 0) {
        if (lambda < 0.01 && froude < L1) {
            return FlowPattern::SEGREGATED;
        } else if (lambda >= 0.01 && lambda < 0.4) {
            if (froude <= L1) {
                return FlowPattern::DISTRIBUTED;
            } else if (froude <= L2) {
                return FlowPattern::INTERMITTENT;
            }
        } else if (lambda >= 0.4 && froude <= L4) {
            return FlowPattern::INTERMITTENT;
        }
    }
    
    // Default to annular for high gas rates
    return FlowPattern::ANNULAR;
}

Real BeggsBrillCorrelation::calculate_liquid_holdup(
    Real lambda, Real froude, Real angle, FlowPattern pattern
) const {
    // Horizontal liquid holdup correlations
    Real Hl_0;
    
    switch (pattern) {
        case FlowPattern::SEGREGATED:
            Hl_0 = 0.98 * std::pow(lambda, 0.4846) / 
                   std::pow(froude, 0.0868);
            break;
            
        case FlowPattern::INTERMITTENT:
            Hl_0 = 0.845 * std::pow(lambda, 0.5351) / 
                   std::pow(froude, 0.0173);
            break;
            
        case FlowPattern::DISTRIBUTED:
            Hl_0 = 1.065 * std::pow(lambda, 0.5824) / 
                   std::pow(froude, 0.0609);
            break;
            
        default:  // ANNULAR
            Hl_0 = lambda;
    }
    
    // Constrain to physical limits
    Hl_0 = std::max(lambda, std::min(1.0, Hl_0));
    
    // Inclination correction
    if (std::abs(angle) < 0.001) {
        return Hl_0;
    }
    
    // Calculate C parameter
    Real NFr = froude;
    Real NLv = vsl * std::pow(fluid.oil_density / 
                             (constants::GRAVITY * oil_water_tension), 0.25);
    
    Real C = 0.0;
    if (pattern == FlowPattern::SEGREGATED && angle > 0) {
        C = (1.0 - lambda) * std::log(0.011 * std::pow(NFr, 3.539) * 
                                      std::pow(lambda, -3.768));
        C = std::max(0.0, C);
    }
    
    // Inclination factor
    Real psi = 1.0 + C * (std::sin(1.8 * angle) - 
                         0.333 * std::pow(std::sin(1.8 * angle), 3));
    
    return Hl_0 * psi;
}

// Mechanistic Model Implementation
FlowCorrelation::Results MechanisticModel::calculate(
    const FluidProperties& fluid,
    const Pipe& pipe,
    Real flow_rate,
    Real inlet_pressure,
    Real inlet_temperature
) const {
    Results results;
    
    // Calculate velocities
    Real area = pipe.area();
    Real liquid_frac = fluid.oil_fraction + fluid.water_fraction;
    Real vsl = flow_rate * liquid_frac / area;
    Real vsg = flow_rate * fluid.gas_fraction / area;
    
    // Determine flow regime
    auto regime = determine_flow_regime(fluid, pipe, vsl, vsg);
    results.flow_pattern = regime.pattern;
    results.liquid_holdup = regime.holdup;
    
    // Calculate mixture properties
    Real rho_l = fluid.oil_density * fluid.oil_fraction / liquid_frac +
                 fluid.water_density * fluid.water_fraction / liquid_frac;
    Real rho_g = fluid.gas_density * 1.225;
    
    results.mixture_density = rho_l * results.liquid_holdup + 
                             rho_g * (1.0 - results.liquid_holdup);
    
    // Calculate pressure gradient based on regime
    Real dp_friction = 0.0;
    Real dp_gravity = results.mixture_density * constants::GRAVITY * 
                     std::sin(pipe.inclination());
    
    switch (regime.pattern) {
        case FlowPattern::STRATIFIED:
        case FlowPattern::SEGREGATED: {
            // Stratified flow model
            Real dpdz_l = calculate_stratified_pressure_gradient(
                fluid, pipe, vsl, results.liquid_holdup, true
            );
            Real dpdz_g = calculate_stratified_pressure_gradient(
                fluid, pipe, vsg, 1.0 - results.liquid_holdup, false
            );
            dp_friction = std::max(dpdz_l, dpdz_g);
            break;
        }
        
        case FlowPattern::SLUG:
        case FlowPattern::INTERMITTENT: {
            // Slug flow model
            dp_friction = calculate_slug_pressure_gradient(
                fluid, pipe, vsl, vsg, results.liquid_holdup
            );
            break;
        }
        
        case FlowPattern::ANNULAR:
        case FlowPattern::MIST: {
            // Annular flow model
            dp_friction = calculate_annular_pressure_gradient(
                fluid, pipe, vsl, vsg, results.liquid_holdup
            );
            break;
        }
        
        default: {
            // Default homogeneous model
            Real vm = vsl + vsg;
            Real mu_m = fluid.mixture_viscosity();
            Real Re = results.mixture_density * vm * pipe.diameter() / mu_m;
            Real f = calculate_friction_factor(Re, pipe.roughness() / pipe.diameter());
            dp_friction = 2.0 * f * results.mixture_density * vm * vm / pipe.diameter();
        }
    }
    
    results.pressure_gradient = dp_friction + dp_gravity;
    results.mixture_velocity = (vsl + vsg);
    
    return results;
}

// Correlation Factory
std::map<std::string, std::function<std::unique_ptr<FlowCorrelation>()>>& 
CorrelationFactory::registry() {
    static std::map<std::string, std::function<std::unique_ptr<FlowCorrelation>()>> reg;
    return reg;
}

std::unique_ptr<FlowCorrelation> CorrelationFactory::create(const std::string& name) {
    auto& reg = registry();
    auto it = reg.find(name);
    if (it != reg.end()) {
        return it->second();
    }
    
    // Default correlations
    if (name == "Beggs-Brill") {
        return std::make_unique<BeggsBrillCorrelation>();
    } else if (name == "Hagedorn-Brown") {
        return std::make_unique<HagedornBrownCorrelation>();
    } else if (name == "Gray") {
        return std::make_unique<GrayCorrelation>();
    } else if (name == "Mechanistic") {
        return std::make_unique<MechanisticModel>();
    }
    
    throw std::runtime_error("Unknown correlation: " + name);
}

std::vector<std::string> CorrelationFactory::available_correlations() {
    std::vector<std::string> names = {
        "Beggs-Brill", "Hagedorn-Brown", "Gray", "Mechanistic"
    };
    
    for (const auto& [name, creator] : registry()) {
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
