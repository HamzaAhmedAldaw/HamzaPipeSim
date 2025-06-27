// ===== src/equipment.cpp =====
#include "pipeline_sim/equipment.h"
#include <cmath>
#include <algorithm>

namespace pipeline_sim {

// Centrifugal Pump Implementation
CentrifugalPump::CentrifugalPump(const std::string& id)
    : Equipment(id, NodeType::PUMP) {
    
    // Default efficiency curve (quadratic)
    efficiency_func_ = [](Real q) {
        // Peak efficiency at 0.5 normalized flow
        Real q_norm = q / 0.1;  // Normalize to ~0.1 m³/s
        return 0.85 - 0.4 * std::pow(q_norm - 0.5, 2);
    };
}

void CentrifugalPump::calculate(
    Real inlet_pressure,
    Real inlet_temperature,
    Real flow_rate,
    Real& outlet_pressure,
    Real& outlet_temperature
) {
    // Calculate head from pump curve
    Real H = head(flow_rate);
    
    // Apply affinity laws for speed variation
    H *= speed_ratio_ * speed_ratio_;
    
    // Convert head to pressure rise
    Real density = 850.0;  // TODO: Get from fluid properties
    Real delta_p = density * constants::GRAVITY * H;
    
    outlet_pressure = inlet_pressure + delta_p;
    outlet_temperature = inlet_temperature;  // Assume isothermal
    
    // Calculate efficiency and power
    efficiency_ = efficiency_func_(flow_rate);
    Real hydraulic_power = flow_rate * delta_p;
    power_ = hydraulic_power / efficiency_;
}

Real CentrifugalPump::head(Real flow_rate) const {
    // H = a - b*Q - c*Q²
    return a_ - b_ * flow_rate - c_ * flow_rate * flow_rate;
}

// Compressor Implementation
Compressor::Compressor(const std::string& id, Type type)
    : Equipment(id, NodeType::COMPRESSOR), type_(type) {
    
    // Default performance map
    pressure_ratio_func_ = [](Real flow, Real speed) {
        // Simple quadratic map
        Real q_norm = flow / 0.1;
        Real pr = 2.5 - 0.5 * q_norm * q_norm;
        return pr * speed;
    };
    
    efficiency_func_ = [](Real flow, Real speed) {
        return 0.75;  // Constant efficiency
    };
}

void Compressor::calculate(
    Real inlet_pressure,
    Real inlet_temperature,
    Real flow_rate,
    Real& outlet_pressure,
    Real& outlet_temperature
) {
    // Get pressure ratio from performance map
    Real pr = pressure_ratio_func_(flow_rate, 1.0);
    outlet_pressure = inlet_pressure * pr;
    
    // Calculate outlet temperature (polytropic compression)
    Real k = 1.4;  // Specific heat ratio for gas
    Real n = k / (polytropic_efficiency_ * (k - 1) + 1);
    
    outlet_temperature = inlet_temperature * 
                        std::pow(pr, (n - 1) / n);
    
    // Calculate power consumption
    Real work = compression_work(inlet_pressure, outlet_pressure,
                                inlet_temperature, k);
    power_ = work * flow_rate / polytropic_efficiency_;
}

Real Compressor::compression_work(
    Real p1, Real p2, Real T1, Real k
) const {
    // Polytropic compression work per unit mass
    Real n = k / (polytropic_efficiency_ * (k - 1) + 1);
    Real R = 287.0;  // Gas constant for air, J/(kg·K)
    
    return n / (n - 1) * R * T1 * 
           (std::pow(p2/p1, (n-1)/n) - 1);
}

// Control Valve Implementation
ControlValve::ControlValve(const std::string& id)
    : Equipment(id, NodeType::VALVE) {
}

void ControlValve::calculate(
    Real inlet_pressure,
    Real inlet_temperature,
    Real flow_rate,
    Real& outlet_pressure,
    Real& outlet_temperature
) {
    // Valve equation: Q = Cv * sqrt(?P / SG)
    Real cv_eff = effective_cv();
    Real sg = 0.85;  // Specific gravity (TODO: from fluid)
    
    // Calculate pressure drop
    Real delta_p = sg * std::pow(flow_rate * 3600 / cv_eff, 2) * 1e5;
    
    outlet_pressure = inlet_pressure - delta_p;
    outlet_temperature = inlet_temperature;  // No temperature change
    
    // Check for choked flow
    Real critical_pr = 0.5;  // Simplified
    if (outlet_pressure < inlet_pressure * critical_pr) {
        outlet_pressure = inlet_pressure * critical_pr;
    }
}

Real ControlValve::effective_cv() const {
    Real f;  // Valve characteristic function
    
    switch (characteristic_) {
        case Characteristic::LINEAR:
            f = opening_;
            break;
            
        case Characteristic::EQUAL_PERCENTAGE:
            f = std::pow(50.0, opening_ - 1.0);
            break;
            
        case Characteristic::QUICK_OPENING:
            f = std::sqrt(opening_);
            break;
            
        default:
            f = opening_;  // Default to linear
            break;
    }
    
    return cv_ * f;
}

Real ControlValve::required_cv(
    Real flow_rate, Real pressure_drop, Real specific_gravity
) {
    // Q (gpm) = Cv * sqrt(?P (psi) / SG)
    Real q_gpm = flow_rate * 15850.3;  // m³/s to gpm
    Real dp_psi = pressure_drop * 0.0145038;  // Pa to psi
    
    return q_gpm / std::sqrt(dp_psi / specific_gravity);
}

// Separator Implementation
Separator::Separator(const std::string& id, Type type)
    : Equipment(id, NodeType::SEPARATOR), type_(type) {
}

void Separator::calculate(
    Real inlet_pressure,
    Real inlet_temperature,
    Real flow_rate,
    Real& outlet_pressure,
    Real& outlet_temperature
) {
    // Minimal pressure drop through separator
    outlet_pressure = inlet_pressure * 0.98;
    outlet_temperature = inlet_temperature;
    
    // Calculate separation (simplified)
    FluidProperties fluid;  // TODO: Get actual fluid
    calculate_separation(fluid, flow_rate);
}

void Separator::calculate_separation(
    const FluidProperties& fluid,
    Real total_flow
) {
    // Simple separation based on efficiency
    Real gas_in = total_flow * fluid.gas_fraction;
    Real oil_in = total_flow * fluid.oil_fraction;
    Real water_in = total_flow * fluid.water_fraction;
    
    separated_flows_.gas_flow = gas_in * gas_efficiency_;
    
    if (type_ == Type::TWO_PHASE) {
        separated_flows_.oil_flow = oil_in + water_in;
        separated_flows_.water_flow = 0.0;
    } else {
        // Three-phase separation
        separated_flows_.oil_flow = oil_in * liquid_efficiency_;
        separated_flows_.water_flow = water_in * liquid_efficiency_;
    }
}

// Heat Exchanger Implementation
HeatExchanger::HeatExchanger(const std::string& id, Type type)
    : Equipment(id, NodeType::HEAT_EXCHANGER), type_(type) {
}

void HeatExchanger::calculate(
    Real inlet_pressure,
    Real inlet_temperature,
    Real flow_rate,
    Real& outlet_pressure,
    Real& outlet_temperature
) {
    // Pressure drop through exchanger
    outlet_pressure = inlet_pressure * 0.95;  // 5% pressure drop
    
    // Heat transfer calculation
    Real cp = 2000.0;  // J/(kg·K) - specific heat
    Real mass_flow = flow_rate * 850.0;  // kg/s (assuming oil density)
    
    // Effectiveness-NTU method
    Real C = mass_flow * cp;  // Heat capacity rate
    Real NTU = ua_ / C;  // Number of transfer units
    Real effectiveness = 1.0 - std::exp(-NTU);  // Counter-flow
    
    // Heat duty
    heat_duty_ = effectiveness * C * (inlet_temperature - cooling_temp_);
    
    // Outlet temperature
    outlet_temperature = inlet_temperature - heat_duty_ / C;
}

Real HeatExchanger::calculate_heat_transfer(
    Real inlet_temp, Real flow_rate, Real specific_heat
) const {
    Real mass_flow = flow_rate * 850.0;  // Assuming oil
    Real C = mass_flow * specific_heat;
    Real NTU = ua_ / C;
    Real effectiveness = 1.0 - std::exp(-NTU);
    
    return effectiveness * C * (inlet_temp - cooling_temp_);
}

} // namespace pipeline_sim
