#pragma once

#include "pipeline_sim/types.h"
#include <string>

namespace pipeline_sim {

/// Node types
enum class NodeType {
    JUNCTION,
    SOURCE,
    SINK,
    PUMP,
    COMPRESSOR,
    VALVE,          // Control valves
    SEPARATOR,      // Gas/liquid separators  
    HEAT_EXCHANGER  // Heat exchangers
};

/// Node in the pipeline network
class Node {
public:
    /// Constructor
    Node(const std::string& id, NodeType type = NodeType::JUNCTION);
    
    /// Getters
    const std::string& id() const { return id_; }
    NodeType type() const { return type_; }
    Real pressure() const { return pressure_; }
    Real temperature() const { return temperature_; }
    Real elevation() const { return elevation_; }
    
    /// Setters
    void set_pressure(Real p) { pressure_ = p; }
    void set_temperature(Real T) { temperature_ = T; }
    void set_elevation(Real z) { elevation_ = z; }
    void set_type(NodeType type) { type_ = type; }
    
    /// Boundary conditions
    bool has_pressure_bc() const { return has_pressure_bc_; }
    Real pressure_bc() const { return pressure_bc_; }
    void set_pressure_bc(Real p) { 
        has_pressure_bc_ = true; 
        pressure_bc_ = p; 
    }
    void remove_pressure_bc() { has_pressure_bc_ = false; }
    
    /// Fixed flow (for sources/sinks)
    Real fixed_flow_rate() const { return fixed_flow_rate_; }
    void set_fixed_flow_rate(Real q) { fixed_flow_rate_ = q; }
    
    /// Pump functionality
    Real pump_speed() const { return pump_speed_; }
    void set_pump_speed(Real speed) { pump_speed_ = speed; }
    Real pump_curve_coefficient_a() const { return pump_curve_a_; }
    Real pump_curve_coefficient_b() const { return pump_curve_b_; }
    void set_pump_curve(Real a, Real b) { 
        pump_curve_a_ = a; 
        pump_curve_b_ = b; 
    }
    
    /// Compressor functionality
    Real compressor_ratio() const { return compressor_ratio_; }
    void set_compressor_ratio(Real ratio) { compressor_ratio_ = ratio; }
    
private:
    std::string id_;
    NodeType type_;
    
    // State variables
    Real pressure_{101325.0};  // Pa
    Real temperature_{288.15};  // K
    Real elevation_{0.0};      // m
    
    // Boundary conditions
    bool has_pressure_bc_{false};
    Real pressure_bc_{0.0};
    
    // Source/sink properties
    Real fixed_flow_rate_{0.0};  // m³/s
    
    // Pump properties
    Real pump_speed_{1.0};  // Normalized speed (0-1)
    Real pump_curve_a_{0.0};  // Head = a - b*Q²
    Real pump_curve_b_{0.0};
    
    // Compressor properties
    Real compressor_ratio_{1.0};  // Pressure ratio
};

} // namespace pipeline_sim