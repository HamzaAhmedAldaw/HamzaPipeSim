#pragma once

#include "pipeline_sim/types.h"
#include "pipeline_sim/node.h"
#include <string>

namespace pipeline_sim {

/// Pipe segment in the network
class Pipe {
public:
    /// Constructor
    Pipe(const std::string& id,
         Ptr<Node> upstream,
         Ptr<Node> downstream,
         Real length,
         Real diameter,
         Real roughness = 0.000045);  // Default roughness for steel pipe
    
    /// Getters
    const std::string& id() const { return id_; }
    Ptr<Node> upstream() const { return upstream_; }
    Ptr<Node> downstream() const { return downstream_; }
    Real length() const { return length_; }
    Real diameter() const { return diameter_; }
    Real roughness() const { return roughness_; }
    Real inclination() const { return inclination_; }
    Real flow_rate() const { return flow_rate_; }
    Real velocity() const { return velocity_; }
    
    /// Setters
    void set_flow_rate(Real q) { flow_rate_ = q; }
    void set_velocity(Real v) { velocity_ = v; }
    void set_inclination(Real angle) { inclination_ = angle; }
    void set_roughness(Real e) { roughness_ = e; }
    
    /// Calculated properties
    Real area() const;
    Real reynolds_number(Real density, Real viscosity) const;
    Real volume() const;
    Real friction_factor(Real reynolds) const;
    
    /// Heat transfer
    Real wall_temperature() const { return wall_temperature_; }
    Real heat_transfer_coefficient() const { return heat_transfer_coefficient_; }
    void set_wall_temperature(Real T) { wall_temperature_ = T; }
    void set_heat_transfer_coefficient(Real U) { heat_transfer_coefficient_ = U; }
    
    /// Boundary conditions
    bool has_flow_bc() const { return has_flow_bc_; }
    Real flow_bc() const { return flow_bc_; }
    void set_flow_bc(Real q) { 
        has_flow_bc_ = true; 
        flow_bc_ = q; 
    }
    void remove_flow_bc() { has_flow_bc_ = false; }
    
    /// Valve functionality
    bool has_valve() const { return has_valve_; }
    const std::string& valve_id() const { return valve_id_; }
    Real valve_opening() const { return valve_opening_; }
    void set_valve(const std::string& valve_id, Real opening = 1.0) {
        has_valve_ = true;
        valve_id_ = valve_id;
        valve_opening_ = opening;
    }
    void set_valve_opening(Real opening) { 
        valve_opening_ = std::max(0.0, std::min(1.0, opening)); 
    }
    void remove_valve() { has_valve_ = false; }
    
private:
    std::string id_;
    Ptr<Node> upstream_;
    Ptr<Node> downstream_;
    
    // Geometry
    Real length_;
    Real diameter_;
    Real roughness_;
    Real inclination_{0.0};  // Angle from horizontal (radians)
    
    // Flow state
    Real flow_rate_{0.0};
    Real velocity_{0.0};
    
    // Heat transfer
    Real wall_temperature_{288.15};  // K
    Real heat_transfer_coefficient_{0.0};  // W/(m²·K)
    
    // Boundary conditions
    bool has_flow_bc_{false};
    Real flow_bc_{0.0};
    
    // Valve properties
    bool has_valve_{false};
    std::string valve_id_;
    Real valve_opening_{1.0};  // 0 = closed, 1 = fully open
};

} // namespace pipeline_sim