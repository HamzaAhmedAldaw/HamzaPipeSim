/// AI_GENERATED: Pipe segment class
/// Generated on: 2025-06-27
#pragma once

#include "pipeline_sim/types.h"
#include "pipeline_sim/node.h"
#include <string>

namespace pipeline_sim {

class Pipe {
public:
    Pipe(const std::string& id, 
         Ptr<Node> upstream, 
         Ptr<Node> downstream,
         Real length,
         Real diameter);
    ~Pipe() = default;

    // Getters
    const std::string& id() const { return id_; }
    Ptr<Node> upstream() const { return upstream_; }
    Ptr<Node> downstream() const { return downstream_; }
    Real length() const { return length_; }
    Real diameter() const { return diameter_; }
    Real roughness() const { return roughness_; }
    Real inclination() const { return inclination_; }
    
    // Setters
    void set_roughness(Real r) { roughness_ = r; }
    void set_inclination(Real i) { inclination_ = i; }
    void set_wall_thickness(Real t) { wall_thickness_ = t; }
    void set_thermal_conductivity(Real k) { thermal_conductivity_ = k; }
    
    // Computed properties
    Real area() const { return constants::PI * diameter_ * diameter_ / 4.0; }
    Real volume() const { return area() * length_; }
    Real hydraulic_diameter() const { return diameter_; }
    
    // Flow properties
    Real flow_rate() const { return flow_rate_; }
    Real velocity() const { return flow_rate_ / area(); }
    Real reynolds_number(Real viscosity, Real density) const;
    Real friction_factor(Real reynolds) const;
    
    // Set flow results
    void set_flow_rate(Real q) { flow_rate_ = q; }
    void set_pressure_gradient(Real dp_dx) { pressure_gradient_ = dp_dx; }
    
private:
    std::string id_;
    Ptr<Node> upstream_;
    Ptr<Node> downstream_;
    Real length_;
    Real diameter_;
    Real roughness_{0.000045};  // Default steel pipe roughness (m)
    Real inclination_{0.0};  // Angle from horizontal (radians)
    Real wall_thickness_{0.01};  // m
    Real thermal_conductivity_{45.0};  // W/(m·K) for steel
    
    // Flow state
    Real flow_rate_{0.0};
    Real pressure_gradient_{0.0};
};

} // namespace pipeline_sim