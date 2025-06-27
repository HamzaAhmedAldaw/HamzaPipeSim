/// AI_GENERATED: Node class definition
/// Generated on: 2025-06-27
#pragma once

#include "pipeline_sim/types.h"
#include <string>
#include <vector>

namespace pipeline_sim {

enum class NodeType {
    SOURCE,
    SINK,
    JUNCTION,
    PUMP,
    COMPRESSOR,
    VALVE,
    SEPARATOR
};

class Node {
public:
    Node(const std::string& id, NodeType type);
    ~Node() = default;

    // Getters
    const std::string& id() const { return id_; }
    NodeType type() const { return type_; }
    Real pressure() const { return pressure_; }
    Real temperature() const { return temperature_; }
    Real elevation() const { return elevation_; }
    
    // Setters
    void set_pressure(Real p) { pressure_ = p; }
    void set_temperature(Real t) { temperature_ = t; }
    void set_elevation(Real e) { elevation_ = e; }
    
    // Flow management
    void add_inflow(Real flow) { total_inflow_ += flow; }
    void add_outflow(Real flow) { total_outflow_ += flow; }
    Real flow_balance() const { return total_inflow_ - total_outflow_; }
    
    // Reset flows for new iteration
    void reset_flows() {
        total_inflow_ = 0.0;
        total_outflow_ = 0.0;
    }

private:
    std::string id_;
    NodeType type_;
    Real pressure_{0.0};
    Real temperature_{constants::STANDARD_TEMPERATURE};
    Real elevation_{0.0};
    Real total_inflow_{0.0};
    Real total_outflow_{0.0};
};

} // namespace pipeline_sim