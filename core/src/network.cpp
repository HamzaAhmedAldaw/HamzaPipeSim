/// AI_GENERATED: Network implementation
/// Generated on: 2025-06-27
#include "pipeline_sim/network.h"
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>

namespace pipeline_sim {

Network::Network() : next_node_id_(0), next_pipe_id_(0) {}

Ptr<Node> Network::add_node(const std::string& id, NodeType type) {
    auto node = std::make_shared<Node>(id, type);
    nodes_[id] = node;
    node_index_[id] = next_node_id_++;
    return node;
}

Ptr<Pipe> Network::add_pipe(const std::string& id,
                             Ptr<Node> upstream,
                             Ptr<Node> downstream,
                             Real length,
                             Real diameter) {
    auto pipe = std::make_shared<Pipe>(id, upstream, downstream, length, diameter);
    pipes_[id] = pipe;
    pipe_index_[id] = next_pipe_id_++;
    
    // Update connectivity
    upstream_pipes_[downstream->id()].push_back(pipe);
    downstream_pipes_[upstream->id()].push_back(pipe);
    
    return pipe;
}

Ptr<Node> Network::get_node(const std::string& id) const {
    auto it = nodes_.find(id);
    return (it != nodes_.end()) ? it->second : nullptr;
}

Ptr<Pipe> Network::get_pipe(const std::string& id) const {
    auto it = pipes_.find(id);
    return (it != pipes_.end()) ? it->second : nullptr;
}

std::vector<Ptr<Pipe>> Network::get_upstream_pipes(const Ptr<Node>& node) const {
    auto it = upstream_pipes_.find(node->id());
    return (it != upstream_pipes_.end()) ? it->second : std::vector<Ptr<Pipe>>{};
}

std::vector<Ptr<Pipe>> Network::get_downstream_pipes(const Ptr<Node>& node) const {
    auto it = downstream_pipes_.find(node->id());
    return (it != downstream_pipes_.end()) ? it->second : std::vector<Ptr<Pipe>>{};
}

void Network::set_pressure(const Ptr<Node>& node, Real pressure) {
    node->set_pressure(pressure);
    pressure_specs_[node->id()] = pressure;
}

void Network::set_flow_rate(const Ptr<Node>& node, Real flow_rate) {
    flow_specs_[node->id()] = flow_rate;
}

void Network::load_from_json(const std::string& filename) {
    std::ifstream file(filename);
    nlohmann::json j;
    file >> j;
    
    // Load nodes
    for (const auto& node_json : j["nodes"]) {
        auto node = add_node(
            node_json["id"],
            node_json["type"]
        );
        
        if (node_json.contains("elevation")) {
            node->set_elevation(node_json["elevation"]);
        }
        
        if (node_json.contains("pressure")) {
            set_pressure(node, node_json["pressure"]);
        }
        
        if (node_json.contains("flow_rate")) {
            set_flow_rate(node, node_json["flow_rate"]);
        }
    }
    
    // Load pipes
    for (const auto& pipe_json : j["pipes"]) {
        auto pipe = add_pipe(
            pipe_json["id"],
            get_node(pipe_json["upstream"]),
            get_node(pipe_json["downstream"]),
            pipe_json["length"],
            pipe_json["diameter"]
        );
        
        if (pipe_json.contains("roughness")) {
            pipe->set_roughness(pipe_json["roughness"]);
        }
        
        if (pipe_json.contains("inclination")) {
            pipe->set_inclination(pipe_json["inclination"]);
        }
    }
}

} // namespace pipeline_sim