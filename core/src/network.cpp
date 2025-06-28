/// AI_GENERATED: Network implementation
/// Generated on: 2025-06-27
#include "pipeline_sim/network.h"
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

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

bool Network::is_valid() const {
    // Check if we have nodes and pipes
    if (nodes_.empty() || pipes_.empty()) {
        return false;
    }
    
    // Check if we have at least one boundary condition
    if (pressure_specs_.empty() && flow_specs_.empty()) {
        return false;
    }
    
    // Check connectivity - all nodes should be connected
    // This is a simple check; a more thorough check would use graph algorithms
    for (const auto& [node_id, node] : nodes_) {
        auto upstream = get_upstream_pipes(node);
        auto downstream = get_downstream_pipes(node);
        
        // Junction nodes must have at least one connection
        if (node->type() == NodeType::JUNCTION && 
            upstream.empty() && downstream.empty()) {
            return false;
        }
    }
    
    return true;
}

void Network::clear() {
    nodes_.clear();
    pipes_.clear();
    node_index_.clear();
    pipe_index_.clear();
    pressure_specs_.clear();
    flow_specs_.clear();
    upstream_pipes_.clear();
    downstream_pipes_.clear();
    next_node_id_ = 0;
    next_pipe_id_ = 0;
}

void Network::save_to_json(const std::string& filename) const {
    nlohmann::json j;
    
    // Save nodes
    j["nodes"] = nlohmann::json::array();
    for (const auto& [id, node] : nodes_) {
        nlohmann::json node_json;
        node_json["id"] = id;
        node_json["type"] = static_cast<int>(node->type());
        node_json["elevation"] = node->elevation();
        
        // Add pressure if specified
        if (pressure_specs_.count(id) > 0) {
            node_json["pressure"] = pressure_specs_.at(id);
        }
        
        // Add flow rate if specified
        if (flow_specs_.count(id) > 0) {
            node_json["flow_rate"] = flow_specs_.at(id);
        }
        
        j["nodes"].push_back(node_json);
    }
    
    // Save pipes
    j["pipes"] = nlohmann::json::array();
    for (const auto& [id, pipe] : pipes_) {
        nlohmann::json pipe_json;
        pipe_json["id"] = id;
        pipe_json["upstream"] = pipe->upstream()->id();
        pipe_json["downstream"] = pipe->downstream()->id();
        pipe_json["length"] = pipe->length();
        pipe_json["diameter"] = pipe->diameter();
        pipe_json["roughness"] = pipe->roughness();
        pipe_json["inclination"] = pipe->inclination();
        
        j["pipes"].push_back(pipe_json);
    }
    
    // Write to file
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    file << j.dump(4);
    file.close();
}

void Network::load_from_json(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    
    nlohmann::json j;
    file >> j;
    
    // Clear existing network
    clear();
    
    // Load nodes
    for (const auto& node_json : j["nodes"]) {
        NodeType type = static_cast<NodeType>(node_json["type"].get<int>());
        auto node = add_node(node_json["id"], type);
        
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
