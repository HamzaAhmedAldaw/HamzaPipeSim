#include "pipeline_sim/network.h"
#include <stdexcept>
#include <fstream>
#include <algorithm>

namespace pipeline_sim {

Network::Network() : next_node_id_(0), next_pipe_id_(0) {}

Ptr<Node> Network::add_node(const std::string& id, NodeType type) {
    if (nodes_.find(id) != nodes_.end()) {
        throw std::runtime_error("Node with ID '" + id + "' already exists");
    }
    
    auto node = std::make_shared<Node>(id, type);
    nodes_[id] = node;
    node_index_[id] = next_node_id_++;
    
    // Initialize connectivity
    upstream_pipes_[id] = std::vector<Ptr<Pipe>>();
    downstream_pipes_[id] = std::vector<Ptr<Pipe>>();
    
    return node;
}

Ptr<Node> Network::get_node(const std::string& id) const {
    auto it = nodes_.find(id);
    return (it != nodes_.end()) ? it->second : nullptr;
}

Ptr<Pipe> Network::add_pipe(const std::string& id,
                            Ptr<Node> upstream,
                            Ptr<Node> downstream,
                            Real length,
                            Real diameter) {
    if (pipes_.find(id) != pipes_.end()) {
        throw std::runtime_error("Pipe with ID '" + id + "' already exists");
    }
    
    if (!upstream || !downstream) {
        throw std::runtime_error("Pipe must have valid upstream and downstream nodes");
    }
    
    auto pipe = std::make_shared<Pipe>(id, upstream, downstream, length, diameter);
    pipes_[id] = pipe;
    pipe_index_[id] = next_pipe_id_++;
    
    // Update connectivity
    upstream_pipes_[downstream->id()].push_back(pipe);
    downstream_pipes_[upstream->id()].push_back(pipe);
    
    return pipe;
}

Ptr<Pipe> Network::get_pipe(const std::string& id) const {
    auto it = pipes_.find(id);
    return (it != pipes_.end()) ? it->second : nullptr;
}

std::vector<Ptr<Pipe>> Network::get_upstream_pipes(const Ptr<Node>& node) const {
    auto it = upstream_pipes_.find(node->id());
    return (it != upstream_pipes_.end()) ? it->second : std::vector<Ptr<Pipe>>();
}

std::vector<Ptr<Pipe>> Network::get_downstream_pipes(const Ptr<Node>& node) const {
    auto it = downstream_pipes_.find(node->id());
    return (it != downstream_pipes_.end()) ? it->second : std::vector<Ptr<Pipe>>();
}

void Network::set_pressure(const Ptr<Node>& node, Real pressure) {
    if (!node) {
        throw std::runtime_error("Invalid node pointer");
    }
    node->set_pressure_bc(pressure);
    pressure_specs_[node->id()] = pressure;
}

void Network::set_flow_rate(const Ptr<Node>& node, Real flow_rate) {
    if (!node) {
        throw std::runtime_error("Invalid node pointer");
    }
    node->set_fixed_flow_rate(flow_rate);
    flow_specs_[node->id()] = flow_rate;
}

void Network::load_from_json(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // TODO: Implement JSON parsing
    // For now, just close the file
    file.close();
    
    throw std::runtime_error("JSON parsing not yet implemented");
}

void Network::save_to_json(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create file: " + filename);
    }
    
    // Simple JSON output
    file << "{\n";
    file << "  \"nodes\": [\n";
    
    bool first_node = true;
    for (const auto& [id, node] : nodes_) {
        if (!first_node) file << ",\n";
        file << "    {\n";
        file << "      \"id\": \"" << id << "\",\n";
        file << "      \"type\": " << static_cast<int>(node->type()) << ",\n";
        file << "      \"pressure\": " << node->pressure() << ",\n";
        file << "      \"temperature\": " << node->temperature() << ",\n";
        file << "      \"elevation\": " << node->elevation() << "\n";
        file << "    }";
        first_node = false;
    }
    
    file << "\n  ],\n";
    file << "  \"pipes\": [\n";
    
    bool first_pipe = true;
    for (const auto& [id, pipe] : pipes_) {
        if (!first_pipe) file << ",\n";
        file << "    {\n";
        file << "      \"id\": \"" << id << "\",\n";
        file << "      \"upstream\": \"" << pipe->upstream()->id() << "\",\n";
        file << "      \"downstream\": \"" << pipe->downstream()->id() << "\",\n";
        file << "      \"length\": " << pipe->length() << ",\n";
        file << "      \"diameter\": " << pipe->diameter() << ",\n";
        file << "      \"roughness\": " << pipe->roughness() << "\n";
        file << "    }";
        first_pipe = false;
    }
    
    file << "\n  ]\n";
    file << "}\n";
    
    file.close();
}

bool Network::is_valid() const {
    // Check if we have at least one node and one pipe
    if (nodes_.empty() || pipes_.empty()) {
        return false;
    }
    
    // Check if we have at least one pressure specification
    bool has_pressure_bc = false;
    for (const auto& [id, node] : nodes_) {
        if (node->has_pressure_bc()) {
            has_pressure_bc = true;
            break;
        }
    }
    
    if (!has_pressure_bc && pressure_specs_.empty()) {
        return false;
    }
    
    // Check connectivity
    for (const auto& [id, pipe] : pipes_) {
        if (!pipe->upstream() || !pipe->downstream()) {
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

} // namespace pipeline_sim
