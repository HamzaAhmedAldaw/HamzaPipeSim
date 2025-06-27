#include "pipeline_sim/network.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace pipeline_sim {

Network::Network() : next_node_id_(0), next_pipe_id_(0) {}

Ptr<Node> Network::add_node(const std::string& id, NodeType type) {
    auto node = std::make_shared<Node>(id, type);
    nodes_[id] = node;
    node_index_[id] = next_node_id_++;
    
    // Initialize empty pipe lists for this node
    upstream_pipes_[id] = std::vector<Ptr<Pipe>>();
    downstream_pipes_[id] = std::vector<Ptr<Pipe>>();
    
    return node;
}

Ptr<Pipe> Network::add_pipe(const std::string& id,
                             Ptr<Node> upstream,
                             Ptr<Node> downstream,
                             Real length,
                             Real diameter) {
    if (!upstream || !downstream) {
        throw std::runtime_error("Cannot create pipe with null nodes");
    }
    
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
    if (!node) return std::vector<Ptr<Pipe>>{};
    
    auto it = upstream_pipes_.find(node->id());
    return (it != upstream_pipes_.end()) ? it->second : std::vector<Ptr<Pipe>>{};
}

std::vector<Ptr<Pipe>> Network::get_downstream_pipes(const Ptr<Node>& node) const {
    if (!node) return std::vector<Ptr<Pipe>>{};
    
    auto it = downstream_pipes_.find(node->id());
    return (it != downstream_pipes_.end()) ? it->second : std::vector<Ptr<Pipe>>{};
}

void Network::set_pressure(const Ptr<Node>& node, Real pressure) {
    if (!node) return;
    
    node->set_pressure_bc(pressure);
    pressure_specs_[node->id()] = pressure;
}

void Network::set_flow_rate(const Ptr<Node>& node, Real flow_rate) {
    if (!node) return;
    
    node->set_fixed_flow_rate(flow_rate);
    flow_specs_[node->id()] = flow_rate;
}

void Network::load_from_json(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Clear existing network
    clear();
    
    std::string line;
    std::string section;
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Check for section headers
        if (line == "[NODES]") {
            section = "NODES";
            continue;
        } else if (line == "[PIPES]") {
            section = "PIPES";
            continue;
        }
        
        // Parse based on current section
        std::istringstream iss(line);
        
        if (section == "NODES") {
            // Format: id type elevation pressure(optional) flow_rate(optional)
            std::string id;
            int type_int;
            Real elevation = 0.0;
            Real pressure = 0.0;
            Real flow_rate = 0.0;
            
            iss >> id >> type_int;
            
            if (iss >> elevation) {
                // elevation provided
            }
            
            auto node = add_node(id, static_cast<NodeType>(type_int));
            node->set_elevation(elevation);
            
            if (iss >> pressure) {
                set_pressure(node, pressure);
            }
            
            if (iss >> flow_rate) {
                set_flow_rate(node, flow_rate);
            }
            
        } else if (section == "PIPES") {
            // Format: id upstream_id downstream_id length diameter roughness(optional)
            std::string id, upstream_id, downstream_id;
            Real length, diameter;
            Real roughness = 0.000045; // Default for steel
            
            iss >> id >> upstream_id >> downstream_id >> length >> diameter;
            
            auto upstream = get_node(upstream_id);
            auto downstream = get_node(downstream_id);
            
            if (!upstream || !downstream) {
                throw std::runtime_error("Invalid node reference in pipe: " + id);
            }
            
            auto pipe = add_pipe(id, upstream, downstream, length, diameter);
            
            if (iss >> roughness) {
                pipe->set_roughness(roughness);
            }
        }
    }
    
    file.close();
}

void Network::save_to_json(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create file: " + filename);
    }
    
    // Write header
    file << "# Pipeline Network Configuration\n";
    file << "# Generated by HamzaPipeSim\n\n";
    
    // Write nodes section
    file << "[NODES]\n";
    file << "# Format: id type elevation pressure(optional) flow_rate(optional)\n";
    
    for (const auto& [id, node] : nodes_) {
        file << id << " " << static_cast<int>(node->type()) << " " 
             << std::fixed << std::setprecision(3) << node->elevation();
        
        // Check if this node has pressure spec
        auto p_it = pressure_specs_.find(id);
        if (p_it != pressure_specs_.end()) {
            file << " " << p_it->second;
        }
        
        // Check if this node has flow spec
        auto f_it = flow_specs_.find(id);
        if (f_it != flow_specs_.end() && p_it == pressure_specs_.end()) {
            file << " 0"; // Placeholder for pressure
            file << " " << f_it->second;
        }
        
        file << "\n";
    }
    
    // Write pipes section
    file << "\n[PIPES]\n";
    file << "# Format: id upstream_id downstream_id length diameter roughness\n";
    
    for (const auto& [id, pipe] : pipes_) {
        file << id << " " 
             << pipe->upstream()->id() << " " 
             << pipe->downstream()->id() << " "
             << std::fixed << std::setprecision(3)
             << pipe->length() << " " 
             << pipe->diameter() << " " 
             << std::scientific << std::setprecision(6)
             << pipe->roughness() << "\n";
    }
    
    file.close();
}

bool Network::is_valid() const {
    // Check basic requirements
    if (nodes_.empty()) return false;
    if (pipes_.empty()) return false;
    
    // Check for at least one pressure boundary condition
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
    
    // Check connectivity (simplified - just ensure all pipes have valid nodes)
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
