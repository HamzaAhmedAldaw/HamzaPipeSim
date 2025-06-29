/// AI_GENERATED: Network topology management header
/// Generated on: 2025-06-27
#pragma once

#include "pipeline_sim/types.h"
#include "pipeline_sim/node.h"
#include "pipeline_sim/pipe.h"
#include <map>
#include <vector>
#include <string>
#include <memory>

namespace pipeline_sim {

/**
 * @brief Network class for managing pipeline topology
 * 
 * This class represents a complete pipeline network consisting of nodes
 * (junctions, sources, sinks) and pipes connecting them. It manages the
 * network topology, connectivity information, and boundary conditions.
 */
class Network {
public:
    /// Default constructor
    Network();
    
    /// Destructor
    ~Network() = default;
    
    // ===== Node Management =====
    
    /**
     * @brief Add a node to the network
     * @param id Unique identifier for the node
     * @param type Node type (SOURCE, SINK, JUNCTION)
     * @return Shared pointer to the created node
     */
    Ptr<Node> add_node(const std::string& id, NodeType type);
    
    /**
     * @brief Get a node by ID
     * @param id Node identifier
     * @return Shared pointer to the node (nullptr if not found)
     */
    Ptr<Node> get_node(const std::string& id) const;
    
    /**
     * @brief Get all nodes in the network
     * @return Map of node IDs to node pointers
     */
    const std::map<std::string, Ptr<Node>>& nodes() const { return nodes_; }
    
    // ===== Pipe Management =====
    
    /**
     * @brief Add a pipe to the network
     * @param id Unique identifier for the pipe
     * @param upstream Pointer to upstream node
     * @param downstream Pointer to downstream node
     * @param length Pipe length in meters
     * @param diameter Pipe diameter in meters
     * @return Shared pointer to the created pipe
     */
    Ptr<Pipe> add_pipe(const std::string& id,
                        Ptr<Node> upstream,
                        Ptr<Node> downstream,
                        Real length,
                        Real diameter);
    
    /**
     * @brief Get a pipe by ID
     * @param id Pipe identifier
     * @return Shared pointer to the pipe (nullptr if not found)
     */
    Ptr<Pipe> get_pipe(const std::string& id) const;
    
    /**
     * @brief Get all pipes in the network
     * @return Map of pipe IDs to pipe pointers
     */
    const std::map<std::string, Ptr<Pipe>>& pipes() const { return pipes_; }
    
    // ===== Connectivity Information =====
    
    /**
     * @brief Get pipes flowing into a node
     * @param node Target node
     * @return Vector of pipes with downstream endpoint at this node
     */
    std::vector<Ptr<Pipe>> get_upstream_pipes(const Ptr<Node>& node) const;
    
    /**
     * @brief Get pipes flowing out of a node
     * @param node Source node
     * @return Vector of pipes with upstream endpoint at this node
     */
    std::vector<Ptr<Pipe>> get_downstream_pipes(const Ptr<Node>& node) const;
    
    // ===== Boundary Conditions =====
    
    /**
     * @brief Set pressure boundary condition at a node
     * @param node Node where pressure is specified
     * @param pressure Pressure value in Pa
     */
    void set_pressure(const Ptr<Node>& node, Real pressure);
    
    /**
     * @brief Set flow rate boundary condition at a node
     * @param node Node where flow is specified
     * @param flow_rate Flow rate in m³/s (positive = injection, negative = production)
     */
    void set_flow_rate(const Ptr<Node>& node, Real flow_rate);
    
    /**
     * @brief Get pressure specifications
     * @return Map of node IDs to specified pressures
     */
    const std::map<std::string, Real>& pressure_specs() const { return pressure_specs_; }
    
    /**
     * @brief Get flow specifications
     * @return Map of node IDs to specified flow rates
     */
    const std::map<std::string, Real>& flow_specs() const { return flow_specs_; }
    
    // ===== Indexing (for matrix assembly) =====
    
    /**
     * @brief Get node index for matrix assembly
     * @param node_id Node identifier
     * @return Zero-based index
     */
    size_t node_index(const std::string& node_id) const {
        auto it = node_index_.find(node_id);
        return (it != node_index_.end()) ? it->second : size_t(-1);
    }
    
    /**
     * @brief Get pipe index for matrix assembly
     * @param pipe_id Pipe identifier
     * @return Zero-based index
     */
    size_t pipe_index(const std::string& pipe_id) const {
        auto it = pipe_index_.find(pipe_id);
        return (it != pipe_index_.end()) ? it->second : size_t(-1);
    }
    
    // ===== Serialization =====
    
    /**
     * @brief Load network from JSON file
     * @param filename Path to JSON file
     * @throws std::runtime_error if file cannot be read or parsed
     */
    void load_from_json(const std::string& filename);
    
    /**
     * @brief Save network to JSON file
     * @param filename Path to output JSON file
     */
    void save_to_json(const std::string& filename) const;
    
    // ===== Network Properties =====
    
    /**
     * @brief Get number of nodes
     * @return Total node count
     */
    size_t node_count() const { return nodes_.size(); }
    
    /**
     * @brief Get number of pipes
     * @return Total pipe count
     */
    size_t pipe_count() const { return pipes_.size(); }
    
    /**
     * @brief Check if network is valid for simulation
     * @return true if network has valid topology and boundary conditions
     */
    bool is_valid() const;
    
    /**
     * @brief Clear all nodes and pipes
     */
    void clear();
    
private:
    /// Node storage
    std::map<std::string, Ptr<Node>> nodes_;
    
    /// Pipe storage
    std::map<std::string, Ptr<Pipe>> pipes_;
    
    /// Node ID to index mapping (for matrix assembly)
    std::map<std::string, size_t> node_index_;
    
    /// Pipe ID to index mapping (for matrix assembly)
    std::map<std::string, size_t> pipe_index_;
    
    /// Pressure boundary conditions (node ID -> pressure)
    std::map<std::string, Real> pressure_specs_;
    
    /// Flow boundary conditions (node ID -> flow rate)
    std::map<std::string, Real> flow_specs_;
    
    /// Connectivity: node ID -> pipes flowing into it
    std::map<std::string, std::vector<Ptr<Pipe>>> upstream_pipes_;
    
    /// Connectivity: node ID -> pipes flowing out of it
    std::map<std::string, std::vector<Ptr<Pipe>>> downstream_pipes_;
    
    /// Counter for generating node indices
    size_t next_node_id_;
    
    /// Counter for generating pipe indices
    size_t next_pipe_id_;
};

} // namespace pipeline_sim
