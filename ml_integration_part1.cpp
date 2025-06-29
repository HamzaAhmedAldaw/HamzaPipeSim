/// AI_GENERATED: ML integration implementation
/// Generated on: 2025-06-27

// ===== src/ml_integration.cpp =====
#include "pipeline_sim/ml_integration.h"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>  // Added for std::numeric_limits

namespace pipeline_sim {
namespace ml {

// Feature Extractor Implementation
Vector FeatureExtractor::extract_features(
    const Network& network,
    const SolutionResults& results,
    const FluidProperties& fluid
) {
    std::vector<Real> features;
    
    // Network topology features
    features.push_back(static_cast<Real>(network.nodes().size()));
    features.push_back(static_cast<Real>(network.pipes().size()));
    
    // Count node types
    int sources = 0, sinks = 0, junctions = 0;
    for (const auto& [id, node] : network.nodes()) {
        switch (node->type()) {
            case NodeType::SOURCE: sources++; break;
            case NodeType::SINK: sinks++; break;
            case NodeType::JUNCTION: junctions++; break;
            default: break;
        }
    }
    features.push_back(static_cast<Real>(sources));
    features.push_back(static_cast<Real>(sinks));
    features.push_back(static_cast<Real>(junctions));
    
    // Fluid properties
    features.push_back(fluid.oil_density);
    features.push_back(fluid.gas_density);
    features.push_back(fluid.water_density);
    features.push_back(fluid.oil_viscosity);
    features.push_back(fluid.gas_viscosity);
    features.push_back(fluid.water_viscosity);
    features.push_back(fluid.oil_fraction);
    features.push_back(fluid.gas_fraction);
    features.push_back(fluid.water_fraction);
    features.push_back(fluid.gas_oil_ratio);
    features.push_back(fluid.water_cut);
    
    // Pressure statistics
    std::vector<Real> pressures;
    for (const auto& [id, pressure] : results.node_pressures) {
        pressures.push_back(pressure);
    }
    
    if (!pressures.empty()) {
        // Min, max, mean, std
        Real min_p = *std::min_element(pressures.begin(), pressures.end());
        Real max_p = *std::max_element(pressures.begin(), pressures.end());
        Real mean_p = std::accumulate(pressures.begin(), pressures.end(), 0.0) / pressures.size();
        
        Real variance = 0.0;
        for (Real p : pressures) {
            variance += (p - mean_p) * (p - mean_p);
        }
        Real std_p = std::sqrt(variance / pressures.size());
        
        features.push_back(min_p);
        features.push_back(max_p);
        features.push_back(mean_p);
        features.push_back(std_p);
    } else {
        features.insert(features.end(), 4, 0.0);
    }
    
    // Flow statistics
    std::vector<Real> flows;
    for (const auto& [id, flow] : results.pipe_flow_rates) {
        flows.push_back(std::abs(flow));
    }
    
    if (!flows.empty()) {
        Real min_q = *std::min_element(flows.begin(), flows.end());
        Real max_q = *std::max_element(flows.begin(), flows.end());
        Real mean_q = std::accumulate(flows.begin(), flows.end(), 0.0) / flows.size();
        Real total_q = std::accumulate(flows.begin(), flows.end(), 0.0);
        
        features.push_back(min_q);
        features.push_back(max_q);
        features.push_back(mean_q);
        features.push_back(total_q);
    } else {
        features.insert(features.end(), 4, 0.0);
    }
    
    // Pipe geometry statistics
    std::vector<Real> lengths, diameters;
    for (const auto& [id, pipe] : network.pipes()) {
        lengths.push_back(pipe->length());
        diameters.push_back(pipe->diameter());
    }
    
    if (!lengths.empty()) {
        Real total_length = std::accumulate(lengths.begin(), lengths.end(), 0.0);
        Real mean_diameter = std::accumulate(diameters.begin(), diameters.end(), 0.0) / diameters.size();
        
        features.push_back(total_length);
        features.push_back(mean_diameter);
    } else {
        features.insert(features.end(), 2, 0.0);
    }
    
    // Convert to Eigen vector
    Vector feature_vector(features.size());
    for (size_t i = 0; i < features.size(); ++i) {
        feature_vector(i) = features[i];
    }
    
    return feature_vector;
}

std::vector<std::string> FeatureExtractor::get_feature_names() {
    return {
        "num_nodes", "num_pipes", "num_sources", "num_sinks", "num_junctions",
        "oil_density", "gas_density", "water_density",
        "oil_viscosity", "gas_viscosity", "water_viscosity",
        "oil_fraction", "gas_fraction", "water_fraction",
        "gas_oil_ratio", "water_cut",
        "min_pressure", "max_pressure", "mean_pressure", "std_pressure",
        "min_flow", "max_flow", "mean_flow", "total_flow",
        "total_length", "mean_diameter"
    };
}
