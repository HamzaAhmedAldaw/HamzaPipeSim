// ===== CHANGES NEEDED IN ml_integration.cpp =====

// 1. Update includes at top:
#include "pipeline_sim/ml_integration.h"
#include "pipeline_sim/network.h"
#include "pipeline_sim/fluid_properties.h"
#include "pipeline_sim/pipe.h"
#include "pipeline_sim/node.h"
// Remove solver.h include - we don't need it directly

#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>


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
    const auto& nodes = network.nodes();
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        const auto& node = it->second;
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
    for (auto it = results.node_pressures.begin(); it != results.node_pressures.end(); ++it) {
        pressures.push_back(it->second);
    }
    
    if (!pressures.empty()) {
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
    for (auto it = results.pipe_flow_rates.begin(); it != results.pipe_flow_rates.end(); ++it) {
        flows.push_back(std::abs(it->second));
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
    const auto& pipes = network.pipes();
    for (auto it = pipes.begin(); it != pipes.end(); ++it) {
        const auto& pipe = it->second;
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

void FeatureExtractor::normalize_features(Vector& features) {
    // Pressure normalization (Pa)
    Real pressure_scale = 100e5;  // 100 bar
    for (int i = 16; i < 20 && i < features.size(); ++i) {
        features(i) /= pressure_scale;
    }
    
    // Flow normalization (m³/s)
    Real flow_scale = 1.0;
    for (int i = 20; i < 24 && i < features.size(); ++i) {
        features(i) /= flow_scale;
    }
    
    // Length normalization (m)
    if (features.size() > 24) {
        features(24) /= 10000.0;  // 10 km
    }
    
    // Diameter normalization (m)
    if (features.size() > 25) {
        features(25) /= 1.0;
    }
}

// Neural Network Implementation
Vector FlowPatternPredictor::NeuralNetwork::forward(const Vector& input) {
    Vector activation = input;
    
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        activation = weights[layer] * activation + biases[layer];
        
        if (layer < weights.size() - 1) {
            activation = activation.cwiseMax(0.0);
        } else {
            Real max_val = activation.maxCoeff();
            activation = (activation.array() - max_val).exp();
            activation /= activation.sum();
        }
    }
    
    return activation;
}

void FlowPatternPredictor::load(const std::string& filename) {
    network_ = std::make_unique<NeuralNetwork>();
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open model file: " + filename);
    }
    
    int num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
    
    network_->weights.resize(num_layers);
    network_->biases.resize(num_layers);
    
    for (int i = 0; i < num_layers; ++i) {
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        
        network_->weights[i].resize(rows, cols);
        network_->biases[i].resize(rows);
        
        file.read(reinterpret_cast<char*>(network_->weights[i].data()),
                 rows * cols * sizeof(Real));
        file.read(reinterpret_cast<char*>(network_->biases[i].data()),
                 rows * sizeof(Real));
    }
}

Vector FlowPatternPredictor::predict(const Vector& features) {
    if (!network_) {
        throw std::runtime_error("Model not loaded");
    }
    
    return network_->forward(features);
}

FlowPattern FlowPatternPredictor::predict_pattern(
    const Pipe& pipe,
    const FluidProperties& fluid,
    Real flow_rate
) {
    Vector features(10);
    
    Real area = pipe.area();
    Real liquid_frac = fluid.oil_fraction + fluid.water_fraction;
    Real vsl = flow_rate * liquid_frac / area;
    Real vsg = flow_rate * fluid.gas_fraction / area;
    Real vm = vsl + vsg;
    
    features(0) = vsl;
    features(1) = vsg;
    features(2) = pipe.diameter();
    features(3) = pipe.inclination();
    features(4) = fluid.oil_density;
    features(5) = fluid.gas_density;
    features(6) = fluid.oil_viscosity;
    features(7) = fluid.gas_viscosity;
    features(8) = vm * vm / (9.81 * pipe.diameter());
    features(9) = vm > 0 ? vsl / vm : 0;
    
    FeatureExtractor::normalize_features(features);
    
    if (network_) {
        Vector output = network_->forward(features);
        
        int max_idx = 0;
        Real max_prob = output(0);
        for (int i = 1; i < output.size(); ++i) {
            if (output(i) > max_prob) {
                max_prob = output(i);
                max_idx = i;
            }
        }
        
        return static_cast<FlowPattern>(max_idx);
    }
    
    return FlowPattern::SLUG;
}

// Anomaly Detector Implementation
Real AnomalyDetector::IsolationTree::path_length(const Vector& sample) const {
    Node* current = root.get();
    Real length = 0.0;
    
    while (current && (current->left || current->right)) {
        if (sample(current->feature_index) < current->split_value) {
            current = current->left.get();
        } else {
            current = current->right.get();
        }
        length += 1.0;
    }
    
    if (current) {
        int remaining_depth = 10 - current->depth;
        if (remaining_depth > 0) {
            length += 2.0 * (std::log(static_cast<double>(remaining_depth)) + 0.5772) - 
                     2.0 * (remaining_depth - 1.0) / remaining_depth;
        }
    }
    
    return length;
}

void AnomalyDetector::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open model file: " + filename);
    }
    
    int num_trees;
    file.read(reinterpret_cast<char*>(&num_trees), sizeof(int));
    
    forest_.resize(num_trees);
    
    for (int i = 0; i < num_trees; ++i) {
        forest_[i].root = std::make_unique<IsolationTree::Node>();
    }
}

Vector AnomalyDetector::predict(const Vector& features) {
    Vector scores(1);
    
    Real avg_path_length = 0.0;
    for (size_t i = 0; i < forest_.size(); ++i) {
        avg_path_length += forest_[i].path_length(features);
    }
    if (!forest_.empty()) {
        avg_path_length /= forest_.size();
    }
    
    Real c_n = 2.0 * (std::log(static_cast<double>(features.size())) + 0.5772) - 
               2.0 * (features.size() - 1.0) / features.size();
    scores(0) = std::pow(2.0, -avg_path_length / c_n);
    
    return scores;
}

AnomalyDetector::AnomalyResult AnomalyDetector::detect(
    const Network& network,
    const SolutionResults& results
) {
    AnomalyResult result;
    
    FluidProperties fluid;  // TODO: Get from network
    Vector features = FeatureExtractor::extract_features(network, results, fluid);
    FeatureExtractor::normalize_features(features);
    
    Vector scores = predict(features);
    result.anomaly_score = scores(0);
    
    Real threshold = 0.6;
    result.is_anomaly = result.anomaly_score > threshold;
    
    if (result.is_anomaly) {
        auto feature_names = FeatureExtractor::get_feature_names();
        
        for (size_t i = 0; i < static_cast<size_t>(features.size()) && i < feature_names.size(); ++i) {
            if (std::abs(features(static_cast<int>(i))) > 2.0) {
                result.anomaly_features.push_back(feature_names[i]);
            }
        }
    }
    
    return result;
}

// ML Optimizer Implementation
MLOptimizer::OptimizationResult MLOptimizer::optimize(
    Network& network,
    const FluidProperties& fluid,
    const OptimizationObjective& objective,
    const OptimizationConstraints& constraints
) {
    OptimizationResult result;
    result.success = false;
    
    initialize_population(50);
    
    const int max_generations = 100;
    for (int gen = 0; gen < max_generations; ++gen) {
        evaluate_fitness(network, fluid, objective);
        
        Real best_fitness = population_[0].fitness;
        if (best_fitness < 1e-6) {
            result.success = true;
            break;
        }
        
        selection();
        crossover();
        mutation();
    }
    
    if (!population_.empty()) {
        const auto& best = population_[0];
        result.objective_value = best.fitness;
        
        size_t idx = 0;
        const auto& nodes = network.nodes();
        for (auto it = nodes.begin(); it != nodes.end(); ++it) {
            const auto& node = it->second;
            if (node->type() == NodeType::PUMP && idx < static_cast<size_t>(best.genes.size())) {
                result.pump_speeds[it->first] = best.genes(static_cast<int>(idx++));
            }
        }
    }
    
    return result;
}

void MLOptimizer::initialize_population(size_t size) {
    population_.resize(size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    int num_vars = 10;
    
    for (size_t i = 0; i < population_.size(); ++i) {
        population_[i].genes.resize(num_vars);
        for (int j = 0; j < num_vars; ++j) {
            population_[i].genes(j) = dis(gen);
        }
        population_[i].fitness = std::numeric_limits<Real>::max();
    }
}

void MLOptimizer::evaluate_fitness(
    Network& network,
    const FluidProperties& fluid,
    const OptimizationObjective& objective
) {
    for (size_t i = 0; i < population_.size(); ++i) {
        // Apply control variables to network
        // (Simplified - would actually modify pump speeds, valve openings, etc.)
        
        // Run simulation using callback instead of direct solver
        SolutionResults results;
        if (solver_callback_) {
            results = solver_callback_(network, fluid);
        } else {
            // Fallback: create dummy results
            results.converged = false;
            results.residual = 1e10;
        }
        
        // Rest of the function remains the same...
        if (results.converged) {
            switch (objective.type) {
                // ... existing cases ...
            }
        } else {
            population_[i].fitness = std::numeric_limits<Real>::max();
        }
    }
    
    // Sort by fitness
    std::sort(population_.begin(), population_.end(),
              [](const Individual& a, const Individual& b) {
                  return a.fitness < b.fitness;
              });
}

void MLOptimizer::selection() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, static_cast<int>(population_.size()) - 1);
    
    std::vector<Individual> new_population;
    new_population.reserve(population_.size());
    
    new_population.push_back(population_[0]);
    
    while (new_population.size() < population_.size()) {
        int idx1 = dis(gen);
        int idx2 = dis(gen);
        
        if (population_[idx1].fitness < population_[idx2].fitness) {
            new_population.push_back(population_[idx1]);
        } else {
            new_population.push_back(population_[idx2]);
        }
    }
    
    population_ = std::move(new_population);
}

void MLOptimizer::crossover() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    const Real crossover_rate = 0.8;
    
    for (size_t i = 1; i < population_.size() - 1; i += 2) {
        if (dis(gen) < crossover_rate) {
            int point = std::uniform_int_distribution<>(0, 
                static_cast<int>(population_[i].genes.size()) - 1)(gen);
            
            for (int j = point; j < population_[i].genes.size(); ++j) {
                std::swap(population_[i].genes(j), population_[i + 1].genes(j));
            }
        }
    }
}

void MLOptimizer::mutation() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::normal_distribution<> mutation_dis(0.0, 0.1);
    
    const Real mutation_rate = 0.1;
    
    for (size_t i = 1; i < population_.size(); ++i) {
        for (int j = 0; j < population_[i].genes.size(); ++j) {
            if (dis(gen) < mutation_rate) {
                population_[i].genes(j) += mutation_dis(gen);
                population_[i].genes(j) = std::max(0.0, std::min(1.0, population_[i].genes(j)));
            }
        }
    }
}

// Data-Driven Correlation Implementation
Real DataDrivenCorrelation::DecisionTree::predict(const Vector& features) const {
    Node* current = root.get();
    
    while (current && !current->is_leaf()) {
        if (features(current->feature_index) < current->threshold) {
            current = current->left.get();
        } else {
            current = current->right.get();
        }
    }
    
    return current ? current->value : 0.0;
}

void DataDrivenCorrelation::train(
    const std::vector<Vector>& features,
    const std::vector<Real>& pressure_drops
) {
    const int num_trees = 100;
    forest_.resize(num_trees);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int t = 0; t < num_trees; ++t) {
        std::vector<size_t> indices(features.size());
        std::uniform_int_distribution<> dis(0, static_cast<int>(features.size()) - 1);
        
        for (size_t& idx : indices) {
            idx = dis(gen);
        }
        
        forest_[t].root = std::make_unique<DecisionTree::Node>();
        forest_[t].root->value = std::accumulate(pressure_drops.begin(), 
                                                pressure_drops.end(), 0.0) / pressure_drops.size();
    }
    
    if (!features.empty()) {
        feature_importance_.resize(features[0].size());
        feature_importance_.setZero();
    }
}

void DataDrivenCorrelation::save_model(const std::string& filename) const {
    // TODO: Implement
}

void DataDrivenCorrelation::load_model(const std::string& filename) {
    // TODO: Implement
}

FlowCorrelation::Results DataDrivenCorrelation::calculate(
    const FluidProperties& fluid,
    const Pipe& pipe,
    Real flow_rate,
    Real inlet_pressure,
    Real inlet_temperature
) const {
    Results results;
    
    Vector features(12);
    
    Real area = pipe.area();
    Real velocity = flow_rate / area;
    Real reynolds = fluid.mixture_density() * velocity * pipe.diameter() / fluid.mixture_viscosity();
    
    features(0) = flow_rate;
    features(1) = pipe.diameter();
    features(2) = pipe.length();
    features(3) = pipe.roughness();
    features(4) = pipe.inclination();
    features(5) = fluid.mixture_density();
    features(6) = fluid.mixture_viscosity();
    features(7) = reynolds;
    features(8) = velocity;
    features(9) = inlet_pressure;
    features(10) = inlet_temperature;
    features(11) = fluid.gas_fraction;
    
    Real dp_predicted = 0.0;
    if (!forest_.empty()) {
        for (size_t i = 0; i < forest_.size(); ++i) {
            dp_predicted += forest_[i].predict(features);
        }
        dp_predicted /= forest_.size();
    }
    
    results.pressure_gradient = dp_predicted;
    results.liquid_holdup = 1.0 - fluid.gas_fraction;
    results.flow_pattern = FlowPattern::SLUG;
    results.friction_factor = 0.02;
    results.mixture_density = fluid.mixture_density();
    results.mixture_velocity = velocity;
    
    return results;
}

// Digital Twin Implementation
void DigitalTwin::initialize(
    const Network& network,
    const FluidProperties& fluid
) {
    network_ = std::make_shared<Network>(network);
    fluid_ = std::make_shared<FluidProperties>(fluid);  // Changed to pointer
    
    size_t state_size = network_->nodes().size() + network_->pipes().size();
    state_estimate_.resize(state_size);
    state_estimate_.setZero();
    
    covariance_matrix_ = Matrix::Identity(static_cast<int>(state_size), static_cast<int>(state_size)) * 1e4;
    process_noise_ = Matrix::Identity(static_cast<int>(state_size), static_cast<int>(state_size)) * 1e2;
    measurement_noise_ = Matrix::Identity(static_cast<int>(state_size), static_cast<int>(state_size)) * 1e1;
    
    anomaly_detector_ = std::make_unique<AnomalyDetector>();
    pattern_predictor_ = std::make_unique<FlowPatternPredictor>();
}

void DigitalTwin::update_with_measurements(
    const std::map<std::string, Real>& pressure_measurements,
    const std::map<std::string, Real>& flow_measurements,
    Real timestamp
) {
    Vector z(pressure_measurements.size() + flow_measurements.size());
    int idx = 0;
    
    for (auto it = pressure_measurements.begin(); it != pressure_measurements.end(); ++it) {
        z(idx++) = it->second;
    }
    
    for (auto it = flow_measurements.begin(); it != flow_measurements.end(); ++it) {
        z(idx++) = it->second;
    }
    
    Matrix H = Matrix::Identity(static_cast<int>(z.size()), static_cast<int>(state_estimate_.size()));
    Vector y = z - H * state_estimate_;
    
    Matrix S = H * covariance_matrix_ * H.transpose() + 
               measurement_noise_.topLeftCorner(static_cast<int>(z.size()), static_cast<int>(z.size()));
    
    Matrix K = covariance_matrix_ * H.transpose() * S.inverse();
    state_estimate_ = state_estimate_ + K * y;
    
    Matrix I = Matrix::Identity(static_cast<int>(state_estimate_.size()), static_cast<int>(state_estimate_.size()));
    covariance_matrix_ = (I - K * H) * covariance_matrix_;
    
    EstimatedState current_state;
    idx = 0;
    const auto& nodes = network_->nodes();
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        current_state.node_pressures[it->first] = state_estimate_(idx++);
    }
    const auto& pipes = network_->pipes();
    for (auto it = pipes.begin(); it != pipes.end(); ++it) {
        current_state.pipe_flows[it->first] = state_estimate_(idx++);
    }
    
    state_history_.push_back(current_state);
    time_history_.push_back(timestamp);
    
    if (state_history_.size() > 1000) {
        state_history_.pop_front();
        time_history_.pop_front();
    }
}

DigitalTwin::EstimatedState DigitalTwin::estimate_state() {
    EstimatedState state;
    
    int idx = 0;
    const auto& nodes = network_->nodes();
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        state.node_pressures[it->first] = state_estimate_(idx);
        state.uncertainties[it->first] = std::sqrt(covariance_matrix_(idx, idx));
        idx++;
    }
    
    const auto& pipes = network_->pipes();
    for (auto it = pipes.begin(); it != pipes.end(); ++it) {
        state.pipe_flows[it->first] = state_estimate_(idx);
        state.uncertainties[it->first] = std::sqrt(covariance_matrix_(idx, idx));
        idx++;
    }
    
    return state;
}

DigitalTwin::EstimatedState DigitalTwin::predict_future(Real time_horizon) {
    EstimatedState future_state = estimate_state();
    
    if (state_history_.size() >= 2) {
        Real dt = time_history_.back() - time_history_[state_history_.size() - 2];
        
        for (auto it = future_state.node_pressures.begin(); it != future_state.node_pressures.end(); ++it) {
            Real current = state_history_.back().node_pressures.at(it->first);
            Real previous = state_history_[state_history_.size() - 2].node_pressures.at(it->first);
            Real trend = (current - previous) / dt;
            
            it->second += trend * time_horizon;
        }
        
        for (auto it = future_state.pipe_flows.begin(); it != future_state.pipe_flows.end(); ++it) {
            Real current = state_history_.back().pipe_flows.at(it->first);
            Real previous = state_history_[state_history_.size() - 2].pipe_flows.at(it->first);
            Real trend = (current - previous) / dt;
            
            it->second += trend * time_horizon;
        }
    }
    
    return future_state;
}

std::vector<DigitalTwin::Discrepancy> DigitalTwin::detect_discrepancies() {
    std::vector<Discrepancy> discrepancies;
    
    if (!anomaly_detector_ || state_history_.empty()) {
        return discrepancies;
    }
    
    SolutionResults results;
    results.converged = true;
    
    const auto& current_node_pressures = state_history_.back().node_pressures;
    for (auto it = current_node_pressures.begin(); it != current_node_pressures.end(); ++it) {
        results.node_pressures[it->first] = it->second;
    }
    
    const auto& current_pipe_flows = state_history_.back().pipe_flows;
    for (auto it = current_pipe_flows.begin(); it != current_pipe_flows.end(); ++it) {
        results.pipe_flow_rates[it->first] = it->second;
    }
    
    AnomalyDetector::AnomalyResult anomaly_result = anomaly_detector_->detect(*network_, results);
    
    if (anomaly_result.is_anomaly) {
        Discrepancy disc;
        disc.location = "Network-wide";
        disc.type = "anomaly";
        disc.severity = anomaly_result.anomaly_score;
        disc.confidence = 0.8;
        discrepancies.push_back(disc);
    }
    
    const auto& nodes = network_->nodes();
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        const auto& node = it->second;
        Real inflow = 0.0;
        Real outflow = 0.0;
        
        const auto& upstream_pipes = network_->get_upstream_pipes(node);
        for (size_t i = 0; i < upstream_pipes.size(); ++i) {
            inflow += std::abs(state_history_.back().pipe_flows.at(upstream_pipes[i]->id()));
        }
        
        const auto& downstream_pipes = network_->get_downstream_pipes(node);
        for (size_t i = 0; i < downstream_pipes.size(); ++i) {
            outflow += std::abs(state_history_.back().pipe_flows.at(downstream_pipes[i]->id()));
        }
        
        Real imbalance = std::abs(inflow - outflow);
        if (imbalance > 0.01) {
            Discrepancy disc;
            disc.location = it->first;
            disc.type = "leak";
            disc.severity = imbalance;
            disc.confidence = 0.7;
            discrepancies.push_back(disc);
        }
    }
    
    return discrepancies;
}

} // namespace ml
} // namespace pipeline_sim
