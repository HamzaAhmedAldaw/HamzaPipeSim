/// AI_GENERATED: ML integration implementation
/// Generated on: 2025-06-27

// ===== src/ml_integration.cpp =====
#include "pipeline_sim/ml_integration.h"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

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

void FeatureExtractor::normalize_features(Vector& features) {
    // Simple min-max normalization
    // In practice, would use saved normalization parameters
    
    // Pressure normalization (Pa)
    Real pressure_scale = 100e5;  // 100 bar
    for (int i = 16; i < 20; ++i) {
        features(i) /= pressure_scale;
    }
    
    // Flow normalization (m³/s)
    Real flow_scale = 1.0;
    for (int i = 20; i < 24; ++i) {
        features(i) /= flow_scale;
    }
    
    // Length normalization (m)
    features(24) /= 10000.0;  // 10 km
    
    // Diameter normalization (m)
    features(25) /= 1.0;
}

// Neural Network Implementation
Vector FlowPatternPredictor::NeuralNetwork::forward(const Vector& input) {
    Vector activation = input;
    
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        // Linear transformation
        activation = weights[layer] * activation + biases[layer];
        
        // ReLU activation (except last layer)
        if (layer < weights.size() - 1) {
            activation = activation.cwiseMax(0.0);
        } else {
            // Softmax for output layer
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
    
    // Read network architecture
    int num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
    
    network_->weights.resize(num_layers);
    network_->biases.resize(num_layers);
    
    // Read weights and biases
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
    // Extract features for single pipe
    Vector features(10);
    
    // Basic features
    features(0) = pipe.length();
    features(1) = pipe.diameter();
    features(2) = pipe.inclination();
    features(3) = flow_rate;
    features(4) = fluid.oil_fraction;
    features(5) = fluid.gas_fraction;
    features(6) = fluid.water_fraction;
    features(7) = fluid.mixture_density();
    features(8) = fluid.mixture_viscosity();
    features(9) = flow_rate / pipe.area();  // Velocity
    
    // Normalize
    features(0) /= 1000.0;   // Length in km
    features(1) /= 0.5;      // Diameter in 0.5m units
    features(3) /= 0.1;      // Flow in 0.1 m³/s units
    features(7) /= 1000.0;   // Density in g/cm³
    features(8) /= 0.01;     // Viscosity in 10 cP units
    features(9) /= 10.0;     // Velocity in 10 m/s units
    
    // Predict
    Vector probabilities = predict(features);
    
    // Find most likely pattern
    int max_idx = 0;
    Real max_prob = probabilities(0);
    for (int i = 1; i < probabilities.size(); ++i) {
        if (probabilities(i) > max_prob) {
            max_prob = probabilities(i);
            max_idx = i;
        }
    }
    
    return static_cast<FlowPattern>(max_idx);
}

// Anomaly Detector Implementation
Real AnomalyDetector::IsolationTree::path_length(const Vector& sample) const {
    Node* current = root.get();
    Real length = 0.0;
    
    while (current && !current->left && !current->right) {
        if (sample(current->feature_index) < current->split_value) {
            current = current->left.get();
        } else {
            current = current->right.get();
        }
        length += 1.0;
    }
    
    // Add average path length for remaining depth
    if (current) {
        int remaining_depth = 10 - current->depth;  // Assume max depth 10
        length += 2.0 * (std::log(remaining_depth) + 0.5772) - 2.0 * (remaining_depth - 1) / remaining_depth;
    }
    
    return length;
}

void AnomalyDetector::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open model file: " + filename);
    }
    
    // Read number of trees
    int num_trees;
    file.read(reinterpret_cast<char*>(&num_trees), sizeof(int));
    
    forest_.resize(num_trees);
    
    // Load each tree
    for (int i = 0; i < num_trees; ++i) {
        // Simplified: create random tree
        forest_[i].root = std::make_unique<IsolationTree::Node>();
        // In practice, would deserialize tree structure
    }
}

Vector AnomalyDetector::predict(const Vector& features) {
    Vector scores(1);
    
    // Calculate average path length
    Real avg_path_length = 0.0;
    for (const auto& tree : forest_) {
        avg_path_length += tree.path_length(features);
    }
    avg_path_length /= forest_.size();
    
    // Calculate anomaly score
    Real c_n = 2.0 * (std::log(features.size()) + 0.5772) - 2.0 * (features.size() - 1) / features.size();
    scores(0) = std::pow(2.0, -avg_path_length / c_n);
    
    return scores;
}

AnomalyDetector::AnomalyResult AnomalyDetector::detect(
    const Network& network,
    const SolutionResults& results
) {
    AnomalyResult result;
    
    // Extract features
    FluidProperties fluid;  // TODO: Get from network
    Vector features = FeatureExtractor::extract_features(network, results, fluid);
    FeatureExtractor::normalize_features(features);
    
    // Get anomaly score
    Vector scores = predict(features);
    result.anomaly_score = scores(0);
    
    // Threshold for anomaly detection
    Real threshold = 0.6;
    result.is_anomaly = result.anomaly_score > threshold;
    
    // Identify anomalous features
    if (result.is_anomaly) {
        auto feature_names = FeatureExtractor::get_feature_names();
        
        // Find features that deviate most from normal
        for (size_t i = 0; i < features.size(); ++i) {
            if (std::abs(features(i)) > 2.0) {  // More than 2 std devs
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
    
    // Initialize genetic algorithm
    initialize_population(50);  // Population size
    
    // Evolution loop
    const int max_generations = 100;
    for (int gen = 0; gen < max_generations; ++gen) {
        // Evaluate fitness
        evaluate_fitness(network, fluid, objective);
        
        // Check for convergence
        Real best_fitness = population_[0].fitness;
        if (best_fitness < 1e-6) {
            result.success = true;
            break;
        }
        
        // Genetic operations
        selection();
        crossover();
        mutation();
    }
    
    // Extract best solution
    if (!population_.empty()) {
        const auto& best = population_[0];
        result.objective_value = best.fitness;
        
        // Decode control variables
        size_t idx = 0;
        for (const auto& [id, node] : network.nodes()) {
            if (node->type() == NodeType::PUMP && idx < best.genes.size()) {
                result.pump_speeds[id] = best.genes(idx++);
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
    
    // Determine number of control variables
    int num_vars = 10;  // Simplified
    
    for (auto& individual : population_) {
        individual.genes.resize(num_vars);
        for (int i = 0; i < num_vars; ++i) {
            individual.genes(i) = dis(gen);
        }
        individual.fitness = std::numeric_limits<Real>::max();
    }
}

void MLOptimizer::evaluate_fitness(
    Network& network,
    const FluidProperties& fluid,
    const OptimizationObjective& objective
) {
    for (auto& individual : population_) {
        // Apply control variables to network
        // (Simplified - would actually modify pump speeds, valve openings, etc.)
        
        // Run simulation
        SteadyStateSolver solver(network, fluid);
        auto results = solver.solve();
        
        // Calculate fitness based on objective
        if (results.converged) {
            switch (objective.type) {
                case OptimizationObjective::MINIMIZE_PRESSURE_DROP: {
                    Real total_dp = 0.0;
                    for (const auto& [id, dp] : results.pipe_pressure_drops) {
                        total_dp += dp;
                    }
                    individual.fitness = total_dp;
                    break;
                }
                
                case OptimizationObjective::MAXIMIZE_FLOW_RATE: {
                    Real total_flow = 0.0;
                    for (const auto& [id, flow] : results.pipe_flow_rates) {
                        total_flow += std::abs(flow);
                    }
                    individual.fitness = -total_flow;  // Negative for maximization
                    break;
                }
                
                case OptimizationObjective::MINIMIZE_ENERGY_CONSUMPTION: {
                    Real total_power = 0.0;
                    // TODO: Sum pump power consumption
                    individual.fitness = total_power;
                    break;
                }
                
                case OptimizationObjective::CUSTOM:
                    individual.fitness = objective.custom_function(results);
                    break;
            }
        } else {
            individual.fitness = std::numeric_limits<Real>::max();
        }
    }
    
    // Sort by fitness
    std::sort(population_.begin(), population_.end(),
              [](const Individual& a, const Individual& b) {
                  return a.fitness < b.fitness;
              });
}

void MLOptimizer::selection() {
    // Tournament selection
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, population_.size() - 1);
    
    std::vector<Individual> new_population;
    new_population.reserve(population_.size());
    
    // Keep best individual (elitism)
    new_population.push_back(population_[0]);
    
    // Tournament selection for rest
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
            // Single-point crossover
            int point = std::uniform_int_distribution<>(0, population_[i].genes.size() - 1)(gen);
            
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
    
    for (size_t i = 1; i < population_.size(); ++i) {  // Skip best individual
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
    // Random Forest training (simplified)
    const int num_trees = 100;
    forest_.resize(num_trees);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int t = 0; t < num_trees; ++t) {
        // Bootstrap sampling
        std::vector<size_t> indices(features.size());
        std::uniform_int_distribution<> dis(0, features.size() - 1);
        
        for (size_t& idx : indices) {
            idx = dis(gen);
        }
        
        // Train tree (simplified - would use recursive partitioning)
        forest_[t].root = std::make_unique<DecisionTree::Node>();
        forest_[t].root->value = std::accumulate(pressure_drops.begin(), 
                                                pressure_drops.end(), 0.0) / pressure_drops.size();
    }
    
    // Calculate feature importance
    feature_importance_.resize(features[0].size());
    feature_importance_.setZero();
}

FlowCorrelation::Results DataDrivenCorrelation::calculate(
    const FluidProperties& fluid,
    const Pipe& pipe,
    Real flow_rate,
    Real inlet_pressure,
    Real inlet_temperature
) const {
    Results results;
    
    // Extract features
    Vector features(12);
    features(0) = pipe.length();
    features(1) = pipe.diameter();
    features(2) = pipe.roughness();
    features(3) = pipe.inclination();
    features(4) = flow_rate;
    features(5) = inlet_pressure;
    features(6) = inlet_temperature;
    features(7) = fluid.oil_fraction;
    features(8) = fluid.gas_fraction;
    features(9) = fluid.water_fraction;
    features(10) = fluid.mixture_density();
    features(11) = fluid.mixture_viscosity();
    
    // Normalize features (would use saved normalization parameters)
    features(0) /= 1000.0;    // km
    features(1) /= 0.5;       // 0.5m units
    features(4) /= 0.1;       // 0.1 m³/s units
    features(5) /= 50e5;      // 50 bar units
    features(6) /= 300.0;     // 300K units
    features(10) /= 1000.0;   // g/cm³
    features(11) /= 0.01;     // 10 cP units
    
    // Predict pressure gradient
    Real pressure_gradient = 0.0;
    for (const auto& tree : forest_) {
        pressure_gradient += tree.predict(features);
    }
    pressure_gradient /= forest_.size();
    
    results.pressure_gradient = pressure_gradient;
    
    // Estimate other properties (simplified)
    results.liquid_holdup = 1.0 - fluid.gas_fraction * 0.8;
    results.flow_pattern = FlowPattern::INTERMITTENT;
    results.mixture_density = fluid.mixture_density();
    results.mixture_velocity = flow_rate / pipe.area();
    
    return results;
}

// Digital Twin Implementation
void DigitalTwin::initialize(
    const Network& network,
    const FluidProperties& fluid
) {
    network_ = std::make_shared<Network>(network);
    fluid_ = fluid;
    
    // Initialize state vector
    size_t state_size = network_->nodes().size() + network_->pipes().size();
    state_estimate_.resize(state_size);
    state_estimate_.setZero();
    
    // Initialize covariance matrix
    covariance_matrix_ = Matrix::Identity(state_size, state_size) * 1e4;
    
    // Process and measurement noise
    process_noise_ = Matrix::Identity(state_size, state_size) * 1e2;
    measurement_noise_ = Matrix::Identity(state_size, state_size) * 1e1;
    
    // Load ML models
    anomaly_detector_ = std::make_unique<AnomalyDetector>();
    pattern_predictor_ = std::make_unique<FlowPatternPredictor>();
}

void DigitalTwin::update_with_measurements(
    const std::map<std::string, Real>& pressure_measurements,
    const std::map<std::string, Real>& flow_measurements,
    Real timestamp
) {
    // Kalman filter update
    size_t num_nodes = network_->nodes().size();
    
    // Build measurement vector
    Vector z(pressure_measurements.size() + flow_measurements.size());
    int idx = 0;
    
    for (const auto& [node_id, pressure] : pressure_measurements) {
        z(idx++) = pressure;
    }
    
    for (const auto& [pipe_id, flow] : flow_measurements) {
        z(idx++) = flow;
    }
    
    // Measurement matrix H (simplified - identity for measured states)
    Matrix H = Matrix::Identity(z.size(), state_estimate_.size());
    
    // Kalman filter equations
    // Innovation
    Vector y = z - H * state_estimate_;
    
    // Innovation covariance
    Matrix S = H * covariance_matrix_ * H.transpose() + measurement_noise_.topLeftCorner(z.size(), z.size());
    
    // Kalman gain
    Matrix K = covariance_matrix_ * H.transpose() * S.inverse();
    
    // Update state estimate
    state_estimate_ = state_estimate_ + K * y;
    
    // Update covariance
    Matrix I = Matrix::Identity(state_estimate_.size(), state_estimate_.size());
    covariance_matrix_ = (I - K * H) * covariance_matrix_;
    
    // Save to history
    EstimatedState current_state;
    idx = 0;
    for (const auto& [node_id, node] : network_->nodes()) {
        current_state.node_pressures[node_id] = state_estimate_(idx++);
    }
    for (const auto& [pipe_id, pipe] : network_->pipes()) {
        current_state.pipe_flows[pipe_id] = state_estimate_(idx++);
    }
    
    state_history_.push_back(current_state);
    time_history_.push_back(timestamp);
    
    // Keep history size manageable
    if (state_history_.size() > 1000) {
        state_history_.pop_front();
        time_history_.pop_front();
    }
}

DigitalTwin::EstimatedState DigitalTwin::estimate_state() {
    EstimatedState state;
    
    size_t idx = 0;
    for (const auto& [node_id, node] : network_->nodes()) {
        state.node_pressures[node_id] = state_estimate_(idx);
        state.uncertainties[node_id] = std::sqrt(covariance_matrix_(idx, idx));
        idx++;
    }
    
    for (const auto& [pipe_id, pipe] : network_->pipes()) {
        state.pipe_flows[pipe_id] = state_estimate_(idx);
        state.uncertainties[pipe_id] = std::sqrt(covariance_matrix_(idx, idx));
        idx++;
    }
    
    return state;
}

DigitalTwin::EstimatedState DigitalTwin::predict_future(Real time_horizon) {
    // Simple prediction using current state and trends
    EstimatedState future_state = estimate_state();
    
    // Calculate trends from history
    if (state_history_.size() >= 2) {
        Real dt = time_history_.back() - time_history_[state_history_.size() - 2];
        
        for (auto& [node_id, pressure] : future_state.node_pressures) {
            Real current = state_history_.back().node_pressures.at(node_id);
            Real previous = state_history_[state_history_.size() - 2].node_pressures.at(node_id);
            Real trend = (current - previous) / dt;
            
            pressure += trend * time_horizon;
        }
        
        for (auto& [pipe_id, flow] : future_state.pipe_flows) {
            Real current = state_history_.back().pipe_flows.at(pipe_id);
            Real previous = state_history_[state_history_.size() - 2].pipe_flows.at(pipe_id);
            Real trend = (current - previous) / dt;
            
            flow += trend * time_horizon;
        }
    }
    
    return future_state;
}

std::vector<DigitalTwin::Discrepancy> DigitalTwin::detect_discrepancies() {
    std::vector<Discrepancy> discrepancies;
    
    if (!anomaly_detector_ || state_history_.empty()) {
        return discrepancies;
    }
    
    // Convert current state to solution results
    SolutionResults results;
    results.converged = true;
    
    for (const auto& [node_id, pressure] : state_history_.back().node_pressures) {
        results.node_pressures[node_id] = pressure;
    }
    
    for (const auto& [pipe_id, flow] : state_history_.back().pipe_flows) {
        results.pipe_flow_rates[pipe_id] = flow;
    }
    
    // Run anomaly detection
    auto anomaly_result = anomaly_detector_->detect(*network_, results);
    
    if (anomaly_result.is_anomaly) {
        Discrepancy disc;
        disc.location = "Network-wide";
        disc.type = "anomaly";
        disc.severity = anomaly_result.anomaly_score;
        disc.confidence = 0.8;
        discrepancies.push_back(disc);
    }
    
    // Check for specific issues
    // Leak detection - mass balance check
    for (const auto& [node_id, node] : network_->nodes()) {
        Real inflow = 0.0;
        Real outflow = 0.0;
        
        for (const auto& pipe : network_->get_upstream_pipes(node)) {
            inflow += std::abs(state_history_.back().pipe_flows.at(pipe->id()));
        }
        
        for (const auto& pipe : network_->get_downstream_pipes(node)) {
            outflow += std::abs(state_history_.back().pipe_flows.at(pipe->id()));
        }
        
        Real imbalance = std::abs(inflow - outflow);
        if (imbalance > 0.01) {  // 0.01 m³/s threshold
            Discrepancy disc;
            disc.location = node_id;
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