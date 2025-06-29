
#include "pipeline_sim/ml_integration.h"
#include "pipeline_sim/solver.h"
#include "pipeline_sim/fluid_properties.h"
#include <cmath>
#include <Eigen/Dense>
#include <stdexcept>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <random>
#include <sstream>
#include <deque>
#include <map>
#include <functional>

// Define M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pipeline_sim {
namespace ml {

// ============================================================================
// FeatureExtractor Implementation
// ============================================================================

Vector FeatureExtractor::extract_features(
    const Network& network,
    const SolutionResults& results,
    const FluidProperties& fluid) {
    
    // Simple feature extraction
    Vector features(10);
    features.setZero();
    
    // Basic network features
    features(0) = static_cast<Real>(network.node_count());
    features(1) = static_cast<Real>(network.pipe_count());
    
    // Fluid properties - use direct member access
    features(2) = fluid.mixture_density();
    features(3) = fluid.mixture_viscosity();
    features(4) = fluid.temperature;
    
    // Results features
    features(5) = results.converged ? 1.0 : 0.0;
    features(6) = static_cast<Real>(results.iterations);
    features(7) = results.residual;
    
    // Calculate average flow and max pressure drop from results
    Real avg_flow = 0.0;
    Real max_pressure_drop = 0.0;
    
    if (!results.pipe_flow_rates.empty()) {
        for (const auto& [pipe_id, flow] : results.pipe_flow_rates) {
            avg_flow += std::abs(flow);
        }
        avg_flow /= results.pipe_flow_rates.size();
    }
    
    if (!results.pipe_pressure_drops.empty()) {
        for (const auto& [pipe_id, dp] : results.pipe_pressure_drops) {
            max_pressure_drop = std::max(max_pressure_drop, std::abs(dp));
        }
    }
    
    features(8) = avg_flow;
    features(9) = max_pressure_drop;
    
    return features;
}

std::vector<std::string> FeatureExtractor::get_feature_names() {
    return {
        "num_nodes", "num_pipes", "density", "viscosity", "temperature",
        "converged", "iterations", "residual", "avg_flow", "max_pressure_drop"
    };
}

void FeatureExtractor::normalize_features(Vector& features) {
    // Simple normalization with reasonable ranges
    for (int i = 0; i < features.size(); ++i) {
        Real min_val = 0.0;
        Real max_val = 1.0;
        
        switch (i) {
            case 0: // num_nodes
            case 1: // num_pipes
                max_val = 1000.0;
                break;
            case 2: // density
                min_val = 500.0;
                max_val = 1500.0;
                break;
            case 3: // viscosity
                min_val = 0.0001;
                max_val = 1.0;
                break;
            case 4: // temperature
                min_val = 273.0;
                max_val = 373.0;
                break;
            case 5: // converged (already 0-1)
                break;
            case 6: // iterations
                max_val = 100.0;
                break;
            case 7: // residual
                max_val = 1.0;
                break;
            case 8: // avg_flow
                max_val = 10.0;
                break;
            case 9: // max_pressure_drop
                max_val = 1e6;
                break;
        }
        
        if (max_val > min_val) {
            features(i) = (features(i) - min_val) / (max_val - min_val);
            features(i) = std::max(0.0, std::min(1.0, features(i)));
        }
    }
}

// ============================================================================
// FlowPatternPredictor Implementation
// ============================================================================

void FlowPatternPredictor::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        // If file doesn't exist, create default network
        network_ = std::make_unique<NeuralNetwork>();
        
        // Default 3-layer network: input(5) -> hidden(10) -> output(4)
        Matrix W1 = Matrix::Random(10, 5) * 0.5;
        Vector b1 = Vector::Zero(10);
        network_->weights.push_back(W1);
        network_->biases.push_back(b1);
        
        Matrix W2 = Matrix::Random(4, 10) * 0.5;
        Vector b2 = Vector::Zero(4);
        network_->weights.push_back(W2);
        network_->biases.push_back(b2);
        return;
    }
    
    network_ = std::make_unique<NeuralNetwork>();
    
    // Simple binary format: number of layers, then for each layer: rows, cols, weights, bias
    int num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
    
    for (int i = 0; i < num_layers; ++i) {
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        
        Matrix W(rows, cols);
        file.read(reinterpret_cast<char*>(W.data()), sizeof(Real) * rows * cols);
        network_->weights.push_back(W);
        
        Vector b(rows);
        file.read(reinterpret_cast<char*>(b.data()), sizeof(Real) * rows);
        network_->biases.push_back(b);
    }
    
    file.close();
}

Vector FlowPatternPredictor::predict(const Vector& features) {
    if (!network_) {
        // Return default prediction if no model loaded
        Vector result(4); // 4 flow patterns
        result << 0.7, 0.2, 0.05, 0.05;
        return result;
    }
    
    return network_->forward(features);
}

FlowPattern FlowPatternPredictor::predict_pattern(
    const Pipe& pipe,
    const FluidProperties& fluid,
    Real flow_rate) {
    
    // Extract features for this pipe
    Vector features(5);
    features(0) = flow_rate;
    features(1) = pipe.diameter();
    features(2) = fluid.mixture_density();
    features(3) = fluid.mixture_viscosity();
    features(4) = fluid.gas_fraction;
    
    // Normalize
    FeatureExtractor::normalize_features(features);
    
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
    
    // Map index to flow pattern (use patterns from correlations.h)
    switch (max_idx) {
        case 0: return FlowPattern::SEGREGATED;
        case 1: return FlowPattern::BUBBLE;
        case 2: return FlowPattern::SLUG;
        case 3: return FlowPattern::ANNULAR;
        default: return FlowPattern::SEGREGATED;
    }
}

Vector FlowPatternPredictor::NeuralNetwork::forward(const Vector& input) {
    Vector activation = input;
    
    for (size_t i = 0; i < weights.size(); ++i) {
        activation = weights[i] * activation + biases[i];
        
        // ReLU activation for all but last layer
        if (i < weights.size() - 1) {
            activation = activation.cwiseMax(0.0);
        }
    }
    
    // Softmax for output layer
    Real max_val = activation.maxCoeff();
    activation = (activation.array() - max_val).exp();
    activation /= activation.sum();
    
    return activation;
}

// ============================================================================
// AnomalyDetector Implementation
// ============================================================================

void AnomalyDetector::load(const std::string& filename) {
    // Create dummy forest for now
    forest_.clear();
    int num_trees = 10;
    
    std::default_random_engine generator(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<Real> distribution(0.0, 1.0);
    
    for (int i = 0; i < num_trees; ++i) {
        IsolationTree tree;
        tree.root = std::make_unique<IsolationTree::Node>();
        tree.root->feature_index = i % 5;
        tree.root->split_value = distribution(generator);
        tree.root->depth = 0;
        forest_.push_back(std::move(tree));
    }
}

Vector AnomalyDetector::predict(const Vector& features) {
    Vector scores(1);
    
    if (forest_.empty()) {
        scores(0) = 0.0; // No anomaly if no model
        return scores;
    }
    
    Real avg_path_length = 0.0;
    for (const auto& tree : forest_) {
        avg_path_length += tree.path_length(features);
    }
    avg_path_length /= forest_.size();
    
    // Convert path length to anomaly score
    Real c_n = 2.0 * std::log(static_cast<Real>(features.size() - 1)) + 0.5772; // Euler's constant
    scores(0) = std::pow(2.0, -avg_path_length / c_n);
    
    return scores;
}

AnomalyDetector::AnomalyResult AnomalyDetector::detect(
    const Network& network,
    const SolutionResults& results) {
    
    AnomalyResult result;
    result.is_anomaly = false;
    result.anomaly_score = 0.0;
    
    // Extract features
    FluidProperties dummy_fluid; // Would need actual fluid properties
    Vector features = FeatureExtractor::extract_features(network, results, dummy_fluid);
    FeatureExtractor::normalize_features(features);
    
    // Get anomaly score
    Vector scores = predict(features);
    result.anomaly_score = scores(0);
    
    // Threshold for anomaly
    if (result.anomaly_score > 0.7) {
        result.is_anomaly = true;
        
        // Identify which features contribute most to anomaly
        auto feature_names = FeatureExtractor::get_feature_names();
        for (int i = 0; i < features.size(); ++i) {
            if (std::abs(features(i) - 0.5) > 0.3) { // Simple heuristic
                result.anomaly_features.push_back(feature_names[i]);
            }
        }
    }
    
    return result;
}

Real AnomalyDetector::IsolationTree::path_length(const Vector& sample) const {
    if (!root) return 0.0;
    
    Real path = 0.0;
    Node* current = root.get();
    int max_depth = 10; // Prevent infinite loops
    
    while (current && path < max_depth) {
        path += 1.0;
        
        if (!current->left || !current->right) {
            // Leaf node
            break;
        }
        
        if (current->feature_index >= 0 && current->feature_index < sample.size()) {
            if (sample(current->feature_index) < current->split_value) {
                current = current->left.get();
            } else {
                current = current->right.get();
            }
        } else {
            break;
        }
    }
    
    return path + current->depth;
}

// ============================================================================
// MLOptimizer Implementation
// ============================================================================

MLOptimizer::OptimizationResult MLOptimizer::optimize(
    Network& network,
    const FluidProperties& fluid,
    const OptimizationObjective& objective,
    const OptimizationConstraints& constraints) {
    
    OptimizationResult result;
    result.success = false;
    
    // Initialize genetic algorithm
    initialize_population(50);
    
    // Run optimization for a few generations
    for (int generation = 0; generation < 20; ++generation) {
        evaluate_fitness(network, fluid, objective);
        selection();
        crossover();
        mutation();
    }
    
    // Find best individual
    if (!population_.empty()) {
        auto best_it = std::max_element(population_.begin(), population_.end(),
            [](const Individual& a, const Individual& b) {
                return a.fitness < b.fitness;
            });
        
        if (best_it != population_.end()) {
            result.success = true;
            result.objective_value = best_it->fitness;
            
            // Extract control variables (simplified)
            result.pump_speeds["pump1"] = std::max(0.0, std::min(1.0, best_it->genes(0)));
            result.valve_openings["valve1"] = std::max(0.0, std::min(1.0, best_it->genes(1)));
            result.compressor_ratios["comp1"] = std::max(1.0, std::min(2.0, 1.0 + best_it->genes(2)));
        }
    }
    
    return result;
}

void MLOptimizer::initialize_population(size_t size) {
    population_.clear();
    population_.reserve(size);
    
    std::default_random_engine generator(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<Real> distribution(0.0, 1.0);
    
    for (size_t i = 0; i < size; ++i) {
        Individual ind;
        ind.genes = Vector(10); // Assuming 10 control variables
        for (int j = 0; j < ind.genes.size(); ++j) {
            ind.genes(j) = distribution(generator);
        }
        ind.fitness = 0.0;
        population_.push_back(ind);
    }
}

void MLOptimizer::evaluate_fitness(
    Network& network,
    const FluidProperties& fluid,
    const OptimizationObjective& objective) {
    
    // Simple fitness evaluation
    for (auto& individual : population_) {
        // Simulate with current genes (simplified)
        individual.fitness = -individual.genes.squaredNorm(); // Dummy fitness
        
        // In real implementation, would:
        // 1. Apply control variables to network
        // 2. Run simulation
        // 3. Calculate fitness based on objective
    }
}

void MLOptimizer::selection() {
    // Tournament selection
    std::vector<Individual> new_population;
    new_population.reserve(population_.size());
    
    std::default_random_engine generator(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_int_distribution<int> distribution(0, static_cast<int>(population_.size() - 1));
    
    while (new_population.size() < population_.size()) {
        int idx1 = distribution(generator);
        int idx2 = distribution(generator);
        
        if (population_[idx1].fitness > population_[idx2].fitness) {
            new_population.push_back(population_[idx1]);
        } else {
            new_population.push_back(population_[idx2]);
        }
    }
    
    population_ = std::move(new_population);
}

void MLOptimizer::crossover() {
    std::default_random_engine generator(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<Real> distribution(0.0, 1.0);
    
    for (size_t i = 0; i < population_.size() - 1; i += 2) {
        if (distribution(generator) < 0.8) { // Crossover probability
            // Single-point crossover
            int crossover_point = static_cast<int>(distribution(generator) * population_[i].genes.size());
            
            for (int j = crossover_point; j < population_[i].genes.size(); ++j) {
                std::swap(population_[i].genes(j), population_[i + 1].genes(j));
            }
        }
    }
}

void MLOptimizer::mutation() {
    std::default_random_engine generator(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<Real> distribution(0.0, 1.0);
    std::normal_distribution<Real> normal(0.0, 0.1);
    
    for (auto& individual : population_) {
        for (int i = 0; i < individual.genes.size(); ++i) {
            if (distribution(generator) < 0.1) { // Mutation probability
                individual.genes(i) += normal(generator);
                individual.genes(i) = std::max(0.0, std::min(1.0, individual.genes(i)));
            }
        }
    }
}

// ============================================================================
// DataDrivenCorrelation Implementation
// ============================================================================

void DataDrivenCorrelation::train(
    const std::vector<Vector>& features,
    const std::vector<Real>& pressure_drops) {
    
    if (features.size() != pressure_drops.size() || features.empty()) {
        throw std::runtime_error("Invalid training data");
    }
    
    // Build a simple random forest
    forest_.clear();
    int num_trees = 10;
    
    std::default_random_engine generator(static_cast<unsigned int>(std::time(nullptr)));
    
    for (int i = 0; i < num_trees; ++i) {
        DecisionTree tree;
        tree.root = std::make_unique<DecisionTree::Node>();
        
        // Simple implementation: just store average
        Real avg = std::accumulate(pressure_drops.begin(), pressure_drops.end(), 0.0) / pressure_drops.size();
        tree.root->value = avg;
        
        forest_.push_back(std::move(tree));
    }
    
    // Calculate feature importance (simplified)
    int feature_dim = static_cast<int>(features[0].size());
    feature_importance_ = Vector::Ones(feature_dim) / feature_dim;
}

void DataDrivenCorrelation::save_model(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create model file: " + filename);
    }
    
    // Save forest size
    int num_trees = static_cast<int>(forest_.size());
    file.write(reinterpret_cast<const char*>(&num_trees), sizeof(int));
    
    // Save feature importance
    int feature_dim = static_cast<int>(feature_importance_.size());
    file.write(reinterpret_cast<const char*>(&feature_dim), sizeof(int));
    file.write(reinterpret_cast<const char*>(feature_importance_.data()), sizeof(Real) * feature_dim);
    
    file.close();
}

void DataDrivenCorrelation::load_model(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        // Create default model
        forest_.clear();
        DecisionTree tree;
        tree.root = std::make_unique<DecisionTree::Node>();
        tree.root->value = 1000.0; // Default pressure drop
        forest_.push_back(std::move(tree));
        
        feature_importance_ = Vector::Ones(5) / 5.0;
        return;
    }
    
    int num_trees;
    file.read(reinterpret_cast<char*>(&num_trees), sizeof(int));
    
    forest_.clear();
    for (int i = 0; i < num_trees; ++i) {
        DecisionTree tree;
        tree.root = std::make_unique<DecisionTree::Node>();
        tree.root->value = 1000.0; // Default value
        forest_.push_back(std::move(tree));
    }
    
    int feature_dim;
    file.read(reinterpret_cast<char*>(&feature_dim), sizeof(int));
    feature_importance_ = Vector(feature_dim);
    file.read(reinterpret_cast<char*>(feature_importance_.data()), sizeof(Real) * feature_dim);
    
    file.close();
}

FlowCorrelation::Results DataDrivenCorrelation::calculate(
    const FluidProperties& fluid,
    const Pipe& pipe,
    Real flow_rate,
    Real inlet_pressure,
    Real inlet_temperature) const {
    
    Results results;
    
    // Extract features
    Vector features(5);
    features(0) = flow_rate;
    features(1) = pipe.diameter();
    features(2) = pipe.length();
    features(3) = fluid.mixture_density();
    features(4) = fluid.mixture_viscosity();
    
    // Predict pressure drop using forest
    Real pressure_drop = 0.0;
    if (!forest_.empty()) {
        for (const auto& tree : forest_) {
            pressure_drop += tree.predict(features);
        }
        pressure_drop /= forest_.size();
    } else {
        // Fallback to simple calculation
        Real area = M_PI * pipe.diameter() * pipe.diameter() / 4.0;
        Real velocity = flow_rate / area;
        Real reynolds = fluid.mixture_density() * velocity * pipe.diameter() / fluid.mixture_viscosity();
        
        // Friction factor
        Real friction = 0.0;
        if (reynolds < 2300) {
            friction = 64.0 / reynolds; // Laminar
        } else {
            friction = 0.3164 / std::pow(reynolds, 0.25); // Blasius for turbulent
        }
        
        pressure_drop = friction * pipe.length() / pipe.diameter() * 
                       0.5 * fluid.mixture_density() * velocity * velocity;
    }
    
    // Fill results according to FlowCorrelation::Results structure
    results.pressure_gradient = pressure_drop / pipe.length();
    results.liquid_holdup = 1.0 - fluid.gas_fraction;
    results.flow_pattern = FlowPattern::SEGREGATED; // Default
    results.friction_factor = 0.02; // Simplified
    results.mixture_density = fluid.mixture_density();
    results.mixture_velocity = flow_rate / (M_PI * pipe.diameter() * pipe.diameter() / 4.0);
    
    return results;
}

Real DataDrivenCorrelation::DecisionTree::predict(const Vector& features) const {
    if (!root) return 0.0;
    
    Node* current = root.get();
    while (current && !current->is_leaf()) {
        if (current->feature_index >= 0 && current->feature_index < features.size()) {
            if (features(current->feature_index) < current->threshold) {
                if (current->left) current = current->left.get();
                else break;
            } else {
                if (current->right) current = current->right.get();
                else break;
            }
        } else {
            break;
        }
    }
    
    return current ? current->value : 0.0;
}

// ============================================================================
// DigitalTwin Implementation
// ============================================================================

void DigitalTwin::initialize(
    const Network& network,
    const FluidProperties& fluid) {
    
    network_ = std::make_shared<Network>();
    fluid_ = fluid;
    
    // Initialize state estimation
    int num_nodes = static_cast<int>(network.node_count());
    int num_pipes = static_cast<int>(network.pipe_count());
    int state_size = num_nodes + num_pipes; // Pressures + flows
    
    state_estimate_ = Vector::Zero(state_size);
    covariance_matrix_ = Matrix::Identity(state_size, state_size) * 1e3;
    process_noise_ = Matrix::Identity(state_size, state_size) * 1e-2;
    measurement_noise_ = Matrix::Identity(state_size, state_size) * 1e-1;
    
    // Initialize ML models
    anomaly_detector_ = std::make_unique<AnomalyDetector>();
    pattern_predictor_ = std::make_unique<FlowPatternPredictor>();
    
    // Load models if available
    anomaly_detector_->load("anomaly_model.bin");
    pattern_predictor_->load("pattern_model.bin");
}

void DigitalTwin::update_with_measurements(
    const std::map<std::string, Real>& pressure_measurements,
    const std::map<std::string, Real>& flow_measurements,
    Real timestamp) {
    
    if (!network_) return;
    
    // Simple state update
    EstimatedState state;
    
    // Update with measurements
    for (const auto& [id, pressure] : pressure_measurements) {
        state.node_pressures[id] = pressure;
        state.uncertainties["pressure_" + id] = 100.0; // Pa
    }
    
    for (const auto& [id, flow] : flow_measurements) {
        state.pipe_flows[id] = flow;
        state.uncertainties["flow_" + id] = 0.01; // m3/s
    }
    
    // Store in history
    state_history_.push_back(state);
    time_history_.push_back(timestamp);
    
    // Keep limited history
    while (state_history_.size() > 100) {
        state_history_.pop_front();
        time_history_.pop_front();
    }
}

DigitalTwin::EstimatedState DigitalTwin::estimate_state() {
    EstimatedState state;
    
    if (!state_history_.empty()) {
        state = state_history_.back();
    } else {
        // Return empty state with default uncertainties
        state.uncertainties["default"] = 1.0;
    }
    
    return state;
}

DigitalTwin::EstimatedState DigitalTwin::predict_future(Real time_horizon) {
    EstimatedState future_state = estimate_state();
    
    if (state_history_.size() < 2) return future_state;
    
    // Simple linear extrapolation
    const auto& current = state_history_.back();
    const auto& previous = state_history_[state_history_.size() - 2];
    Real dt = time_history_.back() - time_history_[time_history_.size() - 2];
    
    if (dt > 0) {
        // Extrapolate pressures
        for (auto& [id, pressure] : future_state.node_pressures) {
            auto it_curr = current.node_pressures.find(id);
            auto it_prev = previous.node_pressures.find(id);
            if (it_curr != current.node_pressures.end() && it_prev != previous.node_pressures.end()) {
                Real dp = (it_curr->second - it_prev->second) / dt;
                pressure += dp * time_horizon;
            }
        }
        
        // Extrapolate flows
        for (auto& [id, flow] : future_state.pipe_flows) {
            auto it_curr = current.pipe_flows.find(id);
            auto it_prev = previous.pipe_flows.find(id);
            if (it_curr != current.pipe_flows.end() && it_prev != previous.pipe_flows.end()) {
                Real dq = (it_curr->second - it_prev->second) / dt;
                flow += dq * time_horizon;
            }
        }
        
        // Increase uncertainties for future predictions
        for (auto& [key, uncertainty] : future_state.uncertainties) {
            uncertainty *= (1.0 + 0.1 * time_horizon);
        }
    }
    
    return future_state;
}

std::vector<DigitalTwin::Discrepancy> DigitalTwin::detect_discrepancies() {
    std::vector<Discrepancy> discrepancies;
    
    if (!network_ || state_history_.size() < 5) return discrepancies;
    
    // Check for potential issues
    const auto& current_state = state_history_.back();
    const auto& old_state = state_history_[state_history_.size() - 5];
    
    // Check for potential leaks (sudden pressure drops)
    for (const auto& [id, pressure] : current_state.node_pressures) {
        auto it = old_state.node_pressures.find(id);
        if (it != old_state.node_pressures.end()) {
            Real old_pressure = it->second;
            if (old_pressure > 0) {
                Real dp = (old_pressure - pressure) / old_pressure;
                
                if (dp > 0.1) { // 10% drop
                    Discrepancy disc;
                    disc.location = "node_" + id;
                    disc.type = "leak";
                    disc.severity = dp;
                    disc.confidence = std::min(0.9, dp * 5.0); // Higher drop = higher confidence
                    discrepancies.push_back(disc);
                }
            }
        }
    }
    
    // Check for blockages (flow reductions)
    for (const auto& [id, flow] : current_state.pipe_flows) {
        auto it = old_state.pipe_flows.find(id);
        if (it != old_state.pipe_flows.end()) {
            Real old_flow = it->second;
            if (old_flow > 0) {
                Real dq = (old_flow - flow) / old_flow;
                
                if (dq > 0.2) { // 20% flow reduction
                    Discrepancy disc;
                    disc.location = "pipe_" + id;
                    disc.type = "blockage";
                    disc.severity = dq;
                    disc.confidence = std::min(0.8, dq * 3.0);
                    discrepancies.push_back(disc);
                }
            }
        }
    }
    
    return discrepancies;
}

} // namespace ml
} // namespace pipeline_sim

