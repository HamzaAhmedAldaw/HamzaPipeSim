// ===== COMPLETE FINAL SOLUTION =====

// FILE: core/include/pipeline_sim/ml_integration.h
#pragma once

#include "pipeline_sim/types.h"
#include "pipeline_sim/correlations.h"  // For FlowCorrelation base class

#include <vector>
#include <memory>
#include <functional>
#include <deque>
#include <map>
#include <string>

namespace pipeline_sim {

// Forward declarations
class Network;
class Node;
class Pipe;
struct FluidProperties;
enum class NodeType;

// Define SolutionResults here to avoid circular dependency
struct SolutionResults {
    bool converged{false};
    int iterations{0};
    Real residual{0.0};
    Real computation_time{0.0};
    
    std::map<std::string, Real> node_pressures;
    std::map<std::string, Real> node_temperatures;
    std::map<std::string, Real> pipe_flow_rates;
    std::map<std::string, Real> pipe_pressure_drops;
};

namespace ml {

/// Feature extractor for ML models
class FeatureExtractor {
public:
    static Vector extract_features(
        const Network& network,
        const SolutionResults& results,
        const FluidProperties& fluid
    );
    
    static std::vector<std::string> get_feature_names();
    static void normalize_features(Vector& features);
};

/// Base ML model interface
class MLModel {
public:
    virtual ~MLModel() = default;
    virtual void load(const std::string& filename) = 0;
    virtual Vector predict(const Vector& features) = 0;
    virtual std::string info() const = 0;
};

/// Flow pattern predictor
class FlowPatternPredictor : public MLModel {
public:
    void load(const std::string& filename) override;
    Vector predict(const Vector& features) override;
    std::string info() const override { return "Flow Pattern Predictor"; }
    
    FlowPattern predict_pattern(
        const Pipe& pipe,
        const FluidProperties& fluid,
        Real flow_rate
    );
    
private:
    struct NeuralNetwork {
        std::vector<Matrix> weights;
        std::vector<Vector> biases;
        Vector forward(const Vector& input);
    };
    
    std::unique_ptr<NeuralNetwork> network_;
};

/// Anomaly detector
class AnomalyDetector : public MLModel {
public:
    void load(const std::string& filename) override;
    Vector predict(const Vector& features) override;
    std::string info() const override { return "Anomaly Detector"; }
    
    struct AnomalyResult {
        bool is_anomaly;
        Real anomaly_score;
        std::vector<std::string> anomaly_features;
    };
    
    AnomalyResult detect(
        const Network& network,
        const SolutionResults& results
    );
    
private:
    struct IsolationTree {
        struct Node {
            int feature_index{-1};
            Real split_value{0.0};
            std::unique_ptr<Node> left;
            std::unique_ptr<Node> right;
            int depth{0};
        };
        
        std::unique_ptr<Node> root;
        Real path_length(const Vector& sample) const;
    };
    
    std::vector<IsolationTree> forest_;
};

/// ML Optimizer - Modified to not use SteadyStateSolver directly
class MLOptimizer {
public:
    struct OptimizationObjective {
        enum Type {
            MINIMIZE_PRESSURE_DROP,
            MAXIMIZE_FLOW_RATE,
            MINIMIZE_ENERGY_CONSUMPTION,
            CUSTOM
        };
        
        Type type;
        std::function<Real(const SolutionResults&)> custom_function;
    };
    
    struct OptimizationConstraints {
        Real min_pressure{1e5};
        Real max_pressure{100e5};
        Real max_velocity{10.0};
        std::map<std::string, Real> node_flow_demands;
    };
    
    struct OptimizationResult {
        bool success;
        Real objective_value;
        std::map<std::string, Real> pump_speeds;
        std::map<std::string, Real> valve_openings;
        std::map<std::string, Real> compressor_ratios;
    };
    
    OptimizationResult optimize(
        Network& network,
        const FluidProperties& fluid,
        const OptimizationObjective& objective,
        const OptimizationConstraints& constraints
    );
    
    // Add a solver callback to avoid direct dependency
    using SolverCallback = std::function<SolutionResults(Network&, const FluidProperties&)>;
    void set_solver_callback(SolverCallback callback) { solver_callback_ = callback; }
    
private:
    struct Individual {
        Vector genes;
        Real fitness;
    };
    
    std::vector<Individual> population_;
    SolverCallback solver_callback_;
    
    void initialize_population(size_t size);
    void evaluate_fitness(
        Network& network,
        const FluidProperties& fluid,
        const OptimizationObjective& objective
    );
    void selection();
    void crossover();
    void mutation();
};

/// Data-driven correlation
class DataDrivenCorrelation : public FlowCorrelation {
public:
    void train(
        const std::vector<Vector>& features,
        const std::vector<Real>& pressure_drops
    );
    
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
    
    Results calculate(
        const FluidProperties& fluid,
        const Pipe& pipe,
        Real flow_rate,
        Real inlet_pressure,
        Real inlet_temperature
    ) const override;
    
    std::string name() const override { return "Data-Driven Model"; }
    
private:
    struct DecisionTree {
        struct Node {
            int feature_index{-1};
            Real threshold{0.0};
            Real value{0.0};
            std::unique_ptr<Node> left;
            std::unique_ptr<Node> right;
            
            bool is_leaf() const { return !left && !right; }
        };
        
        std::unique_ptr<Node> root;
        Real predict(const Vector& features) const;
    };
    
    std::vector<DecisionTree> forest_;
    Vector feature_importance_;
};

/// Digital Twin
class DigitalTwin {
public:
    void initialize(
        const Network& network,
        const FluidProperties& fluid
    );
    
    void update_with_measurements(
        const std::map<std::string, Real>& pressure_measurements,
        const std::map<std::string, Real>& flow_measurements,
        Real timestamp
    );
    
    struct EstimatedState {
        std::map<std::string, Real> node_pressures;
        std::map<std::string, Real> pipe_flows;
        std::map<std::string, Real> uncertainties;
    };
    
    EstimatedState estimate_state();
    EstimatedState predict_future(Real time_horizon);
    
    struct Discrepancy {
        std::string location;
        std::string type;
        Real severity;
        Real confidence;
    };
    
    std::vector<Discrepancy> detect_discrepancies();
    
private:
    Ptr<Network> network_;
    Ptr<FluidProperties> fluid_;
    
    Vector state_estimate_;
    Matrix covariance_matrix_;
    Matrix process_noise_;
    Matrix measurement_noise_;
    
    std::deque<EstimatedState> state_history_;
    std::deque<Real> time_history_;
    
    std::unique_ptr<AnomalyDetector> anomaly_detector_;
    std::unique_ptr<FlowPatternPredictor> pattern_predictor_;
};

} // namespace ml
} // namespace pipeline_sim
