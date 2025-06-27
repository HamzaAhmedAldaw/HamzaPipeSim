// ===== include/pipeline_sim/ml_integration.h =====
#pragma once

#include "pipeline_sim/types.h"
#include "pipeline_sim/network.h"
#include <vector>
#include <memory>

namespace pipeline_sim {
namespace ml {

/// Feature extractor for ML models
class FeatureExtractor {
public:
    /// Extract features from network state
    static Vector extract_features(
        const Network& network,
        const SolutionResults& results,
        const FluidProperties& fluid
    );
    
    /// Get feature names
    static std::vector<std::string> get_feature_names();
    
    /// Normalize features
    static void normalize_features(Vector& features);
};

/// Base ML model interface
class MLModel {
public:
    virtual ~MLModel() = default;
    
    /// Load model from file
    virtual void load(const std::string& filename) = 0;
    
    /// Make prediction
    virtual Vector predict(const Vector& features) = 0;
    
    /// Get model info
    virtual std::string info() const = 0;
};

/// Flow pattern predictor
class FlowPatternPredictor : public MLModel {
public:
    void load(const std::string& filename) override;
    Vector predict(const Vector& features) override;
    std::string info() const override { return "Flow Pattern Predictor"; }
    
    /// Predict flow pattern for a pipe
    FlowPattern predict_pattern(
        const Pipe& pipe,
        const FluidProperties& fluid,
        Real flow_rate
    );
    
private:
    // Simple neural network implementation
    struct NeuralNetwork {
        std::vector<Matrix> weights;
        std::vector<Vector> biases;
        
        Vector forward(const Vector& input);
    };
    
    std::unique_ptr<NeuralNetwork> network_;
};

/// Anomaly detector for pipeline monitoring
class AnomalyDetector : public MLModel {
public:
    void load(const std::string& filename) override;
    Vector predict(const Vector& features) override;
    std::string info() const override { return "Anomaly Detector"; }
    
    /// Detect anomalies in current state
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
    // Isolation Forest implementation
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

/// Optimization solver using ML
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
        Real min_pressure{1e5};      // Pa
        Real max_pressure{100e5};    // Pa
        Real max_velocity{10.0};     // m/s
        std::map<std::string, Real> node_flow_demands;
    };
    
    /// Optimize network operation
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
    
private:
    /// Genetic algorithm implementation
    struct Individual {
        Vector genes;  // Control variables
        Real fitness;
    };
    
    std::vector<Individual> population_;
    
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

/// Data-driven correlation model
class DataDrivenCorrelation : public FlowCorrelation {
public:
    /// Train model from historical data
    void train(
        const std::vector<Vector>& features,
        const std::vector<Real>& pressure_drops
    );
    
    /// Save/load trained model
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
    // Random Forest implementation
    struct DecisionTree {
        struct Node {
            int feature_index{-1};
            Real threshold{0.0};
            Real value{0.0};  // For leaf nodes
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

/// Pipeline digital twin
class DigitalTwin {
public:
    /// Initialize twin with physical network
    void initialize(
        const Network& network,
        const FluidProperties& fluid
    );
    
    /// Update twin with real-time data
    void update_with_measurements(
        const std::map<std::string, Real>& pressure_measurements,
        const std::map<std::string, Real>& flow_measurements,
        Real timestamp
    );
    
    /// State estimation using Kalman filter
    struct EstimatedState {
        std::map<std::string, Real> node_pressures;
        std::map<std::string, Real> pipe_flows;
        std::map<std::string, Real> uncertainties;
    };
    
    EstimatedState estimate_state();
    
    /// Predict future state
    EstimatedState predict_future(Real time_horizon);
    
    /// Detect discrepancies
    struct Discrepancy {
        std::string location;
        std::string type;  // "leak", "blockage", "sensor_fault"
        Real severity;
        Real confidence;
    };
    
    std::vector<Discrepancy> detect_discrepancies();
    
private:
    Ptr<Network> network_;
    FluidProperties fluid_;
    
    // Kalman filter state
    Vector state_estimate_;
    Matrix covariance_matrix_;
    Matrix process_noise_;
    Matrix measurement_noise_;
    
    // Historical data
    std::deque<EstimatedState> state_history_;
    std::deque<Real> time_history_;
    
    // ML models
    std::unique_ptr<AnomalyDetector> anomaly_detector_;
    std::unique_ptr<FlowPatternPredictor> pattern_predictor_;
};

} // namespace ml
} // namespace pipeline_sim