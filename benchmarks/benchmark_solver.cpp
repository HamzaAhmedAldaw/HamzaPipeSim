/// AI_GENERATED: Performance benchmarks and examples
/// Generated on: 2025-06-27

// ===== benchmarks/benchmark_solver.cpp =====
#include <benchmark/benchmark.h>
#include "pipeline_sim/pipeline_sim.h"
#include <random>

using namespace pipeline_sim;

// Create test networks of various sizes
Ptr<Network> create_linear_network(int num_segments) {
    auto network = std::make_shared<Network>();
    
    // Create nodes
    auto source = network->add_node("source", NodeType::SOURCE);
    Ptr<Node> prev = source;
    
    for (int i = 0; i < num_segments - 1; ++i) {
        auto node = network->add_node("node_" + std::to_string(i), NodeType::JUNCTION);
        network->add_pipe("pipe_" + std::to_string(i), prev, node, 1000.0, 0.3);
        prev = node;
    }
    
    auto sink = network->add_node("sink", NodeType::SINK);
    network->add_pipe("pipe_final", prev, sink, 1000.0, 0.3);
    
    // Set boundary conditions
    network->set_pressure(source, 50e5);
    network->set_flow_rate(sink, 0.1);
    
    return network;
}

Ptr<Network> create_grid_network(int grid_size) {
    auto network = std::make_shared<Network>();
    
    // Create grid of nodes
    std::vector<std::vector<Ptr<Node>>> grid(grid_size);
    
    for (int i = 0; i < grid_size; ++i) {
        grid[i].resize(grid_size);
        for (int j = 0; j < grid_size; ++j) {
            NodeType type = NodeType::JUNCTION;
            if (i == 0 && j == 0) type = NodeType::SOURCE;
            else if (i == grid_size-1 && j == grid_size-1) type = NodeType::SINK;
            
            grid[i][j] = network->add_node(
                "node_" + std::to_string(i) + "_" + std::to_string(j), type);
        }
    }
    
    // Connect grid
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            if (i < grid_size - 1) {
                network->add_pipe(
                    "pipe_v_" + std::to_string(i) + "_" + std::to_string(j),
                    grid[i][j], grid[i+1][j], 500.0, 0.2);
            }
            if (j < grid_size - 1) {
                network->add_pipe(
                    "pipe_h_" + std::to_string(i) + "_" + std::to_string(j),
                    grid[i][j], grid[i][j+1], 500.0, 0.2);
            }
        }
    }
    
    // Set boundary conditions
    network->set_pressure(grid[0][0], 70e5);
    network->set_flow_rate(grid[grid_size-1][grid_size-1], 0.2);
    
    return network;
}

// Benchmark steady-state solver
static void BM_SteadyStateSolver_Linear(benchmark::State& state) {
    int num_segments = state.range(0);
    auto network = create_linear_network(num_segments);
    
    FluidProperties fluid;
    fluid.oil_fraction = 0.7;
    fluid.gas_fraction = 0.3;
    
    for (auto _ : state) {
        SteadyStateSolver solver(network, fluid);
        solver.config().verbose = false;
        auto results = solver.solve();
        
        benchmark::DoNotOptimize(results.converged);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetLabel("segments=" + std::to_string(num_segments));
}

static void BM_SteadyStateSolver_Grid(benchmark::State& state) {
    int grid_size = state.range(0);
    auto network = create_grid_network(grid_size);
    
    FluidProperties fluid;
    fluid.oil_fraction = 0.7;
    fluid.gas_fraction = 0.3;
    
    for (auto _ : state) {
        SteadyStateSolver solver(network, fluid);
        solver.config().verbose = false;
        auto results = solver.solve();
        
        benchmark::DoNotOptimize(results.converged);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetLabel("grid=" + std::to_string(grid_size) + "x" + std::to_string(grid_size));
}

// Benchmark correlations
static void BM_BeggsBrill_Calculation(benchmark::State& state) {
    FluidProperties fluid;
    fluid.oil_fraction = 0.7;
    fluid.gas_fraction = 0.3;
    
    auto n1 = std::make_shared<Node>("n1", NodeType::SOURCE);
    auto n2 = std::make_shared<Node>("n2", NodeType::SINK);
    Pipe pipe("pipe", n1, n2, 1000.0, 0.3);
    
    BeggsBrillCorrelation correlation;
    
    for (auto _ : state) {
        auto result = correlation.calculate(fluid, pipe, 0.1, 50e5, 300.0);
        benchmark::DoNotOptimize(result.pressure_gradient);
    }
    
    state.SetItemsProcessed(state.iterations());
}

// Benchmark ML operations
static void BM_FeatureExtraction(benchmark::State& state) {
    auto network = create_grid_network(5);
    
    FluidProperties fluid;
    SteadyStateSolver solver(network, fluid);
    auto results = solver.solve();
    
    for (auto _ : state) {
        auto features = ml::FeatureExtractor::extract_features(*network, results, fluid);
        benchmark::DoNotOptimize(features);
    }
    
    state.SetItemsProcessed(state.iterations());
}

// Register benchmarks
BENCHMARK(BM_SteadyStateSolver_Linear)->Arg(10)->Arg(50)->Arg(100)->Arg(500);
BENCHMARK(BM_SteadyStateSolver_Grid)->Arg(3)->Arg(5)->Arg(10)->Arg(20);
BENCHMARK(BM_BeggsBrill_Calculation);
BENCHMARK(BM_FeatureExtraction);

BENCHMARK_MAIN();
