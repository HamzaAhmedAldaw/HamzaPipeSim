# ===== python/examples/network_from_json.py =====
# AI_GENERATED: Example loading network from JSON
import pipeline_sim as ps
import sys


def main():
    """Load and simulate network from JSON file"""
    
    if len(sys.argv) < 2:
        print("Usage: python network_from_json.py <network.json>")
        print("Using default example file...")
        filename = "../../examples/simple_network.json"
    else:
        filename = sys.argv[1]
    
    print(f"Loading network from: {filename}")
    
    try:
        # Load network and fluid properties
        network, fluid = ps.load_network(filename)
        
        print(f"\nNetwork loaded successfully!")
        print(f"  Nodes: {len(network.nodes)}")
        print(f"  Pipes: {len(network.pipes)}")
        
        # Create solver
        solver = ps.SteadyStateSolver(network, fluid)
        
        # Run simulation
        print("\nRunning simulation...")
        results = solver.solve()
        
        if results.converged:
            print("Simulation converged!")
            
            # Display key results
            print("\nPressure Summary:")
            pressures = list(results.node_pressures.values())
            print(f"  Min: {min(pressures)/1e5:.2f} bar")
            print(f"  Max: {max(pressures)/1e5:.2f} bar")
            
            print("\nFlow Summary:")
            flows = list(results.pipe_flow_rates.values())
            print(f"  Total: {sum(flows):.4f} m³/s")
            
            # Generate visualization
            fig = ps.plot_network(network, results)
            fig.savefig('network_visualization.png', dpi=150)
            print("\nNetwork visualization saved to network_visualization.png")
            
            # Generate report
            ps.generate_report(network, results, fluid)
            print("Detailed report saved to simulation_report.html")
            
        else:
            print("Simulation failed to converge!")
            print(f"Final residual: {results.residual:.2e}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

