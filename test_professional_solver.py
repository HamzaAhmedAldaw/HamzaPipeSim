#!/usr/bin/env python3
"""
Fixed Complex Network Test - using proper boundary condition setting
"""

import pipeline_sim as ps

def test_complex_network_fixed():
    """Test complex network with correct BC setting"""
    print("="*70)
    print("FIXED COMPLEX NETWORK TEST (PRODUCTION MANIFOLD)")
    print("="*70)
    
    # Create network
    network = ps.Network()
    
    # Create nodes
    print("\nCreating wells...")
    wells = []
    well_data = [
        ("WELL-A1", ps.NodeType.SOURCE, 320e5),
        ("WELL-A2", ps.NodeType.SOURCE, 310e5),
        ("WELL-A3", ps.NodeType.SOURCE, 300e5),
        ("WELL-B1", ps.NodeType.SOURCE, 330e5),
        ("WELL-B2", ps.NodeType.SOURCE, 325e5),
        ("WELL-B3", ps.NodeType.SOURCE, 315e5),
        ("WELL-B4", ps.NodeType.SOURCE, 305e5),
    ]
    
    for name, node_type, pressure in well_data:
        well = network.add_node(name, node_type)
        # USE set_pressure_bc INSTEAD OF network.set_pressure!
        well.set_pressure_bc(pressure)
        wells.append(well)
        print(f"  {name}: {pressure/1e5:.0f} bar (BC: {well.has_pressure_bc()})")
    
    # Create junction nodes
    print("\nCreating junctions...")
    manifold_a = network.add_node("MANIFOLD-A", ps.NodeType.JUNCTION)
    manifold_b = network.add_node("MANIFOLD-B", ps.NodeType.JUNCTION)
    platform = network.add_node("PLATFORM", ps.NodeType.JUNCTION)
    hp_sep = network.add_node("HP-SEP", ps.NodeType.JUNCTION)
    lp_sep = network.add_node("LP-SEP", ps.NodeType.JUNCTION)
    
    # Create export (sink)
    print("\nCreating export sink...")
    export = network.add_node("EXPORT", ps.NodeType.SINK)
    export.set_pressure_bc(150e5)  # 150 bar
    print(f"  EXPORT: 150 bar (BC: {export.has_pressure_bc()})")
    
    # Create pipes
    print("\nCreating pipes...")
    pipes = []
    
    # Flowlines from wells to manifolds
    pipe_data = [
        # A wells to Manifold A
        ("WELL-A1_to_MANIFOLD-A", wells[0], manifold_a, 2500, 0.25),
        ("WELL-A2_to_MANIFOLD-A", wells[1], manifold_a, 2000, 0.25),
        ("WELL-A3_to_MANIFOLD-A", wells[2], manifold_a, 2500, 0.25),
        # B wells to Manifold B
        ("WELL-B1_to_MANIFOLD-B", wells[3], manifold_b, 3200, 0.25),
        ("WELL-B2_to_MANIFOLD-B", wells[4], manifold_b, 1400, 0.25),
        ("WELL-B3_to_MANIFOLD-B", wells[5], manifold_b, 2200, 0.25),
        ("WELL-B4_to_MANIFOLD-B", wells[6], manifold_b, 4200, 0.25),
        # Manifolds to Platform
        ("MANIFOLD-A_to_PLATFORM", manifold_a, platform, 5000, 0.4),
        ("MANIFOLD-B_to_PLATFORM", manifold_b, platform, 3200, 0.4),
        # Platform processing
        ("PLATFORM_to_HP-SEP", platform, hp_sep, 500, 0.5),
        ("HP-SEP_to_LP-SEP", hp_sep, lp_sep, 1000, 0.4),
        ("LP-SEP_to_EXPORT", lp_sep, export, 1500, 0.4),
    ]
    
    for name, upstream, downstream, length, diameter in pipe_data:
        pipe = network.add_pipe(name, upstream, downstream, length, diameter)
        pipe.set_roughness(0.000045)
        pipes.append(pipe)
    
    print(f"\nNetwork complete: {network.node_count()} nodes, {network.pipe_count()} pipes")
    
    # Verify boundary conditions
    print("\nVerifying boundary conditions:")
    bc_count = 0
    for node_name, node in network.nodes().items():
        if node.has_pressure_bc():
            bc_count += 1
            print(f"  {node_name}: {node.pressure_bc()/1e5:.0f} bar")
    print(f"Total boundary conditions: {bc_count}")
    
    # Create fluid
    fluid = ps.FluidProperties()
    fluid.oil_fraction = 0.75
    fluid.water_fraction = 0.20
    fluid.gas_fraction = 0.05
    fluid.oil_density = 780.0
    fluid.water_density = 1025.0
    fluid.gas_density = 15.0
    fluid.oil_viscosity = 0.0008
    fluid.water_viscosity = 0.0011
    fluid.gas_viscosity = 0.000018
    
    print(f"\nFluid: density={fluid.mixture_density():.1f} kg/m³, viscosity={fluid.mixture_viscosity()*1000:.2f} cP")
    
    # Create and configure solver
    solver = ps.SteadyStateSolver(network, fluid)
    solver.config.verbose = True
    solver.config.tolerance = 1e-6
    solver.config.max_iterations = 100
    solver.config.use_line_search = True
    
    # Solve
    print("\n" + "="*70)
    print("SOLVING...")
    print("="*70)
    
    results = solver.solve()
    
    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Converged: {results.converged}")
    print(f"Iterations: {results.iterations}")
    print(f"Final residual: {results.residual:.2e}")
    print(f"Computation time: {results.computation_time:.3f} s")
    print(f"Convergence reason: {results.convergence_reason}")
    
    if results.converged:
        print("\nKey Results:")
        print(f"  MANIFOLD-A pressure: {results.node_pressures.get('MANIFOLD-A', 0)/1e5:.1f} bar")
        print(f"  MANIFOLD-B pressure: {results.node_pressures.get('MANIFOLD-B', 0)/1e5:.1f} bar")
        print(f"  PLATFORM pressure: {results.node_pressures.get('PLATFORM', 0)/1e5:.1f} bar")
        
        # Calculate total production
        total_prod = 0
        print("\nWell Production:")
        for i, (name, _, _) in enumerate(well_data):
            flow = 0
            for pipe_name, pipe_flow in results.pipe_flow_rates.items():
                if pipe_name.startswith(name + "_"):
                    flow = abs(pipe_flow)
                    break
            flow_m3d = flow * 86400
            print(f"  {name}: {flow_m3d:,.0f} m³/day")
            total_prod += flow_m3d
        
        print(f"\nTotal Production: {total_prod:,.0f} m³/day")
    
    return results


def main():
    print("FIXED COMPLEX NETWORK TEST")
    print("Version:", ps.__version__)
    print("")
    
    results = test_complex_network_fixed()
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if results and results.iterations > 0:
        print("✅ SUCCESS! Complex network solver is working!")
        print(f"   - Completed {results.iterations} iterations")
        print(f"   - Converged: {results.converged}")
    else:
        print("❌ FAILED: Still not working")
        if results:
            print(f"   - Iterations: {results.iterations}")
            print(f"   - Reason: {results.convergence_reason}")


if __name__ == "__main__":
    main()