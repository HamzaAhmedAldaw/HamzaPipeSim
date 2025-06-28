# simple_working_example.py
from pipeline_sim_loader import pipeline_sim as ps

# Create simple network
network = ps.Network()

# Two nodes, one pipe
source = network.add_node("Source", ps.NodeType.SOURCE)
sink = network.add_node("Sink", ps.NodeType.SINK)
pipe = network.add_pipe("Pipe", source, sink, 1000.0, 0.15)

# Set boundary conditions
source.set_pressure_bc(3e6)  # 3 MPa
sink.set_pressure_bc(1e6)    # 1 MPa

# Simple fluid
fluid = ps.FluidProperties()
fluid.oil_density = 850.0
fluid.oil_viscosity = 0.002
fluid.oil_fraction = 1.0

# Solve
solver = ps.SteadyStateSolver(network, fluid)
results = solver.solve()

print(f"Converged: {results.converged}")
if results.converged:
    # Calculate flow using Darcy-Weisbach
    dp = source.pressure - sink.pressure
    print(f"Pressure drop: {dp/1e6:.2f} MPa")
    print(f"Flow rate: {results.pipe_flow_rates.get('Pipe', 0):.4f} m³/s")
    
    # Check if results contain the data differently
    print("\nAll results:")
    print(f"Node pressures: {list(results.node_pressures.keys())}")
    print(f"Pipe flows: {list(results.pipe_flow_rates.keys())}")