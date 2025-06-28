# correct_pipeline_usage.py
from pipeline_sim_loader import pipeline_sim as ps

# Create network
network = ps.Network()

# Add nodes
source_node = network.add_node("Source", ps.NodeType.SOURCE)
sink_node = network.add_node("Sink", ps.NodeType.SINK)

# Add pipe
pipe = network.add_pipe("Pipe1", source_node, sink_node, 1000.0, 0.2)

# Set boundary conditions
network.set_pressure(source_node, 2e6)  # 2 MPa
network.set_flow_rate(sink_node, 0.1)   # 0.1 m³/s

# Configure fluid
fluid = ps.FluidProperties()
fluid.oil_density = 850.0
fluid.oil_viscosity = 0.001
fluid.oil_fraction = 1.0

# Solve
solver = ps.SteadyStateSolver(network, fluid)
results = solver.solve()

# Display results - NO PARENTHESES for properties
print(f"Converged: {results.converged}")
print(f"Iterations: {results.iterations}")
print(f"Inlet pressure: {source_node.pressure/1e6:.2f} MPa")
print(f"Outlet pressure: {sink_node.pressure/1e6:.2f} MPa")
print(f"Flow rate: {pipe.flow_rate:.4f} m³/s")
print(f"Pressure drop: {(source_node.pressure - sink_node.pressure)/1e6:.2f} MPa")