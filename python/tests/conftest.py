# ===== python/tests/conftest.py =====
# AI_GENERATED: pytest configuration and fixtures
import pytest
import pipeline_sim as ps


@pytest.fixture
def simple_network():
    """Create a simple test network"""
    network = ps.Network()
    
    source = network.add_node("source", ps.NodeType.SOURCE)
    sink = network.add_node("sink", ps.NodeType.SINK)
    pipe = network.add_pipe("pipe", source, sink, 1000, 0.3)
    
    network.set_pressure(source, 50e5)
    network.set_flow_rate(sink, 0.1)
    
    return network


@pytest.fixture
def complex_network():
    """Create a more complex test network"""
    network = ps.Network()
    
    # Create nodes
    wellhead = network.add_node("wellhead", ps.NodeType.SOURCE)
    manifold = network.add_node("manifold", ps.NodeType.JUNCTION)
    booster = network.add_node("booster", ps.NodeType.PUMP)
    junction = network.add_node("junction", ps.NodeType.JUNCTION)
    separator1 = network.add_node("separator1", ps.NodeType.SINK)
    separator2 = network.add_node("separator2", ps.NodeType.SINK)
    
    # Create pipes
    network.add_pipe("riser", wellhead, manifold, 1500, 0.3)
    network.add_pipe("flowline1", manifold, booster, 2000, 0.4)
    network.add_pipe("flowline2", booster, junction, 3000, 0.4)
    network.add_pipe("branch1", junction, separator1, 1000, 0.3)
    network.add_pipe("branch2", junction, separator2, 1500, 0.3)
    
    # Set boundary conditions
    network.set_pressure(wellhead, 70e5)
    network.set_flow_rate(separator1, 0.06)
    network.set_flow_rate(separator2, 0.04)
    
    return network


@pytest.fixture
def test_fluid():
    """Create test fluid properties"""
    fluid = ps.FluidProperties()
    fluid.oil_density = 850
    fluid.gas_density = 0.85
    fluid.water_density = 1025
    fluid.oil_viscosity = 0.01
    fluid.gas_viscosity = 1.8e-5
    fluid.water_viscosity = 0.001
    fluid.gas_oil_ratio = 100
    fluid.water_cut = 0.2
    
    # Calculate fractions
    fluid.water_fraction = 0.2
    fluid.oil_fraction = 0.7
    fluid.gas_fraction = 0.1
    
    return fluid

