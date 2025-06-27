# ===== python/tests/test_network.py =====
# AI_GENERATED: Network component tests
import pytest
import numpy as np
import pipeline_sim as ps


class TestNetwork:
    """Test network creation and manipulation"""
    
    def test_create_empty_network(self):
        """Test creating an empty network"""
        network = ps.Network()
        assert len(network.nodes) == 0
        assert len(network.pipes) == 0
    
    def test_add_nodes(self):
        """Test adding nodes to network"""
        network = ps.Network()
        
        # Add various node types
        n1 = network.add_node("n1", ps.NodeType.SOURCE)
        n2 = network.add_node("n2", ps.NodeType.JUNCTION)
        n3 = network.add_node("n3", ps.NodeType.SINK)
        
        assert len(network.nodes) == 3
        assert n1.id == "n1"
        assert n1.type == ps.NodeType.SOURCE
        assert n2.type == ps.NodeType.JUNCTION
        assert n3.type == ps.NodeType.SINK
    
    def test_add_pipes(self):
        """Test adding pipes between nodes"""
        network = ps.Network()
        
        n1 = network.add_node("n1", ps.NodeType.SOURCE)
        n2 = network.add_node("n2", ps.NodeType.SINK)
        
        pipe = network.add_pipe("p1", n1, n2, length=1000, diameter=0.3)
        
        assert len(network.pipes) == 1
        assert pipe.id == "p1"
        assert pipe.length == 1000
        assert pipe.diameter == 0.3
        assert pipe.upstream.id == "n1"
        assert pipe.downstream.id == "n2"
    
    def test_node_connectivity(self):
        """Test node connectivity queries"""
        network = ps.Network()
        
        n1 = network.add_node("n1", ps.NodeType.SOURCE)
        n2 = network.add_node("n2", ps.NodeType.JUNCTION)
        n3 = network.add_node("n3", ps.NodeType.SINK)
        
        p1 = network.add_pipe("p1", n1, n2, 500, 0.2)
        p2 = network.add_pipe("p2", n2, n3, 500, 0.2)
        
        # Check upstream/downstream pipes
        assert len(network.get_upstream_pipes(n2)) == 1
        assert len(network.get_downstream_pipes(n2)) == 1
        assert network.get_upstream_pipes(n2)[0].id == "p1"
        assert network.get_downstream_pipes(n2)[0].id == "p2"
    
    def test_boundary_conditions(self):
        """Test setting boundary conditions"""
        network = ps.Network()
        
        n1 = network.add_node("n1", ps.NodeType.SOURCE)
        n2 = network.add_node("n2", ps.NodeType.SINK)
        
        # Set pressure BC
        network.set_pressure(n1, 50e5)  # 50 bar
        assert n1.pressure == 50e5
        
        # Set flow BC
        network.set_flow_rate(n2, 0.1)  # 0.1 m³/s
        
    def test_pipe_properties(self):
        """Test pipe property calculations"""
        network = ps.Network()
        
        n1 = network.add_node("n1", ps.NodeType.SOURCE)
        n2 = network.add_node("n2", ps.NodeType.SINK)
        
        pipe = network.add_pipe("p1", n1, n2, length=1000, diameter=0.3)
        
        # Test computed properties
        expected_area = np.pi * 0.3**2 / 4
        assert abs(pipe.area() - expected_area) < 1e-6
        
        expected_volume = expected_area * 1000
        assert abs(pipe.volume() - expected_volume) < 1e-6
        
        # Set flow and check velocity
        pipe.set_flow_rate(0.1)
        expected_velocity = 0.1 / expected_area
        assert abs(pipe.velocity() - expected_velocity) < 1e-6


class TestFluidProperties:
    """Test fluid property calculations"""
    
    def test_default_properties(self):
        """Test default fluid properties"""
        fluid = ps.FluidProperties()
        
        assert fluid.oil_density == 850.0
        assert fluid.gas_density == 0.85
        assert fluid.water_density == 1025.0
    
    def test_mixture_properties(self):
        """Test mixture property calculations"""
        fluid = ps.FluidProperties()
        
        # Single phase
        fluid.oil_fraction = 1.0
        fluid.gas_fraction = 0.0
        fluid.water_fraction = 0.0
        
        assert fluid.mixture_density() == fluid.oil_density
        assert fluid.mixture_viscosity() == fluid.oil_viscosity
        
        # Two-phase
        fluid.oil_fraction = 0.5
        fluid.water_fraction = 0.5
        
        expected_density = 0.5 * 850 + 0.5 * 1025
        assert abs(fluid.mixture_density() - expected_density) < 1e-6
    
    def test_liquid_holdup(self):
        """Test liquid holdup calculation"""
        fluid = ps.FluidProperties()
        
        fluid.oil_fraction = 0.7
        fluid.gas_fraction = 0.3
        fluid.water_fraction = 0.0
        
        # Segregated flow
        holdup = fluid.liquid_holdup(0)  # 0 = segregated
        assert 0.7 <= holdup <= 1.0
        
        # No-slip condition
        holdup = fluid.liquid_holdup(3)  # 3 = annular
        assert abs(holdup - 0.7) < 0.1


class TestSolver:
    """Test solver functionality"""
    
    def test_simple_network_steady_state(self):
        """Test steady-state solution of simple network"""
        # Create simple network
        network = ps.Network()
        
        source = network.add_node("source", ps.NodeType.SOURCE)
        sink = network.add_node("sink", ps.NodeType.SINK)
        pipe = network.add_pipe("pipe", source, sink, 1000, 0.3)
        
        # Set boundary conditions
        network.set_pressure(source, 50e5)  # 50 bar
        network.set_flow_rate(sink, 0.1)    # 0.1 m³/s
        
        # Create fluid
        fluid = ps.FluidProperties()
        fluid.oil_fraction = 1.0
        fluid.gas_fraction = 0.0
        fluid.water_fraction = 0.0
        
        # Solve
        solver = ps.SteadyStateSolver(network, fluid)
        solver.config.tolerance = 1e-6
        solver.config.max_iterations = 100
        
        results = solver.solve()
        
        assert results.converged
        assert results.iterations > 0
        assert results.residual < 1e-6
        
        # Check results make sense
        assert results.node_pressures["source"] == 50e5
        assert results.node_pressures["sink"] < 50e5  # Pressure drop
        assert abs(results.pipe_flow_rates["pipe"] - 0.1) < 1e-6
    
    def test_network_with_junction(self):
        """Test network with junction node"""
        network = ps.Network()
        
        # Create Y-shaped network
        source = network.add_node("source", ps.NodeType.SOURCE)
        junction = network.add_node("junction", ps.NodeType.JUNCTION)
        sink1 = network.add_node("sink1", ps.NodeType.SINK)
        sink2 = network.add_node("sink2", ps.NodeType.SINK)
        
        pipe1 = network.add_pipe("pipe1", source, junction, 1000, 0.3)
        pipe2 = network.add_pipe("pipe2", junction, sink1, 500, 0.2)
        pipe3 = network.add_pipe("pipe3", junction, sink2, 500, 0.2)
        
        # Boundary conditions
        network.set_pressure(source, 50e5)
        network.set_flow_rate(sink1, 0.06)
        network.set_flow_rate(sink2, 0.04)
        
        # Solve
        fluid = ps.FluidProperties()
        solver = ps.SteadyStateSolver(network, fluid)
        results = solver.solve()
        
        assert results.converged
        
        # Check mass balance at junction
        q_in = results.pipe_flow_rates["pipe1"]
        q_out1 = results.pipe_flow_rates["pipe2"]
        q_out2 = results.pipe_flow_rates["pipe3"]
        
        assert abs(q_in - (q_out1 + q_out2)) < 1e-6
