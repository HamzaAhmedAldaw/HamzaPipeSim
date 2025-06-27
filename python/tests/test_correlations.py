# ===== python/tests/test_correlations.py =====
# AI_GENERATED: Test flow correlations
import pytest
import numpy as np
from pipeline_sim.correlations import BeggsBrill, HagedornBrown
import pipeline_sim as ps


class TestBeggsBrill:
    """Test Beggs-Brill correlation"""
    
    def test_single_phase_liquid(self):
        """Test single-phase liquid flow"""
        # Create test conditions
        network = ps.Network()
        n1 = network.add_node("n1", ps.NodeType.SOURCE)
        n2 = network.add_node("n2", ps.NodeType.SINK)
        pipe = network.add_pipe("pipe", n1, n2, 1000, 0.2)
        
        fluid = ps.FluidProperties()
        fluid.oil_fraction = 1.0
        fluid.gas_fraction = 0.0
        fluid.water_fraction = 0.0
        
        # Calculate
        result = BeggsBrill.calculate(fluid, pipe, flow_rate=0.05)
        
        assert result.liquid_holdup == 1.0
        assert result.pressure_gradient > 0
        assert result.flow_pattern_name in ["Segregated", "Distributed"]
    
    def test_two_phase_flow(self):
        """Test two-phase flow patterns"""
        network = ps.Network()
        n1 = network.add_node("n1", ps.NodeType.SOURCE)
        n2 = network.add_node("n2", ps.NodeType.SINK)
        pipe = network.add_pipe("pipe", n1, n2, 1000, 0.2)
        
        fluid = ps.FluidProperties()
        
        # Test different gas fractions
        gas_fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for gf in gas_fractions:
            fluid.gas_fraction = gf
            fluid.oil_fraction = 1 - gf
            fluid.water_fraction = 0.0
            
            result = BeggsBrill.calculate(fluid, pipe, flow_rate=0.05)
            
            # Sanity checks
            assert 0 <= result.liquid_holdup <= 1.0
            assert result.pressure_gradient >= 0
            assert result.flow_pattern in range(4)
    
    def test_inclined_flow(self):
        """Test effect of pipe inclination"""
        network = ps.Network()
        n1 = network.add_node("n1", ps.NodeType.SOURCE)
        n2 = network.add_node("n2", ps.NodeType.SINK)
        
        fluid = ps.FluidProperties()
        fluid.oil_fraction = 0.7
        fluid.gas_fraction = 0.3
        
        # Test different inclinations
        inclinations = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
        pressure_gradients = []
        
        for angle in inclinations:
            pipe = network.add_pipe(f"pipe_{angle}", n1, n2, 1000, 0.2)
            pipe.inclination = angle
            
            result = BeggsBrill.calculate(fluid, pipe, flow_rate=0.05)
            pressure_gradients.append(result.pressure_gradient)
        
        # Upward flow should have higher pressure gradient
        assert pressure_gradients[4] > pressure_gradients[2]  # Vertical up > horizontal
        assert pressure_gradients[0] < pressure_gradients[2]  # Vertical down < horizontal


class TestMechanisticModel:
    """Test mechanistic flow model"""
    
    def test_flow_pattern_prediction(self):
        """Test flow pattern predictions match expected ranges"""
        # This would test the mechanistic model implementation
        pass
