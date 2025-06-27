# ===== python/examples/multiphase_flow.py =====
# AI_GENERATED: Multiphase flow correlation example
import pipeline_sim as ps
from pipeline_sim.correlations import BeggsBrill, HagedornBrown
import numpy as np
import matplotlib.pyplot as plt


def analyze_flow_patterns():
    """Analyze flow patterns for different conditions"""
    
    # Create test pipe
    network = ps.Network()
    n1 = network.add_node("n1", ps.NodeType.SOURCE)
    n2 = network.add_node("n2", ps.NodeType.SINK)
    pipe = network.add_pipe("test_pipe", n1, n2, length=1000, diameter=0.2)
    
    # Fluid properties
    fluid = ps.FluidProperties()
    fluid.oil_density = 850
    fluid.gas_density = 0.85
    fluid.water_density = 1025
    fluid.oil_viscosity = 0.01
    fluid.gas_viscosity = 1.8e-5
    fluid.water_viscosity = 0.001
    
    # Vary gas fraction
    gas_fractions = np.linspace(0, 0.9, 20)
    flow_patterns = []
    pressure_gradients = []
    holdups = []
    
    for gf in gas_fractions:
        fluid.gas_fraction = gf
        fluid.oil_fraction = (1 - gf) * 0.7
        fluid.water_fraction = (1 - gf) * 0.3
        
        # Calculate using Beggs-Brill
        results = BeggsBrill.calculate(fluid, pipe, flow_rate=0.1)
        
        flow_patterns.append(results.flow_pattern)
        pressure_gradients.append(results.pressure_gradient)
        holdups.append(results.liquid_holdup)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    # Flow patterns
    ax1.scatter(gas_fractions, flow_patterns, c=flow_patterns, 
               cmap='viridis', s=50)
    ax1.set_ylabel('Flow Pattern')
    ax1.set_ylim(-0.5, 3.5)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['Segregated', 'Intermittent', 'Distributed', 'Annular'])
    ax1.set_title('Flow Pattern vs Gas Fraction')
    ax1.grid(True, alpha=0.3)
    
    # Pressure gradient
    ax2.plot(gas_fractions, np.array(pressure_gradients)/1000, 'b-', linewidth=2)
    ax2.set_ylabel('Pressure Gradient (kPa/m)')
    ax2.set_title('Pressure Gradient vs Gas Fraction')
    ax2.grid(True, alpha=0.3)
    
    # Liquid holdup
    ax3.plot(gas_fractions, holdups, 'r-', linewidth=2)
    ax3.plot(gas_fractions, 1-gas_fractions, 'k--', alpha=0.5, label='No-slip')
    ax3.set_xlabel('Gas Fraction')
    ax3.set_ylabel('Liquid Holdup')
    ax3.set_title('Liquid Holdup vs Gas Fraction')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('flow_pattern_analysis.png', dpi=150)
    plt.show()


def compare_correlations():
    """Compare different flow correlations"""
    
    print("Comparing Multiphase Flow Correlations")
    print("=" * 40)
    
    # Test conditions
    network = ps.Network()
    n1 = network.add_node("n1", ps.NodeType.SOURCE)
    n2 = network.add_node("n2", ps.NodeType.SINK) 
    
    # Vertical pipe
    pipe_v = network.add_pipe("vertical", n1, n2, length=1000, diameter=0.15)
    pipe_v.inclination = np.pi/2  # 90 degrees
    
    # Horizontal pipe
    pipe_h = network.add_pipe("horizontal", n1, n2, length=1000, diameter=0.15)
    pipe_h.inclination = 0
    
    # Inclined pipe
    pipe_i = network.add_pipe("inclined", n1, n2, length=1000, diameter=0.15)
    pipe_i.inclination = np.pi/6  # 30 degrees
    
    # Fluid
    fluid = ps.FluidProperties()
    fluid.oil_density = 900
    fluid.gas_density = 0.9
    fluid.water_density = 1030
    fluid.oil_viscosity = 0.02
    fluid.gas_viscosity = 1.5e-5
    fluid.water_viscosity = 0.001
    fluid.gas_fraction = 0.3
    fluid.oil_fraction = 0.5
    fluid.water_fraction = 0.2
    
    pipes = [("Horizontal", pipe_h), ("Inclined", pipe_i), ("Vertical", pipe_v)]
    
    print("\nBeggs-Brill Correlation Results:")
    print(f"{'Configuration':<15} {'Pattern':<15} {'ΔP (kPa/m)':<12} {'Holdup':<8}")
    print("-" * 50)
    
    for name, pipe in pipes:
        bb_result = BeggsBrill.calculate(fluid, pipe, flow_rate=0.05)
        print(f"{name:<15} {bb_result.flow_pattern_name:<15} "
              f"{bb_result.pressure_gradient/1000:>10.2f} "
              f"{bb_result.liquid_holdup:>8.3f}")
    
    # For vertical flow, also show Hagedorn-Brown
    print("\nHagedorn-Brown (Vertical only):")
    hb_result = HagedornBrown.calculate(fluid, pipe_v, flow_rate=0.05)
    print(f"Pressure Drop: {hb_result['pressure_drop']/1000:.2f} kPa")
    print(f"Flow Regime: {hb_result['flow_regime']}")


if __name__ == "__main__":
    print("Running multiphase flow analysis...")
    analyze_flow_patterns()
    compare_correlations()
    print("\nAnalysis complete! Check generated plots.")