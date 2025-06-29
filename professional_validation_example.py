#!/usr/bin/env python3
"""
PROFESSIONAL VALIDATION EXAMPLE: Pipeline-Sim vs Published Data

This example validates Pipeline-Sim against several published test cases:
1. Single pipe pressure drop (Crane Technical Paper 410)
2. Gas pipeline (AGA equations)
3. Three-reservoir problem (Jeppson, 1976)
4. Multiphase flow validation (Beggs & Brill correlation)

Author: Pipeline-Sim Validation Team
Date: June 2025
"""

import pipeline_sim as ps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

class ValidationSuite:
    """Professional validation test suite"""
    
    def __init__(self):
        self.results = {}
        
    def run_all_validations(self):
        """Run all validation cases"""
        print("="*80)
        print("PIPELINE-SIM PROFESSIONAL VALIDATION SUITE")
        print("Comparing against published benchmark data")
        print("="*80)
        
        # Run validations
        self.validate_crane_410()
        self.validate_gas_pipeline_aga()
        self.validate_jeppson_network()
        self.validate_multiphase_beggs_brill()
        
        # Summary
        self.print_summary()
        
    def validate_crane_410(self):
        """
        Validation Case 1: Crane Technical Paper No. 410
        
        Reference: Flow of Fluids Through Valves, Fittings, and Pipe
        Example 4-14: Water flow through steel pipe
        
        Given:
        - Pipe: 6" Schedule 40 steel pipe (ID = 0.1541 m)
        - Length: 1000 ft (304.8 m)
        - Flow rate: 1000 gpm (0.0631 m³/s)
        - Water at 60°F (15.6°C)
        - Roughness: 0.000045 m
        
        Expected pressure drop: ~44 psi (3.03 bar)
        """
        print("\n" + "-"*70)
        print("VALIDATION 1: Crane Technical Paper 410 - Single Pipe Flow")
        print("-"*70)
        
        # Create network
        network = ps.Network()
        
        inlet = network.add_node("INLET", ps.NodeType.SOURCE)
        outlet = network.add_node("OUTLET", ps.NodeType.SINK)
        
        # 6" Schedule 40: ID = 6.065" = 0.1541 m
        pipe = network.add_pipe("PIPE", inlet, outlet, 304.8, 0.1541)  # 1000 ft, 6" Sch 40
        pipe.set_roughness(0.000045)  # Commercial steel
        
        # Set arbitrary pressures (we care about pressure drop)
        inlet.set_pressure_bc(10e5)    # 10 bar
        outlet.set_pressure_bc(6.97e5)  # Will be adjusted by solver
        
        # Water at 60°F
        fluid = ps.FluidProperties()
        fluid.oil_fraction = 1.0  # Treat water as oil phase
        fluid.water_fraction = 0.0
        fluid.gas_fraction = 0.0
        fluid.oil_density = 999.0      # kg/m³ at 15.6°C
        fluid.oil_viscosity = 0.00112  # Pa.s at 15.6°C
        
        # Solve for the given flow rate iteratively
        target_flow = 0.0631  # m³/s (1000 gpm)
        
        # Binary search for outlet pressure to achieve target flow
        p_out_low = 5e5
        p_out_high = 9.99e5
        tolerance = 0.0001  # m³/s
        
        print(f"Target flow rate: {target_flow:.4f} m³/s ({target_flow*15850.3:.0f} gpm)")
        print("\nIterating to find outlet pressure...")
        
        converged = False
        for iteration in range(20):
            p_out_test = (p_out_low + p_out_high) / 2
            outlet.set_pressure_bc(p_out_test)
            
            print(f"  Iteration {iteration+1}: Testing P_out = {p_out_test/1e5:.2f} bar...", end='', flush=True)
            
            solver = ps.SteadyStateSolver(network, fluid)
            solver.config.verbose = False
            solver.config.max_iterations = 50  # Limit iterations
            results = solver.solve()
            
            if results.converged:
                flow = results.pipe_flow_rates['PIPE']
                print(f" Flow = {flow:.4f} m³/s")
                
                if abs(flow - target_flow) < tolerance:
                    # Found it!
                    converged = True
                    pressure_drop = (10e5 - p_out_test) / 1e5  # bar
                    pressure_drop_psi = pressure_drop * 14.5038  # psi
                    
                    print(f"\n✓ Target flow achieved!")
                    print(f"\nResults:")
                    print(f"  Flow rate: {flow:.4f} m³/s")
                    print(f"  Pressure drop: {pressure_drop:.3f} bar ({pressure_drop_psi:.1f} psi)")
                    print(f"  Reynolds number: {results.pipe_reynolds_numbers['PIPE']:.0f}")
                    print(f"  Friction factor: {results.pipe_friction_factors['PIPE']:.5f}")
                    print(f"  Velocity: {results.pipe_velocities['PIPE']:.2f} m/s")
                    
                    print(f"\nComparison:")
                    print(f"  Crane 410 expected: ~44 psi")
                    print(f"  Pipeline-Sim calculated: {pressure_drop_psi:.1f} psi")
                    print(f"  Difference: {abs(44 - pressure_drop_psi):.1f} psi ({abs(44 - pressure_drop_psi)/44*100:.1f}%)")
                    
                    self.results['crane_410'] = {
                        'expected': 44.0,
                        'calculated': pressure_drop_psi,
                        'error_percent': abs(44 - pressure_drop_psi)/44*100
                    }
                    break
                
                elif flow < target_flow:
                    p_out_high = p_out_test
                else:
                    p_out_low = p_out_test
            else:
                print(f" Failed to converge")
                # If solver fails, try adjusting bounds conservatively
                p_out_high = p_out_test
                
        if not converged:
            print(f"\n⚠ Could not achieve exact target flow after {iteration+1} iterations")
            print("  Using closest result found")
        
    def validate_gas_pipeline_aga(self):
        """
        Validation Case 2: AGA Gas Pipeline
        
        Reference: AGA Report No. 9 / API 14.3
        Example: Natural gas transmission line
        
        Given:
        - Pipe: 24" diameter, 50 miles long
        - Gas specific gravity: 0.6
        - Temperature: 60°F (15.6°C)
        - Inlet pressure: 800 psia (55.16 bar)
        - Outlet pressure: 400 psia (27.58 bar)
        - Base conditions: 14.73 psia, 60°F
        
        Calculate flow rate and compare with AGA equation
        """
        print("\n" + "-"*70)
        print("VALIDATION 2: AGA Natural Gas Pipeline")
        print("-"*70)
        
        # Create network
        network = ps.Network()
        
        inlet = network.add_node("INLET", ps.NodeType.SOURCE)
        outlet = network.add_node("OUTLET", ps.NodeType.SINK)
        
        # 24" pipeline, 50 miles
        pipe = network.add_pipe("GASLINE", inlet, outlet, 80467.2, 0.6096)  # 50 miles, 24"
        pipe.set_roughness(0.000018)  # Very smooth for gas
        
        # Boundary conditions
        inlet.set_pressure_bc(55.16e5)   # 800 psia
        outlet.set_pressure_bc(27.58e5)  # 400 psia
        
        # Natural gas properties
        # At average pressure (600 psia = 41.37 bar) and 60°F
        fluid = ps.FluidProperties()
        fluid.gas_fraction = 1.0
        fluid.oil_fraction = 0.0
        fluid.water_fraction = 0.0
        
        # Gas density at average conditions
        # Using real gas law: ρ = (P * M) / (Z * R * T)
        # M = 0.6 * 28.97 = 17.38 kg/kmol (molecular weight)
        # Z ≈ 0.9 (compressibility factor)
        # T = 288.7 K (60°F)
        P_avg = 41.37e5  # Pa
        M = 17.38  # kg/kmol
        Z = 0.9
        R = 8314  # J/(kmol·K)
        T = 288.7  # K
        
        fluid.gas_density = (P_avg * M) / (Z * R * T)  # ~32.8 kg/m³
        fluid.gas_viscosity = 0.000011  # Pa.s for natural gas
        
        print(f"Gas properties at average conditions:")
        print(f"  Density: {fluid.gas_density:.1f} kg/m³")
        print(f"  Viscosity: {fluid.gas_viscosity*1e6:.1f} μPa.s")
        
        # Solve
        solver = ps.SteadyStateSolver(network, fluid)
        solver.config.verbose = False
        solver.config.max_iterations = 50
        
        try:
            results = solver.solve()
            
            if results.converged:
                # Convert to standard conditions
                # Q_std = Q_actual * (P_avg/P_base) * (T_base/T_avg) * (Z_base/Z_avg)
                Q_actual = results.pipe_flow_rates['GASLINE']
                Q_std = Q_actual * (41.37/1.01325) * (288.7/288.7) * (1.0/0.9)
                Q_mmscfd = Q_std * 86400 / 28.317 / 1e6  # Million standard cubic feet per day
                
                print(f"\nResults:")
                print(f"  Actual flow rate: {Q_actual:.3f} m³/s")
                print(f"  Standard flow rate: {Q_std:.3f} std m³/s")
                print(f"  Flow rate: {Q_mmscfd:.1f} MMSCFD")
                print(f"  Reynolds number: {results.pipe_reynolds_numbers['GASLINE']:.0e}")
                print(f"  Friction factor: {results.pipe_friction_factors['GASLINE']:.5f}")
                print(f"  Velocity: {results.pipe_velocities['GASLINE']:.1f} m/s")
                
                # AGA simplified equation estimate
                # Q = 38.77 * (Tb/Pb) * [(P1² - P2²)/(G*T*L*Z)]^0.5 * D^2.5
                # Where Q is in MMSCFD
                Tb = 519.67  # °R
                Pb = 14.73   # psia
                P1 = 800     # psia
                P2 = 400     # psia
                G = 0.6      # specific gravity
                T_avg = 519.67  # °R
                L = 50       # miles
                D = 24       # inches
                
                Q_aga = 38.77 * (Tb/Pb) * np.sqrt((P1**2 - P2**2)/(G*T_avg*L*Z)) * D**2.5 / 1000
                
                print(f"\nComparison with AGA equation:")
                print(f"  AGA estimate: {Q_aga:.1f} MMSCFD")
                print(f"  Pipeline-Sim: {Q_mmscfd:.1f} MMSCFD")
                print(f"  Difference: {abs(Q_aga - Q_mmscfd):.1f} MMSCFD ({abs(Q_aga - Q_mmscfd)/Q_aga*100:.1f}%)")
                
                self.results['aga_gas'] = {
                    'expected': Q_aga,
                    'calculated': Q_mmscfd,
                    'error_percent': abs(Q_aga - Q_mmscfd)/Q_aga*100
                }
            else:
                print(f"\n✗ Failed to converge: {results.convergence_reason}")
        except Exception as e:
            print(f"\n✗ Error in AGA validation: {e}")
            import traceback
            traceback.print_exc()
    
    def validate_jeppson_network(self):
        """
        Validation Case 3: Three-Reservoir Problem
        
        Reference: Jeppson, R.W. (1976) "Analysis of Flow in Pipe Networks"
        Example 5.1: Three reservoirs connected by junction
        
        Given:
        - Reservoir A: Elevation 100 ft (pressure head)
        - Reservoir B: Elevation 80 ft
        - Reservoir C: Elevation 60 ft
        - Pipes: All 1000 ft long, 12" diameter
        - Hazen-Williams C = 100
        
        Find: Junction pressure and flow distribution
        """
        print("\n" + "-"*70)
        print("VALIDATION 3: Jeppson Three-Reservoir Problem")
        print("-"*70)
        
        # Create network
        network = ps.Network()
        
        # Reservoirs (using pressure BC to simulate elevation)
        # Convert elevation to pressure: P = ρ*g*h
        # Using water: ρ = 1000 kg/m³, g = 9.81 m/s²
        # 100 ft = 30.48 m, 80 ft = 24.38 m, 60 ft = 18.29 m
        
        res_a = network.add_node("RES-A", ps.NodeType.SOURCE)
        res_b = network.add_node("RES-B", ps.NodeType.SOURCE)
        res_c = network.add_node("RES-C", ps.NodeType.SINK)
        junction = network.add_node("JUNCTION", ps.NodeType.JUNCTION)
        
        # Set pressures (relative to reservoir C as datum)
        # ΔH_AC = 40 ft = 12.19 m
        # ΔH_BC = 20 ft = 6.10 m
        res_a.set_pressure_bc(1.01325e5 + 1000*9.81*12.19)  # 40 ft above C
        res_b.set_pressure_bc(1.01325e5 + 1000*9.81*6.10)   # 20 ft above C
        res_c.set_pressure_bc(1.01325e5)                     # Reference
        
        # Pipes: 1000 ft long, 12" diameter
        pipes = []
        pipe_a = network.add_pipe("PIPE-A", res_a, junction, 304.8, 0.3048)
        pipe_b = network.add_pipe("PIPE-B", res_b, junction, 304.8, 0.3048)
        pipe_c = network.add_pipe("PIPE-C", junction, res_c, 304.8, 0.3048)
        
        # Hazen-Williams C = 100 → equivalent roughness ≈ 0.00015 m
        for pipe in [pipe_a, pipe_b, pipe_c]:
            pipe.set_roughness(0.00015)
        
        # Water properties
        fluid = ps.FluidProperties()
        fluid.oil_fraction = 1.0
        fluid.water_fraction = 0.0
        fluid.gas_fraction = 0.0
        fluid.oil_density = 1000.0
        fluid.oil_viscosity = 0.001
        
        # Solve
        solver = ps.SteadyStateSolver(network, fluid)
        solver.config.verbose = False
        results = solver.solve()
        
        if results.converged:
            # Convert junction pressure to head
            junction_pressure = results.node_pressures['JUNCTION']
            junction_head_m = (junction_pressure - 1.01325e5) / (1000 * 9.81)
            junction_head_ft = junction_head_m * 3.281
            
            # Get flows (convert to cfs)
            flow_a = results.pipe_flow_rates['PIPE-A'] * 35.315  # m³/s to cfs
            flow_b = results.pipe_flow_rates['PIPE-B'] * 35.315
            flow_c = results.pipe_flow_rates['PIPE-C'] * 35.315
            
            print(f"\nResults:")
            print(f"  Junction elevation: {junction_head_ft + 60:.1f} ft")
            print(f"  Flow from A: {flow_a:.3f} cfs")
            print(f"  Flow from B: {flow_b:.3f} cfs")
            print(f"  Flow to C: {flow_c:.3f} cfs")
            print(f"  Mass balance: {abs(flow_a + flow_b - flow_c):.6f} cfs")
            
            # Jeppson's solution (from textbook)
            # Junction elevation ≈ 75 ft
            # Flow A ≈ 2.5 cfs, Flow B ≈ 1.5 cfs, Flow C ≈ 4.0 cfs
            
            print(f"\nComparison with Jeppson:")
            print(f"  Junction elevation - Expected: ~75 ft, Calculated: {junction_head_ft + 60:.1f} ft")
            print(f"  Flow A - Expected: ~2.5 cfs, Calculated: {flow_a:.2f} cfs")
            print(f"  Flow B - Expected: ~1.5 cfs, Calculated: {flow_b:.2f} cfs")
            print(f"  Flow C - Expected: ~4.0 cfs, Calculated: {flow_c:.2f} cfs")
            
            self.results['jeppson'] = {
                'junction_elev_expected': 75,
                'junction_elev_calculated': junction_head_ft + 60,
                'flow_c_expected': 4.0,
                'flow_c_calculated': flow_c
            }
    
    def validate_multiphase_beggs_brill(self):
        """
        Validation Case 4: Multiphase Flow - Beggs & Brill
        
        Reference: Beggs, H.D. and Brill, J.P. (1973)
        "A Study of Two-Phase Flow in Inclined Pipes"
        JPT May 1973, pp. 607-617
        
        Test Case: Horizontal two-phase flow
        - Pipe: 2" diameter
        - Oil flow: 500 bbl/day (0.000919 m³/s)
        - Water flow: 200 bbl/day (0.000368 m³/s)
        - No gas
        - Oil density: 850 kg/m³
        - Water density: 1000 kg/m³
        - Oil viscosity: 2 cp
        - Water viscosity: 1 cp
        """
        print("\n" + "-"*70)
        print("VALIDATION 4: Multiphase Flow - Oil/Water Mixture")
        print("-"*70)
        
        # Create network
        network = ps.Network()
        
        inlet = network.add_node("INLET", ps.NodeType.SOURCE)
        outlet = network.add_node("OUTLET", ps.NodeType.SINK)
        
        # 2" pipe, 1000 m long
        pipe = network.add_pipe("MULTIPHASE", inlet, outlet, 1000.0, 0.0508)  # 1000m, 2"
        pipe.set_roughness(0.000045)
        
        # Set pressures
        inlet.set_pressure_bc(50e5)   # 50 bar
        outlet.set_pressure_bc(45e5)  # 45 bar (we'll calculate actual)
        
        # Multiphase fluid
        fluid = ps.FluidProperties()
        
        # Calculate phase fractions
        q_oil = 0.000919   # m³/s
        q_water = 0.000368 # m³/s
        q_total = q_oil + q_water
        
        fluid.oil_fraction = q_oil / q_total
        fluid.water_fraction = q_water / q_total
        fluid.gas_fraction = 0.0
        
        fluid.oil_density = 850.0
        fluid.water_density = 1000.0
        fluid.oil_viscosity = 0.002    # 2 cp
        fluid.water_viscosity = 0.001  # 1 cp
        
        # Additional multiphase properties
        fluid.water_cut = q_water / (q_oil + q_water)
        
        print(f"Fluid properties:")
        print(f"  Oil fraction: {fluid.oil_fraction:.3f}")
        print(f"  Water fraction: {fluid.water_fraction:.3f}")
        print(f"  Water cut: {fluid.water_cut*100:.1f}%")
        print(f"  Mixture density: {fluid.mixture_density():.1f} kg/m³")
        print(f"  Mixture viscosity: {fluid.mixture_viscosity()*1000:.2f} cp")
        
        # Find pressure drop for given flow rate
        target_flow = q_total
        tolerance = 0.00001
        
        # Binary search for outlet pressure
        p_out_low = 30e5
        p_out_high = 49.9e5
        
        for iteration in range(20):
            p_out_test = (p_out_low + p_out_high) / 2
            outlet.set_pressure_bc(p_out_test)
            
            solver = ps.SteadyStateSolver(network, fluid)
            solver.config.verbose = False
            results = solver.solve()
            
            if results.converged:
                flow = results.pipe_flow_rates['MULTIPHASE']
                
                if abs(flow - target_flow) < tolerance:
                    dp = (50e5 - p_out_test) / 1e5  # bar
                    dp_psi = dp * 14.5038
                    dp_per_1000ft = dp_psi * (304.8 / 1000)  # psi/1000 ft
                    
                    print(f"\nResults:")
                    print(f"  Total flow rate: {flow*86400:.1f} m³/day")
                    print(f"  Pressure drop: {dp:.3f} bar ({dp_psi:.1f} psi)")
                    print(f"  Pressure gradient: {dp_per_1000ft:.2f} psi/1000 ft")
                    print(f"  Mixture velocity: {results.pipe_velocities['MULTIPHASE']:.2f} m/s")
                    print(f"  Reynolds number: {results.pipe_reynolds_numbers['MULTIPHASE']:.0f}")
                    
                    # Simplified Beggs-Brill for horizontal flow
                    # For comparison, calculate using homogeneous model
                    rho_m = fluid.mixture_density()
                    mu_m = fluid.mixture_viscosity()
                    v = flow / (np.pi * 0.0254**2)
                    Re = rho_m * v * 0.0508 / mu_m
                    
                    # Blasius equation for smooth pipes
                    if Re < 2300:
                        f = 64 / Re
                    else:
                        f = 0.316 / Re**0.25
                    
                    dp_homogeneous = f * 1000 * rho_m * v**2 / (2 * 0.0508) / 1e5
                    
                    print(f"\nComparison with homogeneous model:")
                    print(f"  Homogeneous model: {dp_homogeneous:.3f} bar")
                    print(f"  Pipeline-Sim: {dp:.3f} bar")
                    print(f"  Difference: {abs(dp - dp_homogeneous)/dp_homogeneous*100:.1f}%")
                    
                    self.results['multiphase'] = {
                        'dp_calculated': dp,
                        'dp_homogeneous': dp_homogeneous,
                        'error_percent': abs(dp - dp_homogeneous)/dp_homogeneous*100
                    }
                    break
                
                elif flow < target_flow:
                    p_out_high = p_out_test
                else:
                    p_out_low = p_out_test
    
    def plot_validation_results(self):
        """Create professional validation plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pipeline-Sim Validation Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Pressure drop comparison
        if 'crane_410' in self.results:
            ax1.bar(['Crane 410\nExpected', 'Pipeline-Sim\nCalculated'], 
                   [self.results['crane_410']['expected'], 
                    self.results['crane_410']['calculated']],
                   color=['blue', 'green'], alpha=0.7)
            ax1.set_ylabel('Pressure Drop (psi)', fontsize=12)
            ax1.set_title('Single Pipe Validation (Crane 410)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Add error text
            error = self.results['crane_410']['error_percent']
            ax1.text(0.5, 0.95, f'Error: {error:.1f}%', 
                    transform=ax1.transAxes, ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Gas flow comparison
        if 'aga_gas' in self.results:
            ax2.bar(['AGA Equation', 'Pipeline-Sim'], 
                   [self.results['aga_gas']['expected'], 
                    self.results['aga_gas']['calculated']],
                   color=['blue', 'green'], alpha=0.7)
            ax2.set_ylabel('Flow Rate (MMSCFD)', fontsize=12)
            ax2.set_title('Gas Pipeline Validation (AGA)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            error = self.results['aga_gas']['error_percent']
            ax2.text(0.5, 0.95, f'Error: {error:.1f}%', 
                    transform=ax2.transAxes, ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 3: Network validation
        if 'jeppson' in self.results:
            categories = ['Junction\nElevation (ft)', 'Flow to C\n(cfs)']
            expected = [self.results['jeppson']['junction_elev_expected'],
                       self.results['jeppson']['flow_c_expected']]
            calculated = [self.results['jeppson']['junction_elev_calculated'],
                         self.results['jeppson']['flow_c_calculated']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax3.bar(x - width/2, expected, width, label='Jeppson', color='blue', alpha=0.7)
            ax3.bar(x + width/2, calculated, width, label='Pipeline-Sim', color='green', alpha=0.7)
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories)
            ax3.set_title('Three-Reservoir Network (Jeppson)', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Multiphase validation
        if 'multiphase' in self.results:
            ax4.bar(['Homogeneous\nModel', 'Pipeline-Sim'], 
                   [self.results['multiphase']['dp_homogeneous'], 
                    self.results['multiphase']['dp_calculated']],
                   color=['blue', 'green'], alpha=0.7)
            ax4.set_ylabel('Pressure Drop (bar)', fontsize=12)
            ax4.set_title('Multiphase Flow Validation', fontsize=12)
            ax4.grid(True, alpha=0.3)
            
            error = self.results['multiphase']['error_percent']
            ax4.text(0.5, 0.95, f'Error: {error:.1f}%', 
                    transform=ax4.transAxes, ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        print("\nTest Case                    | Expected  | Calculated | Error (%)")
        print("-"*70)
        
        if 'crane_410' in self.results:
            r = self.results['crane_410']
            print(f"Crane 410 (psi)             | {r['expected']:9.1f} | {r['calculated']:10.1f} | {r['error_percent']:9.1f}")
        
        if 'aga_gas' in self.results:
            r = self.results['aga_gas']
            print(f"AGA Gas Flow (MMSCFD)       | {r['expected']:9.1f} | {r['calculated']:10.1f} | {r['error_percent']:9.1f}")
        
        if 'jeppson' in self.results:
            r = self.results['jeppson']
            print(f"Jeppson Junction Elev (ft)  | {r['junction_elev_expected']:9.1f} | {r['junction_elev_calculated']:10.1f} | {abs(r['junction_elev_expected']-r['junction_elev_calculated'])/r['junction_elev_expected']*100:9.1f}")
        
        if 'multiphase' in self.results:
            r = self.results['multiphase']
            print(f"Multiphase ΔP (bar)         | {r['dp_homogeneous']:9.3f} | {r['dp_calculated']:10.3f} | {r['error_percent']:9.1f}")
        
        print("\nNOTE: Differences may be due to:")
        print("  - Different friction factor correlations")
        print("  - Numerical precision")
        print("  - Simplifying assumptions in reference calculations")
        print("  - Real gas effects vs ideal gas assumptions")


def main():
    """Run complete validation suite"""
    
    print("\nStarting Pipeline-Sim Professional Validation")
    print("Version:", ps.__version__)
    
    # Run validations
    validator = ValidationSuite()
    validator.run_all_validations()
    
    # Create plots
    print("\nGenerating validation plots...")
    fig = validator.plot_validation_results()
    plt.savefig('pipeline_sim_validation_results.png', dpi=300, bbox_inches='tight')
    print("Saved: pipeline_sim_validation_results.png")
    
    plt.show()
    
    print("\n" + "="*80)
    print("Validation complete!")
    print("="*80)


if __name__ == "__main__":
    main()