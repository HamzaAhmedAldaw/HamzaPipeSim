#!/usr/bin/env python3
"""
Professional Oil Field Development Network Analysis
Simulates a complete offshore field development with:
- Multiple wells at different pressures
- Subsea manifolds
- Production separators
- Export and injection systems
"""

import sys
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Wedge
from matplotlib.patches import FancyArrowPatch, ArrowStyle
import matplotlib.patches as mpatches
from datetime import datetime

# Load Pipeline-Sim
print("Loading Pipeline-Sim...")
pyd_path = r"C:\Users\KIMO STORE\AppData\Roaming\Python\Python313\site-packages\pipeline_sim.cp313-win_amd64.pyd"
spec = importlib.util.spec_from_file_location("pipeline_sim", pyd_path)
pipeline_sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_sim)

# Import classes
Network = pipeline_sim.Network
NodeType = pipeline_sim.NodeType
FluidProperties = pipeline_sim.FluidProperties
SteadyStateSolver = pipeline_sim.SteadyStateSolver
constants = pipeline_sim.constants

print("✅ Pipeline-Sim loaded successfully")

class OilFieldDevelopment:
    """Professional oil field development simulation"""
    
    def __init__(self):
        self.network = Network()
        self.nodes = {}
        self.pipes = {}
        self.well_data = {}
        self.results = None
        
    def create_field_layout(self):
        """Create a realistic offshore field layout"""
        
        print("\n=== Creating Offshore Field Development ===")
        
        # Production Wells (different reservoir pressures and productivities)
        wells = [
            # (name, x_km, y_km, reservoir_pressure_bar, productivity_index)
            ("WELL-A1", -3, 2, 320, 5.0),
            ("WELL-A2", -3, 0, 310, 4.5),
            ("WELL-A3", -3, -2, 300, 4.0),
            ("WELL-B1", -1, 3, 330, 6.0),
            ("WELL-B2", -1, 1, 325, 5.5),
            ("WELL-B3", -1, -1, 315, 5.0),
            ("WELL-B4", -1, -3, 305, 4.5),
        ]
        
        # Water injection wells
        injection_wells = [
            ("INJ-1", -4, 0, 350, 0),
            ("INJ-2", 0, 0, 350, 0),
        ]
        
        # Create production wells
        for name, x, y, res_p, pi in wells:
            node = self.network.add_node(name, NodeType.SOURCE)
            self.nodes[name] = node
            self.well_data[name] = {
                'x': x, 'y': y, 
                'reservoir_pressure': res_p,
                'productivity_index': pi,
                'type': 'producer'
            }
            # Set wellhead pressure (will be calculated based on reservoir pressure and flow)
            self.network.set_pressure(node, res_p * 1e5)  # Convert to Pa
        
        # Create injection wells
        for name, x, y, inj_p, _ in injection_wells:
            node = self.network.add_node(name, NodeType.SINK)
            self.nodes[name] = node
            self.well_data[name] = {
                'x': x, 'y': y,
                'injection_pressure': inj_p,
                'type': 'injector'
            }
            self.network.set_pressure(node, inj_p * 1e5)
        
        # Subsea manifolds
        manifolds = [
            ("MANIFOLD-A", -2, 0),
            ("MANIFOLD-B", 0, 1),
        ]
        
        for name, x, y in manifolds:
            node = self.network.add_node(name, NodeType.JUNCTION)
            self.nodes[name] = node
            self.well_data[name] = {'x': x, 'y': y, 'type': 'manifold'}
        
        # Platform facilities
        platform_nodes = [
            ("PLATFORM", 3, 0),
            ("HP-SEP", 3.5, 0.5),    # High pressure separator
            ("LP-SEP", 3.5, -0.5),   # Low pressure separator
            ("EXPORT", 4.5, 0.5),     # Export pump
            ("GAS-COMP", 4.5, -0.5),  # Gas compressor
        ]
        
        for i, (name, x, y) in enumerate(platform_nodes):
            if name == "EXPORT":
                node_type = NodeType.SINK
            elif name == "PLATFORM":
                node_type = NodeType.JUNCTION
            else:
                node_type = NodeType.JUNCTION
            
            node = self.network.add_node(name, node_type)
            self.nodes[name] = node
            self.well_data[name] = {'x': x, 'y': y, 'type': 'platform'}
        
        # Create pipelines
        self.create_pipelines()
        
        # Set remaining boundary conditions
        self.network.set_pressure(self.nodes["EXPORT"], 150e5)  # 150 bar export pressure
        
        print(f"\nField Development Created:")
        print(f"  Production wells: {len([w for w in wells])}")
        print(f"  Injection wells: {len(injection_wells)}")
        print(f"  Manifolds: {len(manifolds)}")
        print(f"  Total nodes: {len(self.nodes)}")
        print(f"  Total pipelines: {len(self.pipes)}")
        
    def create_pipelines(self):
        """Create pipeline connections"""
        
        # Well to manifold connections
        connections = [
            # A-wells to Manifold-A
            ("WELL-A1", "MANIFOLD-A", 2.5, 0.25, "flowline"),
            ("WELL-A2", "MANIFOLD-A", 2.0, 0.25, "flowline"),
            ("WELL-A3", "MANIFOLD-A", 2.5, 0.25, "flowline"),
            
            # B-wells to Manifold-B
            ("WELL-B1", "MANIFOLD-B", 3.2, 0.25, "flowline"),
            ("WELL-B2", "MANIFOLD-B", 1.4, 0.25, "flowline"),
            ("WELL-B3", "MANIFOLD-B", 2.2, 0.25, "flowline"),
            ("WELL-B4", "MANIFOLD-B", 4.2, 0.25, "flowline"),
            
            # Manifolds to Platform
            ("MANIFOLD-A", "PLATFORM", 5.0, 0.4, "trunk"),
            ("MANIFOLD-B", "PLATFORM", 3.2, 0.4, "trunk"),
            
            # Platform internal
            ("PLATFORM", "HP-SEP", 0.5, 0.5, "platform"),
            ("HP-SEP", "LP-SEP", 1.0, 0.4, "platform"),
            ("LP-SEP", "EXPORT", 1.5, 0.4, "platform"),
            
            # Water injection lines
            ("GAS-COMP", "INJ-1", 6.0, 0.3, "injection"),
            ("GAS-COMP", "INJ-2", 4.0, 0.3, "injection"),
        ]
        
        for from_node, to_node, length_km, diameter, pipe_type in connections:
            pipe_name = f"{from_node}_to_{to_node}"
            pipe = self.network.add_pipe(pipe_name, 
                                       self.nodes[from_node], 
                                       self.nodes[to_node],
                                       length_km * 1000, 
                                       diameter)
            self.pipes[pipe_name] = {
                'pipe': pipe,
                'type': pipe_type,
                'length_km': length_km,
                'diameter': diameter
            }
    
    def create_fluid_properties(self):
        """Create realistic fluid properties for different streams"""
        
        # Main production fluid (live oil with dissolved gas)
        production_fluid = FluidProperties()
        production_fluid.oil_fraction = 0.75
        production_fluid.water_fraction = 0.20
        production_fluid.gas_fraction = 0.05
        production_fluid.oil_density = 780.0    # Light oil with dissolved gas
        production_fluid.water_density = 1025.0  # Seawater
        production_fluid.gas_density = 15.0      # At separator conditions
        production_fluid.oil_viscosity = 0.0008  # Low viscosity due to dissolved gas
        production_fluid.water_viscosity = 0.0011
        production_fluid.gas_viscosity = 0.000018
        
        # Calculate GOR and water cut
        production_fluid.gas_oil_ratio = 150.0  # sm³/sm³
        production_fluid.water_cut = 0.20
        
        return production_fluid
    
    def solve_network(self):
        """Solve the field network"""
        
        fluid = self.create_fluid_properties()
        
        print(f"\n=== Fluid Properties ===")
        print(f"  Oil fraction: {fluid.oil_fraction:.0%}")
        print(f"  Water cut: {fluid.water_cut:.0%}")
        print(f"  GOR: {fluid.gas_oil_ratio} sm³/sm³")
        print(f"  Mixture density: {fluid.mixture_density():.1f} kg/m³")
        print(f"  Mixture viscosity: {fluid.mixture_viscosity()*1000:.2f} cP")
        
        # Create solver
        solver = SteadyStateSolver(self.network, fluid)
        config = solver.config
        config.verbose = True
        config.tolerance = 1e-6
        config.max_iterations = 100
        solver.set_config(config)
        
        print("\n=== Solving Field Network ===")
        self.results = solver.solve()
        
        return self.results
    
    def create_field_visualization(self):
        """Create professional field layout visualization"""
        
        if not self.results or not self.results.converged:
            print("Cannot visualize - no converged solution")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Main field layout
        self._draw_field_layout(ax1)
        
        # Production allocation chart
        self._draw_production_allocation(ax2)
        
        plt.suptitle('Offshore Field Development Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _draw_field_layout(self, ax):
        """Draw the field layout with flows"""
        
        ax.set_aspect('equal')
        ax.set_xlim(-5, 6)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)
        
        # Draw seafloor
        ax.fill_between([-5, 6], [-4, -4], [-3.8, -3.8], color='brown', alpha=0.3)
        ax.text(-4.5, -3.9, 'Seafloor', fontsize=10, style='italic')
        
        # Draw pipelines first (so they appear under nodes)
        for pipe_name, pipe_data in self.pipes.items():
            from_node = pipe_name.split('_to_')[0]
            to_node = pipe_name.split('_to_')[1]
            
            if from_node not in self.well_data or to_node not in self.well_data:
                continue
                
            x1, y1 = self.well_data[from_node]['x'], self.well_data[from_node]['y']
            x2, y2 = self.well_data[to_node]['x'], self.well_data[to_node]['y']
            
            # Get flow data
            flow = self.results.pipe_flow_rates.get(pipe_name, 0)
            velocity = self.results.pipe_velocities.get(pipe_name, 0)
            
            # Determine pipe appearance
            if pipe_data['type'] == 'flowline':
                color = 'blue'
                style = '-'
                width = 2
            elif pipe_data['type'] == 'trunk':
                color = 'darkblue'
                style = '-'
                width = 4
            elif pipe_data['type'] == 'injection':
                color = 'red'
                style = '--'
                width = 2
            else:
                color = 'gray'
                style = '-'
                width = 2
            
            # Draw pipe
            ax.plot([x1, x2], [y1, y2], style, color=color, 
                   linewidth=width, alpha=0.6)
            
            # Add flow arrow
            if abs(flow) > 0.001:
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                dx, dy = dx/length, dy/length
                
                arrow = FancyArrowPatch((mid_x - 0.2*dx, mid_y - 0.2*dy),
                                      (mid_x + 0.2*dx, mid_y + 0.2*dy),
                                      arrowstyle='->', mutation_scale=20,
                                      color=color, linewidth=2)
                ax.add_patch(arrow)
                
                # Flow label
                flow_m3h = abs(flow) * 3600
                ax.text(mid_x, mid_y + 0.1, f'{flow_m3h:.0f} m³/h',
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle="round,pad=0.2", 
                               facecolor="white", alpha=0.8))
        
        # Draw nodes
        for node_name, node_data in self.well_data.items():
            x, y = node_data['x'], node_data['y']
            
            if node_data['type'] == 'producer':
                # Production well
                circle = Circle((x, y), 0.15, color='green', zorder=5)
                ax.add_patch(circle)
                ax.plot(x, y, 'ko', markersize=6, zorder=6)
                
                # Add well info
                pressure = self.results.node_pressures.get(node_name, 0) / 1e5
                ax.text(x, y - 0.3, f'{node_name}\n{pressure:.0f} bar',
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle="round,pad=0.2", 
                               facecolor="lightgreen", alpha=0.8))
                
            elif node_data['type'] == 'injector':
                # Injection well
                triangle = plt.Polygon([(x-0.15, y-0.15), (x+0.15, y-0.15), 
                                      (x, y+0.15)], color='red', zorder=5)
                ax.add_patch(triangle)
                
                ax.text(x, y - 0.3, f'{node_name}',
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle="round,pad=0.2", 
                               facecolor="lightcoral", alpha=0.8))
                
            elif node_data['type'] == 'manifold':
                # Subsea manifold
                rect = Rectangle((x-0.2, y-0.15), 0.4, 0.3, 
                               color='yellow', zorder=5)
                ax.add_patch(rect)
                
                pressure = self.results.node_pressures.get(node_name, 0) / 1e5
                ax.text(x, y + 0.3, f'{node_name}\n{pressure:.0f} bar',
                       fontsize=9, ha='center', fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor="yellow", alpha=0.9))
                
            elif node_data['type'] == 'platform':
                # Platform facilities
                if node_name == 'PLATFORM':
                    # Main platform
                    rect = FancyBboxPatch((x-0.3, y-0.3), 0.6, 0.6,
                                        boxstyle="round,pad=0.05",
                                        facecolor='gray', edgecolor='black',
                                        linewidth=2, zorder=5)
                    ax.add_patch(rect)
                    ax.text(x, y, 'FPSO', fontsize=10, ha='center',
                           va='center', fontweight='bold')
                else:
                    # Other facilities
                    circle = Circle((x, y), 0.15, color='lightblue', 
                                  edgecolor='blue', linewidth=2, zorder=5)
                    ax.add_patch(circle)
                    ax.text(x, y - 0.3, node_name, fontsize=8, ha='center')
        
        # Add title and labels
        ax.set_xlabel('Distance East-West (km)', fontsize=12)
        ax.set_ylabel('Distance North-South (km)', fontsize=12)
        ax.set_title('Field Layout and Flow Distribution', fontsize=14, fontweight='bold')
        
        # Add legend
        legend_elements = [
            mpatches.Circle((0, 0), 0.1, color='green', label='Production Well'),
            mpatches.Polygon([(0, 0), (0.2, 0), (0.1, 0.2)], color='red', label='Injection Well'),
            mpatches.Rectangle((0, 0), 0.2, 0.1, color='yellow', label='Manifold'),
            mpatches.Rectangle((0, 0), 0.2, 0.1, color='gray', label='Platform'),
            mpatches.Line2D([0], [0], color='blue', linewidth=2, label='Flowline'),
            mpatches.Line2D([0], [0], color='darkblue', linewidth=4, label='Trunk Line'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    def _draw_production_allocation(self, ax):
        """Draw production allocation and analysis"""
        
        # Calculate well production rates
        well_production = {}
        total_production = 0
        
        for well_name in self.well_data:
            if self.well_data[well_name]['type'] == 'producer':
                # Sum outgoing flows from well
                production = 0
                for pipe_name, flow in self.results.pipe_flow_rates.items():
                    if pipe_name.startswith(well_name + '_to_'):
                        production += abs(flow)
                
                well_production[well_name] = production * 3600  # Convert to m³/h
                total_production += production * 3600
        
        # Sort wells by production
        sorted_wells = sorted(well_production.items(), key=lambda x: x[1], reverse=True)
        
        # Create bar chart
        wells = [w[0] for w in sorted_wells]
        production = [w[1] for w in sorted_wells]
        
        bars = ax.bar(wells, production, color='green', alpha=0.7, edgecolor='darkgreen')
        
        # Add value labels
        for bar, prod in zip(bars, production):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                   f'{prod:.0f}', ha='center', va='bottom', fontsize=10)
        
        # Add cumulative line
        cumulative = np.cumsum(production)
        cumulative_pct = cumulative / total_production * 100
        
        ax2 = ax.twinx()
        ax2.plot(wells, cumulative_pct, 'r-o', linewidth=2, markersize=8)
        ax2.set_ylabel('Cumulative Production (%)', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 105)
        
        # Add 80/20 reference line
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5)
        ax2.text(len(wells)-1, 82, '80% line', fontsize=10, color='red')
        
        ax.set_xlabel('Well Name', fontsize=12)
        ax.set_ylabel('Production Rate (m³/h)', fontsize=12)
        ax.set_title('Well Production Allocation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add summary box
        summary_text = f"""Field Production Summary:
Total Production: {total_production:.0f} m³/h ({total_production*24:.0f} m³/day)
Number of Wells: {len(wells)}
Average per Well: {total_production/len(wells):.0f} m³/h
Best Well: {wells[0]} ({production[0]:.0f} m³/h)
Poorest Well: {wells[-1]} ({production[-1]:.0f} m³/h)"""
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
               fontsize=11, va='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    def create_pressure_network_graph(self):
        """Create pressure distribution network graph"""
        
        if not self.results or not self.results.converged:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use networkx-style layout
        pos = {}
        for node_name, node_data in self.well_data.items():
            pos[node_name] = (node_data['x'], node_data['y'])
        
        # Color nodes by pressure
        pressures = []
        for node_name in self.nodes:
            p = self.results.node_pressures.get(node_name, 0) / 1e5
            pressures.append(p)
        
        # Create colormap
        vmin, vmax = min(pressures), max(pressures)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.coolwarm
        
        # Draw edges with thickness based on flow
        for pipe_name in self.pipes:
            from_node = pipe_name.split('_to_')[0]
            to_node = pipe_name.split('_to_')[1]
            
            if from_node in pos and to_node in pos:
                x1, y1 = pos[from_node]
                x2, y2 = pos[to_node]
                
                flow = abs(self.results.pipe_flow_rates.get(pipe_name, 0))
                width = 1 + flow * 5000  # Scale width by flow
                
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=width, alpha=0.3)
        
        # Draw nodes
        for node_name, (x, y) in pos.items():
            pressure = self.results.node_pressures.get(node_name, 0) / 1e5
            color = cmap(norm(pressure))
            
            ax.scatter(x, y, s=500, c=[color], edgecolors='black', 
                      linewidth=2, zorder=3)
            
            ax.text(x, y - 0.4, f'{node_name}\n{pressure:.0f} bar',
                   fontsize=9, ha='center')
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Pressure (bar)', fontsize=12)
        
        ax.set_xlim(-5.5, 6.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect('equal')
        ax.set_xlabel('Distance East-West (km)', fontsize=12)
        ax.set_ylabel('Distance North-South (km)', fontsize=12)
        ax.set_title('Network Pressure Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def generate_production_report(self):
        """Generate a professional production report"""
        
        if not self.results or not self.results.converged:
            print("No results to report")
            return
        
        print("\n" + "="*70)
        print("OFFSHORE FIELD PRODUCTION REPORT")
        print("="*70)
        print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*70)
        
        # Field summary
        total_oil = 0
        total_water = 0
        total_gas = 0
        
        for pipe_name, flow in self.results.pipe_flow_rates.items():
            if pipe_name.startswith("LP-SEP_to_EXPORT"):
                total_oil = abs(flow) * 0.75 * 3600  # m³/h
                total_water = abs(flow) * 0.20 * 3600
                total_gas = abs(flow) * 0.05 * 3600
        
        print("\nFIELD PRODUCTION SUMMARY:")
        print(f"  Total Liquid: {(total_oil + total_water):.0f} m³/h ({(total_oil + total_water)*24:.0f} m³/day)")
        print(f"  Oil Production: {total_oil:.0f} m³/h ({total_oil*24:.0f} m³/day)")
        print(f"  Water Production: {total_water:.0f} m³/h ({total_water*24:.0f} m³/day)")
        print(f"  Water Cut: {total_water/(total_oil + total_water)*100:.1f}%")
        print(f"  Gas Production: {total_gas*1000:.0f} sm³/h")
        
        print("\nWELL PERFORMANCE:")
        print(f"{'Well':<10} {'Pressure (bar)':<15} {'Flow (m³/h)':<15} {'Status':<10}")
        print("-"*50)
        
        for well_name in sorted(self.well_data.keys()):
            if self.well_data[well_name]['type'] == 'producer':
                pressure = self.results.node_pressures.get(well_name, 0) / 1e5
                
                # Get well flow
                flow = 0
                for pipe_name, pipe_flow in self.results.pipe_flow_rates.items():
                    if pipe_name.startswith(well_name + '_to_'):
                        flow = abs(pipe_flow) * 3600
                
                status = "FLOWING" if flow > 10 else "LOW"
                print(f"{well_name:<10} {pressure:<15.0f} {flow:<15.0f} {status:<10}")
        
        print("\nFACILITY PRESSURES:")
        facility_nodes = ['MANIFOLD-A', 'MANIFOLD-B', 'PLATFORM', 'HP-SEP', 'LP-SEP']
        for node in facility_nodes:
            if node in self.results.node_pressures:
                pressure = self.results.node_pressures[node] / 1e5
                print(f"  {node}: {pressure:.1f} bar")
        
        print("\nPIPELINE HYDRAULICS:")
        print(f"{'Pipeline':<25} {'Flow (m³/h)':<15} {'Velocity (m/s)':<15} {'ΔP (bar)':<10}")
        print("-"*65)
        
        key_pipes = [p for p in self.pipes if 'MANIFOLD' in p or 'PLATFORM' in p]
        for pipe_name in sorted(key_pipes):
            if pipe_name in self.results.pipe_flow_rates:
                flow = self.results.pipe_flow_rates[pipe_name] * 3600
                velocity = self.results.pipe_velocities.get(pipe_name, 0)
                dp = self.results.pipe_pressure_drops.get(pipe_name, 0) / 1e5
                print(f"{pipe_name:<25} {abs(flow):<15.0f} {velocity:<15.1f} {abs(dp):<10.1f}")
        
        print("\n" + "="*70)

def main():
    """Run the professional field development example"""
    
    print("\n" + "="*70)
    print("PROFESSIONAL OIL FIELD DEVELOPMENT NETWORK ANALYSIS")
    print("Pipeline-Sim - Advanced Petroleum Engineering")
    print("="*70)
    
    # Create field development
    field = OilFieldDevelopment()
    field.create_field_layout()
    
    # Solve network
    results = field.solve_network()
    
    if results and results.converged:
        print(f"\n✅ Solution converged in {results.iterations} iterations")
        print(f"   Max mass imbalance: {results.max_mass_imbalance:.2e} kg/s")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # Main field visualization
        fig1 = field.create_field_visualization()
        fig1.savefig('field_development_layout.png', dpi=300, bbox_inches='tight')
        
        # Pressure network graph
        fig2 = field.create_pressure_network_graph()
        fig2.savefig('field_pressure_network.png', dpi=300, bbox_inches='tight')
        
        print("✅ Visualizations saved:")
        print("   - field_development_layout.png")
        print("   - field_pressure_network.png")
        
        # Generate report
        field.generate_production_report()
        
        # Show plots
        plt.show()
        
    else:
        print("\n❌ Solution did not converge")
        print("   Check boundary conditions and network connectivity")
    
    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)

if __name__ == "__main__":
    main()