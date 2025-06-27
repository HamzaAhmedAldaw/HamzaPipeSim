# AI_GENERATED: Utility functions for Pipeline-Sim
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, Optional


def load_network(filename: str):
    """Load network from JSON file"""
    from . import Network, NodeType, FluidProperties
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    network = Network()
    
    # Create nodes
    node_map = {}
    for node_data in data['nodes']:
        node_type = getattr(NodeType, node_data['type'])
        node = network.add_node(node_data['id'], node_type)
        
        if 'elevation' in node_data:
            node.elevation = node_data['elevation']
        
        if 'pressure' in node_data:
            network.set_pressure(node, node_data['pressure'])
        
        if 'flow_rate' in node_data:
            network.set_flow_rate(node, node_data['flow_rate'])
        
        node_map[node_data['id']] = node
    
    # Create pipes
    for pipe_data in data['pipes']:
        pipe = network.add_pipe(
            pipe_data['id'],
            node_map[pipe_data['upstream']],
            node_map[pipe_data['downstream']],
            pipe_data['length'],
            pipe_data['diameter']
        )
        
        if 'roughness' in pipe_data:
            pipe.roughness = pipe_data['roughness']
        
        if 'inclination' in pipe_data:
            pipe.inclination = pipe_data['inclination']
    
    # Load fluid properties if present
    fluid = None
    if 'fluid' in data:
        fluid = FluidProperties()
        fluid_data = data['fluid']
        
        for attr in ['oil_density', 'gas_density', 'water_density',
                    'oil_viscosity', 'gas_viscosity', 'water_viscosity',
                    'gas_oil_ratio', 'water_cut', 'api_gravity']:
            if attr in fluid_data:
                setattr(fluid, attr, fluid_data[attr])
        
        # Calculate phase fractions
        if 'water_cut' in fluid_data:
            fluid.water_fraction = fluid_data['water_cut']
            fluid.oil_fraction = 1.0 - fluid_data['water_cut']
        
        if 'gas_oil_ratio' in fluid_data:
            fluid.gas_fraction = fluid_data['gas_oil_ratio'] / 1000.0
    
    return network, fluid


def save_results(results, filename: str, format: str = 'csv'):
    """Save simulation results to file"""
    
    if format == 'csv':
        # Node results
        node_df = pd.DataFrame([
            {
                'Node ID': node_id,
                'Pressure (Pa)': pressure,
                'Pressure (bar)': pressure / 1e5,
                'Temperature (K)': results.node_temperatures.get(node_id, 288.15)
            }
            for node_id, pressure in results.node_pressures.items()
        ])
        
        # Pipe results
        pipe_df = pd.DataFrame([
            {
                'Pipe ID': pipe_id,
                'Flow Rate (m³/s)': flow,
                'Velocity (m/s)': flow / 0.07,  # Approximate
                'Pressure Drop (Pa)': results.pipe_pressure_drops.get(pipe_id, 0),
                'Pressure Drop (bar)': results.pipe_pressure_drops.get(pipe_id, 0) / 1e5
            }
            for pipe_id, flow in results.pipe_flow_rates.items()
        ])
        
        # Save to CSV
        base_name = filename.rsplit('.', 1)[0]
        node_df.to_csv(f"{base_name}_nodes.csv", index=False)
        pipe_df.to_csv(f"{base_name}_pipes.csv", index=False)
        
    elif format == 'json':
        data = {
            'converged': results.converged,
            'iterations': results.iterations,
            'residual': results.residual,
            'computation_time': results.computation_time,
            'nodes': {
                node_id: {
                    'pressure': pressure,
                    'temperature': results.node_temperatures.get(node_id, 288.15)
                }
                for node_id, pressure in results.node_pressures.items()
            },
            'pipes': {
                pipe_id: {
                    'flow_rate': flow,
                    'pressure_drop': results.pipe_pressure_drops.get(pipe_id, 0)
                }
                for pipe_id, flow in results.pipe_flow_rates.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


def plot_network(network, results=None, figsize=(12, 8)):
    """Visualize pipeline network with optional results"""
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    pos = {}
    for i, (node_id, node) in enumerate(network.nodes.items()):
        G.add_node(node_id, type=node.type)
        # Simple layout - can be improved
        x = i % 5
        y = node.elevation / 100 if node.elevation > 0 else i // 5
        pos[node_id] = (x, y)
    
    # Add edges (pipes)
    for pipe_id, pipe in network.pipes.items():
        G.add_edge(pipe.upstream.id, pipe.downstream.id, 
                  pipe_id=pipe_id, length=pipe.length)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw network
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue',
            node_size=1000, font_size=10, font_weight='bold',
            arrows=True, edge_color='gray', width=2)
    
    # Add pressure labels if results available
    if results:
        pressure_labels = {
            node_id: f"{results.node_pressures.get(node_id, 0)/1e5:.1f} bar"
            for node_id in G.nodes()
        }
        nx.draw_networkx_labels(G, pos, pressure_labels, 
                               font_size=8, verticalalignment='bottom')
    
    ax.set_title("Pipeline Network Topology", fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    return fig
