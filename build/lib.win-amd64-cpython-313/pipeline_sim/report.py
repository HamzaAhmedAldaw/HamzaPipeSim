# ===== python/pipeline_sim/report.py =====
# AI_GENERATED: Report generation utilities
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any
import io
import base64


def generate_report(network, results, fluid, filename='report.html'):
    """Generate comprehensive simulation report"""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pipeline-Sim Simulation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
            .warning {{ color: #ff6b6b; font-weight: bold; }}
            .success {{ color: #51cf66; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Pipeline Simulation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Simulation Summary</h2>
            <p>Status: <span class="{'success' if results.converged else 'warning'}">
                {'Converged' if results.converged else 'Not Converged'}
            </span></p>
            <p>Iterations: {results.iterations}</p>
            <p>Final Residual: {results.residual:.2e}</p>
            <p>Computation Time: {results.computation_time:.3f} seconds</p>
        </div>
        
        <h2>Network Configuration</h2>
        <p>Nodes: {len(network.nodes)}</p>
        <p>Pipes: {len(network.pipes)}</p>
        
        <h2>Fluid Properties</h2>
        <table>
            <tr><th>Property</th><th>Value</th><th>Unit</th></tr>
            <tr><td>Oil Density</td><td>{fluid.oil_density:.1f}</td><td>kg/m³</td></tr>
            <tr><td>Gas Density (rel)</td><td>{fluid.gas_density:.3f}</td><td>-</td></tr>
            <tr><td>Water Density</td><td>{fluid.water_density:.1f}</td><td>kg/m³</td></tr>
            <tr><td>Oil Viscosity</td><td>{fluid.oil_viscosity*1000:.1f}</td><td>cP</td></tr>
            <tr><td>GOR</td><td>{fluid.gas_oil_ratio:.1f}</td><td>sm³/sm³</td></tr>
            <tr><td>Water Cut</td><td>{fluid.water_cut*100:.1f}</td><td>%</td></tr>
        </table>
        
        <h2>Node Results</h2>
        <table>
            <tr><th>Node ID</th><th>Type</th><th>Pressure (bar)</th><th>Temperature (K)</th></tr>
    """
    
    for node_id, node in network.nodes.items():
        pressure_bar = results.node_pressures.get(node_id, 0) / 1e5
        temp = results.node_temperatures.get(node_id, 288.15)
        html += f"""
            <tr>
                <td>{node_id}</td>
                <td>{node.type}</td>
                <td>{pressure_bar:.2f}</td>
                <td>{temp:.1f}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Pipe Results</h2>
        <table>
            <tr><th>Pipe ID</th><th>Length (m)</th><th>Diameter (m)</th>
                <th>Flow Rate (m³/s)</th><th>Velocity (m/s)</th><th>Pressure Drop (bar)</th></tr>
    """
    
    for pipe_id, pipe in network.pipes.items():
        flow = results.pipe_flow_rates.get(pipe_id, 0)
        velocity = flow / pipe.area()
        dp_bar = results.pipe_pressure_drops.get(pipe_id, 0) / 1e5
        
        html += f"""
            <tr>
                <td>{pipe_id}</td>
                <td>{pipe.length:.1f}</td>
                <td>{pipe.diameter:.3f}</td>
                <td>{flow:.4f}</td>
                <td>{velocity:.2f}</td>
                <td>{dp_bar:.3f}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Warnings and Recommendations</h2>
        <ul>
    """
    
    # Add warnings based on results
    for pipe_id, velocity in [(p_id, results.pipe_flow_rates.get(p_id, 0) / network.pipes[p_id].area()) 
                              for p_id in network.pipes]:
        if velocity > 10:
            html += f'<li class="warning">High velocity in {pipe_id}: {velocity:.1f} m/s</li>'
    
    if not results.converged:
        html += '<li class="warning">Simulation did not converge - check boundary conditions</li>'
    
    html += """
        </ul>
    </body>
    </html>
    """
    
    with open(filename, 'w') as f:
        f.write(html)
    
    print(f"Report generated: {filename}")
