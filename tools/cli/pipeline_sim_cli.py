# AI_GENERATED: CLI tools and testing framework
# Generated on: 2025-06-27

# ===== tools/cli/pipeline_sim_cli.py =====
#!/usr/bin/env python3
"""
Pipeline-Sim Command Line Interface

AI_GENERATED: CLI tool for pipeline simulation
"""

import click
import json
import yaml
import sys
import os
from pathlib import Path
from tabulate import tabulate
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pipeline_sim as ps


@click.group()
@click.version_option(version=ps.__version__)
def cli():
    """Pipeline-Sim: Next-generation petroleum pipeline simulation"""
    pass


@cli.command()
@click.argument('network_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='results.csv', 
              help='Output file for results')
@click.option('--solver', '-s', default='steady-state',
              type=click.Choice(['steady-state', 'transient']),
              help='Solver type')
@click.option('--correlation', '-c', default='Beggs-Brill',
              help='Flow correlation to use')
@click.option('--tolerance', '-t', default=1e-6, type=float,
              help='Convergence tolerance')
@click.option('--max-iterations', '-i', default=1000, type=int,
              help='Maximum iterations')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
@click.option('--plot', '-p', is_flag=True,
              help='Generate plots')
@click.option('--report', '-r', is_flag=True,
              help='Generate HTML report')
def simulate(network_file, output, solver, correlation, 
            tolerance, max_iterations, verbose, plot, report):
    """Run pipeline simulation"""
    
    click.echo(f"Loading network from {network_file}...")
    
    try:
        # Load network
        network, fluid = ps.load_network(network_file)
        
        if verbose:
            click.echo(f"Network loaded: {len(network.nodes)} nodes, "
                      f"{len(network.pipes)} pipes")
            click.echo(f"Fluid: GOR={fluid.gas_oil_ratio}, "
                      f"Water cut={fluid.water_cut}")
        
        # Create solver
        if solver == 'steady-state':
            solver_obj = ps.SteadyStateSolver(network, fluid)
        else:
            solver_obj = ps.TransientSolver(network, fluid)
            # TODO: Configure transient parameters
        
        solver_obj.config.tolerance = tolerance
        solver_obj.config.max_iterations = max_iterations
        solver_obj.config.verbose = verbose
        
        # TODO: Set correlation
        
        # Run simulation
        click.echo("Running simulation...")
        results = solver_obj.solve()
        
        if results.converged:
            click.secho("✓ Simulation converged!", fg='green')
            click.echo(f"  Iterations: {results.iterations}")
            click.echo(f"  Residual: {results.residual:.2e}")
            click.echo(f"  Time: {results.computation_time:.2f} s")
        else:
            click.secho("✗ Simulation failed to converge!", fg='red')
            sys.exit(1)
        
        # Save results
        ps.save_results(results, output)
        click.echo(f"Results saved to {output}")
        
        # Generate plots if requested
        if plot:
            generate_plots(network, results)
        
        # Generate report if requested
        if report:
            report_file = output.replace('.csv', '.html')
            ps.generate_report(network, results, fluid, report_file)
            click.echo(f"Report saved to {report_file}")
            
    except Exception as e:
        click.secho(f"Error: {e}", fg='red')
        sys.exit(1)


@cli.command()
@click.argument('network_file', type=click.Path(exists=True))
def validate(network_file):
    """Validate network configuration"""
    
    click.echo(f"Validating {network_file}...")
    
    try:
        # Load and validate
        network, fluid = ps.load_network(network_file)
        
        errors = []
        warnings = []
        
        # Check connectivity
        orphaned_nodes = []
        for node_id, node in network.nodes.items():
            upstream = network.get_upstream_pipes(node)
            downstream = network.get_downstream_pipes(node)
            
            if not upstream and not downstream:
                orphaned_nodes.append(node_id)
            
            if node.type == ps.NodeType.SOURCE and upstream:
                errors.append(f"Source node '{node_id}' has upstream pipes")
            
            if node.type == ps.NodeType.SINK and downstream:
                errors.append(f"Sink node '{node_id}' has downstream pipes")
        
        if orphaned_nodes:
            warnings.append(f"Orphaned nodes: {', '.join(orphaned_nodes)}")
        
        # Check boundary conditions
        pressure_specs = network.pressure_specs
        flow_specs = network.flow_specs
        
        if not pressure_specs:
            errors.append("No pressure boundary conditions specified")
        
        if not flow_specs and len(pressure_specs) == len(network.nodes):
            warnings.append("All nodes have pressure specified - "
                          "system may be over-constrained")
        
        # Check pipe properties
        for pipe_id, pipe in network.pipes.items():
            if pipe.diameter <= 0:
                errors.append(f"Pipe '{pipe_id}' has invalid diameter")
            
            if pipe.length <= 0:
                errors.append(f"Pipe '{pipe_id}' has invalid length")
            
            if pipe.roughness < 0:
                errors.append(f"Pipe '{pipe_id}' has negative roughness")
        
        # Check fluid properties
        if fluid.oil_density <= 0:
            errors.append("Invalid oil density")
        
        if fluid.oil_fraction + fluid.gas_fraction + fluid.water_fraction > 1.01:
            errors.append("Phase fractions sum to more than 1")
        
        # Display results
        if errors:
            click.secho(f"\n✗ Found {len(errors)} errors:", fg='red')
            for error in errors:
                click.echo(f"  • {error}")
        else:
            click.secho("\n✓ No errors found", fg='green')
        
        if warnings:
            click.secho(f"\n⚠ Found {len(warnings)} warnings:", fg='yellow')
            for warning in warnings:
                click.echo(f"  • {warning}")
        
        # Summary
        click.echo(f"\nNetwork summary:")
        click.echo(f"  Nodes: {len(network.nodes)}")
        click.echo(f"  Pipes: {len(network.pipes)}")
        click.echo(f"  Pressure BCs: {len(pressure_specs)}")
        click.echo(f"  Flow BCs: {len(flow_specs)}")
        
        return 0 if not errors else 1
        
    except Exception as e:
        click.secho(f"Error loading network: {e}", fg='red')
        return 1


@cli.command()
@click.argument('results_file', type=click.Path(exists=True))
@click.option('--format', '-f', default='table',
              type=click.Choice(['table', 'json', 'summary']),
              help='Output format')
def analyze(results_file, format):
    """Analyze simulation results"""
    
    click.echo(f"Analyzing {results_file}...")
    
    try:
        # Load results
        import pandas as pd
        
        if results_file.endswith('.csv'):
            # Assume node results
            df = pd.read_csv(results_file)
            
            if format == 'table':
                click.echo("\nNode Results:")
                click.echo(tabulate(df, headers='keys', tablefmt='grid'))
                
            elif format == 'summary':
                click.echo("\nResults Summary:")
                click.echo(f"  Nodes: {len(df)}")
                click.echo(f"  Pressure range: {df['Pressure (bar)'].min():.1f} - "
                          f"{df['Pressure (bar)'].max():.1f} bar")
                
                # Look for pipe results
                pipe_file = results_file.replace('_nodes.csv', '_pipes.csv')
                if os.path.exists(pipe_file):
                    pipe_df = pd.read_csv(pipe_file)
                    click.echo(f"  Pipes: {len(pipe_df)}")
                    click.echo(f"  Total flow: {pipe_df['Flow Rate (m³/s)'].sum():.3f} m³/s")
                    click.echo(f"  Max velocity: {pipe_df['Velocity (m/s)'].max():.1f} m/s")
                    
        elif results_file.endswith('.json'):
            with open(results_file) as f:
                data = json.load(f)
            
            if format == 'json':
                click.echo(json.dumps(data, indent=2))
            else:
                click.echo(f"Converged: {data.get('converged', False)}")
                click.echo(f"Iterations: {data.get('iterations', 0)}")
                
    except Exception as e:
        click.secho(f"Error: {e}", fg='red')
        return 1


@cli.command()
def correlations():
    """List available flow correlations"""
    
    click.echo("Available flow correlations:")
    click.echo()
    
    correlations_info = [
        ("Beggs-Brill", "Multiphase", "All inclinations", "Most general"),
        ("Hagedorn-Brown", "Multiphase", "Vertical wells", "Good for vertical flow"),
        ("Gray", "Wet gas", "All inclinations", "High gas fraction"),
        ("Mechanistic", "Multiphase", "All inclinations", "Physics-based"),
    ]
    
    headers = ["Correlation", "Flow Type", "Application", "Notes"]
    click.echo(tabulate(correlations_info, headers=headers, tablefmt='grid'))


@cli.command()
@click.option('--components', '-c', multiple=True,
              default=['core', 'python'],
              help='Components to test')
@click.option('--coverage', is_flag=True,
              help='Generate coverage report')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose test output')
def test(components, coverage, verbose):
    """Run test suite"""
    
    click.echo("Running tests...")
    
    exit_code = 0
    
    for component in components:
        click.echo(f"\nTesting {component}...")
        
        if component == 'core':
            # Run C++ tests
            cmd = ['ctest', '--output-on-failure']
            if verbose:
                cmd.append('-V')
            
            import subprocess
            result = subprocess.run(cmd, cwd='build')
            if result.returncode != 0:
                exit_code = 1
                
        elif component == 'python':
            # Run Python tests
            cmd = ['pytest', 'python/tests']
            if verbose:
                cmd.append('-v')
            if coverage:
                cmd.extend(['--cov=pipeline_sim', '--cov-report=html'])
            
            import subprocess
            result = subprocess.run(cmd)
            if result.returncode != 0:
                exit_code = 1
    
    sys.exit(exit_code)


def generate_plots(network, results):
    """Generate visualization plots"""
    
    # Pressure profile
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Extract node data
    node_ids = list(network.nodes.keys())
    pressures = [results.node_pressures.get(nid, 0)/1e5 for nid in node_ids]
    elevations = [network.nodes[nid].elevation for nid in node_ids]
    
    ax1.bar(range(len(node_ids)), pressures)
    ax1.set_xticks(range(len(node_ids)))
    ax1.set_xticklabels(node_ids, rotation=45)
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_title('Node Pressures')
    ax1.grid(True, alpha=0.3)
    
    # Flow rates
    pipe_ids = list(network.pipes.keys())
    flows = [results.pipe_flow_rates.get(pid, 0) for pid in pipe_ids]
    
    ax2.bar(range(len(pipe_ids)), flows)
    ax2.set_xticks(range(len(pipe_ids)))
    ax2.set_xticklabels(pipe_ids, rotation=45)
    ax2.set_ylabel('Flow Rate (m³/s)')
    ax2.set_title('Pipe Flow Rates')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150)
    click.echo("Plots saved to simulation_results.png")