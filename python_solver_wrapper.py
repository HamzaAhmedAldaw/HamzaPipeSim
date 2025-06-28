#!/usr/bin/env python3
"""
Python-based solver wrapper for Pipeline-Sim
Works around C++ solver issues by implementing the solving logic in Python
"""

import numpy as np
import pipeline_sim
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import time

class PythonSolver:
    """Python implementation of steady-state solver"""
    
    def __init__(self, network, fluid):
        self.network = network
        self.fluid = fluid
        self.nodes = {}
        self.pipes = {}
        self.node_indices = {}
        self.pipe_indices = {}
        
        # Build node and pipe dictionaries
        self._build_indices()
        
    def _build_indices(self):
        """Build indices for nodes and pipes"""
        # Get all nodes
        node_idx = 0
        for node_id, node in self.network.nodes().items():
            self.nodes[node_id] = node
            self.node_indices[node_id] = node_idx
            node_idx += 1
        
        # Get all pipes
        pipe_idx = 0
        for pipe_id, pipe in self.network.pipes().items():
            self.pipes[pipe_id] = pipe
            self.pipe_indices[pipe_id] = pipe_idx
            pipe_idx += 1
    
    def solve(self, max_iter=50, tolerance=1e-6, verbose=True):
        """Solve the network using Newton-Raphson method"""
        
        n_nodes = len(self.nodes)
        n_pipes = len(self.pipes)
        n_unknowns = n_nodes + n_pipes
        
        if verbose:
            print(f"Solving network: {n_nodes} nodes, {n_pipes} pipes")
        
        # Initial guess
        x = np.zeros(n_unknowns)
        
        # Set initial pressures
        for i, (node_id, node) in enumerate(self.nodes.items()):
            x[i] = 101325.0  # Atmospheric pressure
        
        # Set initial flows (small positive value)
        for i, pipe_id in enumerate(self.pipes):
            x[n_nodes + i] = 0.001  # Small flow rate
        
        # Get boundary conditions
        pressure_specs = {}
        try:
            pressure_specs = self.network.pressure_specs()
        except:
            # If method doesn't exist, extract from nodes
            for node_id, node in self.nodes.items():
                # Check if node has fixed pressure (sources and sinks typically do)
                if node.type() in [pipeline_sim.NodeType.SOURCE, pipeline_sim.NodeType.SINK]:
                    pressure_specs[node_id] = node.pressure()
        
        if verbose:
            print(f"Pressure boundary conditions: {len(pressure_specs)}")
        
        # Newton-Raphson iterations
        for iteration in range(max_iter):
            # Build system matrix and RHS
            A, b = self._build_system(x, pressure_specs)
            
            # Solve linear system
            try:
                dx = spsolve(A, b)
            except Exception as e:
                print(f"Matrix solve failed: {e}")
                return self._create_failed_result()
            
            # Update solution
            x = x - 0.8 * dx  # With relaxation
            
            # Check convergence
            residual = np.linalg.norm(dx)
            
            if verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration}: residual = {residual:.2e}")
            
            if residual < tolerance:
                if verbose:
                    print(f"Converged in {iteration + 1} iterations")
                return self._create_result(x, True, iteration + 1, residual)
        
        if verbose:
            print(f"Failed to converge after {max_iter} iterations")
        return self._create_result(x, False, max_iter, residual)
    
    def _build_system(self, x, pressure_specs):
        """Build the system matrix A and RHS vector b"""
        n_nodes = len(self.nodes)
        n_pipes = len(self.pipes)
        n = n_nodes + n_pipes
        
        # Initialize sparse matrix components
        rows = []
        cols = []
        data = []
        b = np.zeros(n)
        
        # Current solution
        pressures = x[:n_nodes]
        flows = x[n_nodes:]
        
        # 1. Mass conservation equations at nodes
        for node_id, node in self.nodes.items():
            node_idx = self.node_indices[node_id]
            
            if node_id in pressure_specs:
                # Pressure is specified - add identity equation
                rows.append(node_idx)
                cols.append(node_idx)
                data.append(1.0)
                b[node_idx] = pressure_specs[node_id]
            else:
                # Mass conservation: sum of flows = 0
                # Find connected pipes
                for pipe_id, pipe in self.pipes.items():
                    pipe_idx = n_nodes + self.pipe_indices[pipe_id]
                    
                    upstream_id = pipe.upstream().id()
                    downstream_id = pipe.downstream().id()
                    
                    if upstream_id == node_id:
                        # Flow leaves this node (negative)
                        rows.append(node_idx)
                        cols.append(pipe_idx)
                        data.append(-1.0)
                    elif downstream_id == node_id:
                        # Flow enters this node (positive)
                        rows.append(node_idx)
                        cols.append(pipe_idx)
                        data.append(1.0)
                
                b[node_idx] = 0.0
        
        # 2. Momentum equations for pipes
        density = self.fluid.mixture_density()
        viscosity = self.fluid.mixture_viscosity()
        
        for pipe_id, pipe in self.pipes.items():
            pipe_idx = n_nodes + self.pipe_indices[pipe_id]
            eq_idx = pipe_idx
            
            upstream_id = pipe.upstream().id()
            downstream_id = pipe.downstream().id()
            upstream_idx = self.node_indices[upstream_id]
            downstream_idx = self.node_indices[downstream_id]
            
            # Get current flow
            q = flows[self.pipe_indices[pipe_id]]
            
            # Pipe properties
            L = pipe.length()
            D = pipe.diameter()
            A = np.pi * D**2 / 4
            e = pipe.roughness()
            
            # Calculate friction factor
            if abs(q) > 1e-10:
                v = q / A
                Re = density * abs(v) * D / viscosity
                
                if Re < 2300:
                    f = 64 / Re
                else:
                    # Colebrook equation (simplified)
                    f = 0.25 / (np.log10(e/(3.7*D) + 5.74/Re**0.9))**2
            else:
                f = 64 / 2300  # Laminar approximation
            
            # Linearized momentum equation
            # P_upstream - P_downstream - resistance * Q = 0
            
            # Pressure terms
            rows.append(eq_idx)
            cols.append(upstream_idx)
            data.append(1.0)
            
            rows.append(eq_idx)
            cols.append(downstream_idx)
            data.append(-1.0)
            
            # Flow resistance term (linearized)
            resistance = f * L * density * abs(q) / (2 * D * A**2)
            if abs(q) < 1e-10:
                resistance = f * L * density / (2 * D * A**2)  # Avoid division by zero
            
            rows.append(eq_idx)
            cols.append(pipe_idx)
            data.append(-resistance)
            
            # Gravity term
            dz = pipe.downstream().elevation() - pipe.upstream().elevation()
            b[eq_idx] = -density * 9.81 * dz
        
        # Create sparse matrix
        A = csr_matrix((data, (rows, cols)), shape=(n, n))
        
        return A, b
    
    def _create_result(self, x, converged, iterations, residual):
        """Create result object"""
        n_nodes = len(self.nodes)
        
        # Update node pressures
        for i, (node_id, node) in enumerate(self.nodes.items()):
            node.set_pressure(x[i])
        
        # Update pipe flows
        for i, (pipe_id, pipe) in enumerate(self.pipes.items()):
            pipe.set_flow_rate(x[n_nodes + i])
        
        # Create result dictionary (mimics C++ SolutionResults)
        class Result:
            pass
        
        result = Result()
        result.converged = converged
        result.iterations = iterations
        result.residual = residual
        result.node_pressures = {}
        result.pipe_flow_rates = {}
        result.pipe_velocities = {}
        result.pipe_pressure_drops = {}
        
        # Fill results
        for node_id, node in self.nodes.items():
            result.node_pressures[node_id] = node.pressure()
        
        for pipe_id, pipe in self.pipes.items():
            result.pipe_flow_rates[pipe_id] = pipe.flow_rate()
            result.pipe_velocities[pipe_id] = pipe.velocity()
            
            # Calculate pressure drop
            p_up = pipe.upstream().pressure()
            p_down = pipe.downstream().pressure()
            result.pipe_pressure_drops[pipe_id] = p_up - p_down
        
        return result
    
    def _create_failed_result(self):
        """Create a failed result"""
        class Result:
            pass
        
        result = Result()
        result.converged = False
        result.iterations = 0
        result.residual = 1e10
        result.node_pressures = {}
        result.pipe_flow_rates = {}
        result.pipe_velocities = {}
        result.pipe_pressure_drops = {}
        
        return result


# Test the Python solver
if __name__ == "__main__":
    print("=== Testing Python Solver ===\n")
    
    # Create simple network
    network = pipeline_sim.Network()
    
    # Create nodes
    w1 = network.add_node("WELL-1", pipeline_sim.NodeType.SOURCE)
    w2 = network.add_node("WELL-2", pipeline_sim.NodeType.SOURCE)
    man = network.add_node("MANIFOLD", pipeline_sim.NodeType.JUNCTION)
    sep = network.add_node("SEP", pipeline_sim.NodeType.SINK)
    
    # Create pipes
    p1 = network.add_pipe("P1", w1, man, 2000, 0.25)
    p2 = network.add_pipe("P2", w2, man, 2500, 0.25)
    p3 = network.add_pipe("P3", man, sep, 5000, 0.4)
    
    # Set boundary conditions
    network.set_pressure(w1, 300e5)  # 300 bar
    network.set_pressure(w2, 320e5)  # 320 bar
    network.set_pressure(sep, 150e5)  # 150 bar
    
    # Create fluid
    fluid = pipeline_sim.FluidProperties()
    fluid.oil_fraction = 1.0
    fluid.oil_density = 850.0
    fluid.oil_viscosity = 0.001
    
    # Solve with Python solver
    solver = PythonSolver(network, fluid)
    result = solver.solve(verbose=True)
    
    if result.converged:
        print("\nResults:")
        print("Pressures:")
        for node_id, pressure in result.node_pressures.items():
            print(f"  {node_id}: {pressure/1e5:.1f} bar")
        
        print("\nFlows:")
        for pipe_id, flow in result.pipe_flow_rates.items():
            print(f"  {pipe_id}: {flow*3600:.1f} m³/h")
    else:
        print("Failed to converge!")