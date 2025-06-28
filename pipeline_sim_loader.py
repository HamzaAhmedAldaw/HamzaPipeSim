#!/usr/bin/env python3
"""
Pipeline-Sim Module Loader with IDE Support
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add build directory to path
REPO_ROOT = Path(__file__).parent
BUILD_DIR = REPO_ROOT / "build" / "lib.win-amd64-cpython-313"

if BUILD_DIR.exists():
    sys.path.insert(0, str(BUILD_DIR))
else:
    # Try to find the build directory
    for item in (REPO_ROOT / "build").iterdir() if (REPO_ROOT / "build").exists() else []:
        if item.is_dir() and "lib" in item.name:
            sys.path.insert(0, str(item))
            BUILD_DIR = item
            break

# For type checking and IDE support
if TYPE_CHECKING:
    # Create stub definitions for IDE
    class NodeType:
        SOURCE: int = 0
        SINK: int = 1
        JUNCTION: int = 2
        PUMP: int = 3
        COMPRESSOR: int = 4
        VALVE: int = 5
        SEPARATOR: int = 6
    
    class Node:
        def __init__(self, id: str, type: NodeType): ...
        @property
        def id(self) -> str: ...
        @property
        def type(self) -> NodeType: ...
        @property
        def pressure(self) -> float: ...
        @pressure.setter
        def pressure(self, value: float): ...
        @property
        def temperature(self) -> float: ...
        @temperature.setter
        def temperature(self, value: float): ...
        @property
        def elevation(self) -> float: ...
        @elevation.setter
        def elevation(self, value: float): ...
    
    class Pipe:
        def __init__(self, id: str, upstream: Node, downstream: Node, length: float, diameter: float): ...
        @property
        def id(self) -> str: ...
        @property
        def length(self) -> float: ...
        @property
        def diameter(self) -> float: ...
        @property
        def roughness(self) -> float: ...
        @roughness.setter
        def roughness(self, value: float): ...
        @property
        def inclination(self) -> float: ...
        @inclination.setter
        def inclination(self, value: float): ...
        def area(self) -> float: ...
        def velocity(self) -> float: ...
        def reynolds_number(self) -> float: ...
    
    class FluidProperties:
        def __init__(self): ...
        oil_density: float
        gas_density: float
        water_density: float
        oil_viscosity: float
        gas_viscosity: float
        water_viscosity: float
        oil_fraction: float
        gas_fraction: float
        water_fraction: float
        gas_oil_ratio: float
        water_cut: float
        def mixture_density(self) -> float: ...
        def mixture_viscosity(self) -> float: ...
    
    class Network:
        def __init__(self): ...
        def add_node(self, id: str, type: NodeType) -> Node: ...
        def add_pipe(self, id: str, upstream: Node, downstream: Node, length: float, diameter: float) -> Pipe: ...
        def set_pressure(self, node_id: str, pressure: float): ...
        def set_flow_rate(self, node_id: str, flow_rate: float): ...
        @property
        def nodes(self) -> dict: ...
        @property
        def pipes(self) -> dict: ...
    
    class SolutionResults:
        converged: bool
        iterations: int
        residual: float
        node_pressures: dict
        pipe_flow_rates: dict
        pipe_pressure_drops: dict
    
    class SteadyStateSolver:
        def __init__(self, network: Network, fluid: FluidProperties): ...
        def solve(self) -> SolutionResults: ...

# Actual loading logic
def load_module():
    """Load the pipeline_sim C++ module"""
    # Method 1: Try direct import from build
    try:
        pyd_path = BUILD_DIR / "pipeline_sim.cp313-win_amd64.pyd"
        if pyd_path.exists():
            spec = importlib.util.spec_from_file_location("pipeline_sim", str(pyd_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✓ Loaded pipeline_sim from: {pyd_path}")
            return module
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Try finding any .pyd file
    try:
        for pyd_file in BUILD_DIR.rglob("*.pyd"):
            if "pipeline_sim" in pyd_file.name:
                spec = importlib.util.spec_from_file_location("pipeline_sim", str(pyd_file))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"✓ Loaded pipeline_sim from: {pyd_file}")
                return module
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    # Method 3: Try the egg installation
    try:
        egg_path = Path(r"C:\Users\KIMO STORE\miniconda3\Lib\site-packages\pipeline_sim-0.1.0-py3.13-win-amd64.egg")
        if egg_path not in sys.path:
            sys.path.insert(0, str(egg_path))
        import pipeline_sim
        print(f"✓ Loaded pipeline_sim from egg: {egg_path}")
        return pipeline_sim
    except Exception as e:
        print(f"Method 3 failed: {e}")
    
    raise ImportError("Could not load pipeline_sim module from any location")

# Load the module
try:
    pipeline_sim = load_module()
    
    # Export all attributes
    for attr in dir(pipeline_sim):
        if not attr.startswith('_'):
            globals()[attr] = getattr(pipeline_sim, attr)
    
    # For convenience
    Network = pipeline_sim.Network
    Node = pipeline_sim.Node
    Pipe = pipeline_sim.Pipe
    NodeType = pipeline_sim.NodeType
    FluidProperties = pipeline_sim.FluidProperties
    SteadyStateSolver = pipeline_sim.SteadyStateSolver
    
except ImportError as e:
    print(f"Failed to load pipeline_sim: {e}")
    if not TYPE_CHECKING:
        raise