# pipeline_sim_loader.py
"""Pipeline-Sim module loader"""
import importlib.util

# Load the module once
spec = importlib.util.spec_from_file_location(
    "pipeline_sim", 
    r"C:\Users\KIMO STORE\HamzaPipeSim\python\pipeline_sim\_core.cp313-win_amd64.pyd"
)
pipeline_sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_sim)

# Make all classes available
Network = pipeline_sim.Network
Node = pipeline_sim.Node
Pipe = pipeline_sim.Pipe
NodeType = pipeline_sim.NodeType
FluidProperties = pipeline_sim.FluidProperties
SteadyStateSolver = pipeline_sim.SteadyStateSolver
TransientSolver = pipeline_sim.TransientSolver
# Add more as needed...