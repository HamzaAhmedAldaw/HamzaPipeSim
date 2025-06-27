# AI_GENERATED: GUI application and deployment
# Generated on: 2025-06-27

# ===== tools/gui/pipeline_sim_gui.py =====
#!/usr/bin/env python3
"""
Pipeline-Sim GUI Application

AI_GENERATED: Graphical interface for pipeline simulation
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import networkx as nx
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import pipeline_sim as ps


class PipelineSimGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Pipeline-Sim v{ps.__version__}")
        self.root.geometry("1200x800")
        
        # State
        self.network = None
        self.fluid = None
        self.results = None
        self.network_file = None
        
        # Create UI
        self.create_menu()
        self.create_toolbar()
        self.create_main_layout()
        self.create_status_bar()
        
        # Initial state
        self.update_ui_state()
        
    def create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Network", command=self.new_network)
        file_menu.add_command(label="Open Network...", command=self.open_network)
        file_menu.add_command(label="Save Network", command=self.save_network)
        file_menu.add_command(label="Save Network As...", command=self.save_network_as)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Add Node...", command=self.add_node_dialog)
        edit_menu.add_command(label="Add Pipe...", command=self.add_pipe_dialog)
        edit_menu.add_separator()
        edit_menu.add_command(label="Fluid Properties...", command=self.edit_fluid_dialog)
        
        # Simulation menu
        sim_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(label="Run Steady State", command=self.run_steady_state)
        sim_menu.add_command(label="Run Transient...", command=self.run_transient_dialog)
        sim_menu.add_separator()
        sim_menu.add_command(label="Solver Settings...", command=self.solver_settings_dialog)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Validate Network", command=self.validate_network)
        tools_menu.add_command(label="Optimize Network...", command=self.optimize_dialog)
        tools_menu.add_separator()
        tools_menu.add_command(label="Generate Report", command=self.generate_report)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_docs)
        help_menu.add_command(label="About", command=self.show_about)
        
    def create_toolbar(self):
        """Create toolbar"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Toolbar buttons
        ttk.Button(toolbar, text="Open", command=self.open_network).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save", command=self.save_network).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(toolbar, text="Run", command=self.run_steady_state).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Stop", command=self.stop_simulation, state=tk.DISABLED).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Correlation selector
        ttk.Label(toolbar, text="Correlation:").pack(side=tk.LEFT, padx=5)
        self.correlation_var = tk.StringVar(value="Beggs-Brill")
        correlation_combo = ttk.Combobox(toolbar, textvariable=self.correlation_var,
                                       values=["Beggs-Brill", "Hagedorn-Brown", "Gray", "Mechanistic"],
                                       width=15)
        correlation_combo.pack(side=tk.LEFT, padx=2)
        
    def create_main_layout(self):
        """Create main application layout"""
        # Main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Network tree and properties
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Network tree
        tree_frame = ttk.LabelFrame(left_frame, text="Network Structure")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.network_tree = ttk.Treeview(tree_frame, columns=("Type", "Value"))
        self.network_tree.heading("#0", text="Item")
        self.network_tree.heading("Type", text="Type")
        self.network_tree.heading("Value", text="Value")
        self.network_tree.column("#0", width=150)
        self.network_tree.column("Type", width=100)
        self.network_tree.column("Value", width=100)
        
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.network_tree.yview)
        self.network_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.network_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind tree selection
        self.network_tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        
        # Properties panel
        props_frame = ttk.LabelFrame(left_frame, text="Properties")
        props_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.props_text = tk.Text(props_frame, height=8, width=40)
        self.props_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - Visualization
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        
        # Tab control for different views
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Network view tab
        self.network_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.network_tab, text="Network View")
        
        # Results view tab
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Results")
        
        # Plots tab
        self.plots_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plots_tab, text="Plots")
        
        # Create matplotlib figure for network view
        self.create_network_view()
        
    def create_network_view(self):
        """Create network visualization"""
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.network_tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.network_tab)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.progress_bar = ttk.Progressbar(self.status_bar, length=200, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
        
    def update_ui_state(self):
        """Update UI based on current state"""
        has_network = self.network is not None
        has_results = self.results is not None
        
        # Update tree
        if has_network:
            self.populate_network_tree()
            self.draw_network()
        
        # Update results
        if has_results:
            self.display_results()
            
    def populate_network_tree(self):
        """Populate network tree view"""
        self.network_tree.delete(*self.network_tree.get_children())
        
        if not self.network:
            return
        
        # Add nodes
        nodes_item = self.network_tree.insert("", "end", text="Nodes", values=("", f"{len(self.network.nodes)}"))
        for node_id, node in self.network.nodes.items():
            values = (str(node.type).split('.')[-1], f"{node.pressure/1e5:.1f} bar")
            self.network_tree.insert(nodes_item, "end", text=node_id, values=values, tags=("node",))
        
        # Add pipes
        pipes_item = self.network_tree.insert("", "end", text="Pipes", values=("", f"{len(self.network.pipes)}"))
        for pipe_id, pipe in self.network.pipes.items():
            values = (f"L={pipe.length}m", f"D={pipe.diameter}m")
            self.network_tree.insert(pipes_item, "end", text=pipe_id, values=values, tags=("pipe",))
        
        # Expand all
        self.network_tree.item(nodes_item, open=True)
        self.network_tree.item(pipes_item, open=True)
        
    def draw_network(self):
        """Draw network diagram"""
        if not self.network:
            return
        
        self.ax.clear()
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        pos = {}
        for i, (node_id, node) in enumerate(self.network.nodes.items()):
            G.add_node(node_id)
            # Simple layout - can be improved
            x = i % 4
            y = i // 4
            pos[node_id] = (x, y)
        
        # Add edges
        for pipe_id, pipe in self.network.pipes.items():
            G.add_edge(pipe.upstream.id, pipe.downstream.id, label=pipe_id)
        
        # Draw
        nx.draw(G, pos, ax=self.ax, with_labels=True, node_color='lightblue',
                node_size=1500, font_size=10, font_weight='bold',
                arrows=True, edge_color='gray', width=2)
        
        # Add edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=self.ax)
        
        self.ax.set_title("Pipeline Network")
        self.canvas.draw()
        
    def on_tree_select(self, event):
        """Handle tree selection"""
        selection = self.network_tree.selection()
        if not selection:
            return
        
        item = self.network_tree.item(selection[0])
        item_text = item['text']
        
        # Clear properties
        self.props_text.delete(1.0, tk.END)
        
        # Show properties based on selection
        if 'node' in item.get('tags', []):
            node = self.network.nodes.get(item_text)
            if node:
                props = f"Node: {node.id}\n"
                props += f"Type: {node.type}\n"
                props += f"Pressure: {node.pressure/1e5:.2f} bar\n"
                props += f"Temperature: {node.temperature:.1f} K\n"
                props += f"Elevation: {node.elevation:.1f} m\n"
                self.props_text.insert(1.0, props)
                
        elif 'pipe' in item.get('tags', []):
            pipe = self.network.pipes.get(item_text)
            if pipe:
                props = f"Pipe: {pipe.id}\n"
                props += f"Length: {pipe.length:.1f} m\n"
                props += f"Diameter: {pipe.diameter:.3f} m\n"
                props += f"Roughness: {pipe.roughness*1000:.3f} mm\n"
                props += f"Inclination: {pipe.inclination*180/3.14159:.1f}°\n"
                if self.results:
                    flow = self.results.pipe_flow_rates.get(pipe.id, 0)
                    props += f"\nFlow: {flow:.4f} m³/s\n"
                    props += f"Velocity: {flow/pipe.area():.2f} m/s\n"
                self.props_text.insert(1.0, props)
    
    def new_network(self):
        """Create new network"""
        self.network = ps.Network()
        self.fluid = ps.FluidProperties()
        self.results = None
        self.network_file = None
        self.update_ui_state()
        self.set_status("New network created")
        
    def open_network(self):
        """Open network file"""
        filename = filedialog.askopenfilename(
            title="Open Network",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.network, self.fluid = ps.load_network(filename)
                self.network_file = filename
                self.results = None
                self.update_ui_state()
                self.set_status(f"Loaded: {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load network:\n{str(e)}")
                
    def save_network(self):
        """Save network"""
        if not self.network_file:
            self.save_network_as()
        else:
            self.save_network_to_file(self.network_file)
            
    def save_network_as(self):
        """Save network with new filename"""
        filename = filedialog.asksaveasfilename(
            title="Save Network",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            self.save_network_to_file(filename)
            self.network_file = filename
            
    def save_network_to_file(self, filename):
        """Save network to file"""
        try:
            # Build network data
            data = {
                "nodes": [],
                "pipes": [],
                "fluid": {}
            }
            
            # Add nodes
            for node_id, node in self.network.nodes.items():
                node_data = {
                    "id": node_id,
                    "type": str(node.type).split('.')[-1],
                    "elevation": node.elevation
                }
                
                # Add boundary conditions
                if node_id in self.network.pressure_specs:
                    node_data["pressure"] = self.network.pressure_specs[node_id]
                if node_id in self.network.flow_specs:
                    node_data["flow_rate"] = self.network.flow_specs[node_id]
                    
                data["nodes"].append(node_data)
            
            # Add pipes
            for pipe_id, pipe in self.network.pipes.items():
                pipe_data = {
                    "id": pipe_id,
                    "upstream": pipe.upstream.id,
                    "downstream": pipe.downstream.id,
                    "length": pipe.length,
                    "diameter": pipe.diameter,
                    "roughness": pipe.roughness,
                    "inclination": pipe.inclination
                }
                data["pipes"].append(pipe_data)
            
            # Add fluid properties
            if self.fluid:
                data["fluid"] = {
                    "oil_density": self.fluid.oil_density,
                    "gas_density": self.fluid.gas_density,
                    "water_density": self.fluid.water_density,
                    "oil_viscosity": self.fluid.oil_viscosity,
                    "gas_viscosity": self.fluid.gas_viscosity,
                    "water_viscosity": self.fluid.water_viscosity,
                    "gas_oil_ratio": self.fluid.gas_oil_ratio,
                    "water_cut": self.fluid.water_cut
                }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.set_status(f"Saved: {Path(filename).name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save network:\n{str(e)}")
            
    def run_steady_state(self):
        """Run steady-state simulation"""
        if not self.network:
            messagebox.showwarning("Warning", "No network loaded")
            return
            
        try:
            self.set_status("Running simulation...")
            self.progress_bar.start()
            
            # Create solver
            solver = ps.SteadyStateSolver(self.network, self.fluid)
            
            # TODO: Apply correlation selection
            
            # Run simulation
            self.results = solver.solve()
            
            self.progress_bar.stop()
            
            if self.results.converged:
                self.set_status(f"Simulation converged in {self.results.iterations} iterations")
                self.update_ui_state()
            else:
                messagebox.showwarning("Warning", "Simulation failed to converge")
                self.set_status("Simulation failed")
                
        except Exception as e:
            self.progress_bar.stop()
            messagebox.showerror("Error", f"Simulation error:\n{str(e)}")
            traceback.print_exc()
            
    def display_results(self):
        """Display simulation results"""
        if not self.results:
            return
            
        # Clear results tab
        for widget in self.results_tab.winfo_children():
            widget.destroy()
            
        # Create results notebook
        results_notebook = ttk.Notebook(self.results_tab)
        results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary tab
        summary_frame = ttk.Frame(results_notebook)
        results_notebook.add(summary_frame, text="Summary")
        
        summary_text = tk.Text(summary_frame, wrap=tk.WORD)
        summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        summary = f"Simulation Results\n"
        summary += f"="*50 + "\n"
        summary += f"Converged: {'Yes' if self.results.converged else 'No'}\n"
        summary += f"Iterations: {self.results.iterations}\n"
        summary += f"Residual: {self.results.residual:.2e}\n"
        summary += f"Computation Time: {self.results.computation_time:.3f} s\n"
        summary += f"\nNodes: {len(self.results.node_pressures)}\n"
        summary += f"Pipes: {len(self.results.pipe_flow_rates)}\n"
        
        summary_text.insert(1.0, summary)
        summary_text.config(state=tk.DISABLED)
        
        # Node results tab
        nodes_frame = ttk.Frame(results_notebook)
        results_notebook.add(nodes_frame, text="Nodes")
        
        # Create treeview for node results
        node_tree = ttk.Treeview(nodes_frame, columns=("Pressure", "Temperature"))
        node_tree.heading("#0", text="Node ID")
        node_tree.heading("Pressure", text="Pressure (bar)")
        node_tree.heading("Temperature", text="Temperature (K)")
        
        for node_id, pressure in self.results.node_pressures.items():
            temp = self.results.node_temperatures.get(node_id, 288.15)
            node_tree.insert("", "end", text=node_id, 
                           values=(f"{pressure/1e5:.2f}", f"{temp:.1f}"))
        
        node_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Pipe results tab
        pipes_frame = ttk.Frame(results_notebook)
        results_notebook.add(pipes_frame, text="Pipes")
        
        # Create treeview for pipe results
        pipe_tree = ttk.Treeview(pipes_frame, columns=("Flow", "Velocity", "Pressure Drop"))
        pipe_tree.heading("#0", text="Pipe ID")
        pipe_tree.heading("Flow", text="Flow (m³/s)")
        pipe_tree.heading("Velocity", text="Velocity (m/s)")
        pipe_tree.heading("Pressure Drop", text="ΔP (bar)")
        
        for pipe_id, flow in self.results.pipe_flow_rates.items():
            pipe = self.network.pipes[pipe_id]
            velocity = flow / pipe.area()
            dp = self.results.pipe_pressure_drops.get(pipe_id, 0)
            pipe_tree.insert("", "end", text=pipe_id,
                           values=(f"{flow:.4f}", f"{velocity:.2f}", f"{dp/1e5:.3f}"))
        
        pipe_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create plots
        self.create_result_plots()
        
    def create_result_plots(self):
        """Create result plots"""
        # Clear plots tab
        for widget in self.plots_tab.winfo_children():
            widget.destroy()
            
        # Create figure with subplots
        fig = Figure(figsize=(10, 8), dpi=100)
        
        # Pressure distribution
        ax1 = fig.add_subplot(221)
        node_ids = list(self.results.node_pressures.keys())
        pressures = [self.results.node_pressures[nid]/1e5 for nid in node_ids]
        
        ax1.bar(range(len(node_ids)), pressures)
        ax1.set_xticks(range(len(node_ids)))
        ax1.set_xticklabels(node_ids, rotation=45)
        ax1.set_ylabel('Pressure (bar)')
        ax1.set_title('Node Pressures')
        ax1.grid(True, alpha=0.3)
        
        # Flow distribution
        ax2 = fig.add_subplot(222)
        pipe_ids = list(self.results.pipe_flow_rates.keys())
        flows = [self.results.pipe_flow_rates[pid] for pid in pipe_ids]
        
        ax2.bar(range(len(pipe_ids)), flows)
        ax2.set_xticks(range(len(pipe_ids)))
        ax2.set_xticklabels(pipe_ids, rotation=45)
        ax2.set_ylabel('Flow Rate (m³/s)')
        ax2.set_title('Pipe Flow Rates')
        ax2.grid(True, alpha=0.3)
        
        # Velocity distribution
        ax3 = fig.add_subplot(223)
        velocities = []
        for pid in pipe_ids:
            pipe = self.network.pipes[pid]
            velocity = flows[pipe_ids.index(pid)] / pipe.area()
            velocities.append(velocity)
            
        ax3.bar(range(len(pipe_ids)), velocities)
        ax3.set_xticks(range(len(pipe_ids)))
        ax3.set_xticklabels(pipe_ids, rotation=45)
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('Flow Velocities')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=10, color='r', linestyle='--', label='Max recommended')
        ax3.legend()
        
        # Pressure drop
        ax4 = fig.add_subplot(224)
        pressure_drops = [self.results.pipe_pressure_drops.get(pid, 0)/1e5 for pid in pipe_ids]
        
        ax4.bar(range(len(pipe_ids)), pressure_drops)
        ax4.set_xticks(range(len(pipe_ids)))
        ax4.set_xticklabels(pipe_ids, rotation=45)
        ax4.set_ylabel('Pressure Drop (bar)')
        ax4.set_title('Pressure Drops')
        ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=self.plots_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.plots_tab)
        toolbar.update()
        
    def set_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update()
        
    def stop_simulation(self):
        """Stop running simulation"""
        # TODO: Implement simulation cancellation
        pass
        
    def validate_network(self):
        """Validate network configuration"""
        if not self.network:
            messagebox.showwarning("Warning", "No network loaded")
            return
            
        errors = []
        warnings = []
        
        # Check connectivity
        for node_id, node in self.network.nodes.items():
            upstream = self.network.get_upstream_pipes(node)
            downstream = self.network.get_downstream_pipes(node)
            
            if not upstream and not downstream:
                warnings.append(f"Node '{node_id}' is not connected")
                
        # Check boundary conditions
        if not self.network.pressure_specs:
            errors.append("No pressure boundary conditions")
            
        # Display results
        msg = "Network Validation Results\n\n"
        
        if errors:
            msg += f"Errors ({len(errors)}):\n"
            for error in errors:
                msg += f"• {error}\n"
            msg += "\n"
            
        if warnings:
            msg += f"Warnings ({len(warnings)}):\n"
            for warning in warnings:
                msg += f"• {warning}\n"
        else:
            msg += "No issues found!"
            
        messagebox.showinfo("Validation Results", msg)
        
    def generate_report(self):
        """Generate simulation report"""
        if not self.results:
            messagebox.showwarning("Warning", "No simulation results available")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                ps.generate_report(self.network, self.results, self.fluid, filename)
                self.set_status(f"Report saved: {Path(filename).name}")
                
                # Ask to open report
                if messagebox.askyesno("Report Generated", 
                                     "Report generated successfully.\nOpen in browser?"):
                    import webbrowser
                    webbrowser.open(filename)
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate report:\n{str(e)}")
                
    def show_about(self):
        """Show about dialog"""
        about_text = f"""Pipeline-Sim v{ps.__version__}

Next-generation petroleum pipeline simulation

© 2025 Pipeline-Sim Contributors
MIT License

AI-assisted development for
advanced pipeline analysis"""
        
        messagebox.showinfo("About Pipeline-Sim", about_text)
        
    def show_docs(self):
        """Open documentation"""
        import webbrowser
        webbrowser.open("https://pipeline-sim.readthedocs.io")
        
    # Dialog methods would be implemented here...
    def add_node_dialog(self):
        """Dialog to add new node"""
        # TODO: Implement node addition dialog
        pass
        
    def add_pipe_dialog(self):
        """Dialog to add new pipe"""
        # TODO: Implement pipe addition dialog
        pass
        
    def edit_fluid_dialog(self):
        """Dialog to edit fluid properties"""
        # TODO: Implement fluid properties dialog
        pass
        
    def solver_settings_dialog(self):
        """Dialog for solver settings"""
        # TODO: Implement solver settings dialog
        pass
        
    def run_transient_dialog(self):
        """Dialog to configure and run transient simulation"""
        # TODO: Implement transient simulation dialog
        pass
        
    def optimize_dialog(self):
        """Dialog for network optimization"""
        # TODO: Implement optimization dialog
        pass
        
    def export_results(self):
        """Export simulation results"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                ps.save_results(self.results, filename)
                self.set_status(f"Results exported: {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results:\n{str(e)}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = PipelineSimGUI(root)
    root.mainloop()
