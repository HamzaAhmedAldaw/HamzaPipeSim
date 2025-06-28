#!/usr/bin/env python3
"""
Professional Production Analysis System
Pipeline Network Simulation & Optimization Tool
Similar to PIPESIM/OLGA professional interface
"""

import sys
import os
from datetime import datetime
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import pipeline_sim

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')

class ProductionAnalysisSystem(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.simulation_results = None
        self.init_ui()
        self.apply_dark_theme()
        
    def init_ui(self):
        """Initialize the user interface"""
        
        # Window setup
        self.setWindowTitle("Pipeline-Sim Professional - Production Analysis System")
        self.setGeometry(100, 100, 1600, 900)
        
        # Set window icon
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Create main content area
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(5, 5, 5, 5)
        
        # Left panel - Project tree and properties
        left_panel = self.create_left_panel()
        content_layout.addWidget(left_panel, 1)
        
        # Center area - Main workspace
        self.main_tabs = QTabWidget()
        self.main_tabs.setTabPosition(QTabWidget.North)
        
        # Create tabs
        self.create_network_tab()
        self.create_wells_tab()
        self.create_pipes_tab()
        self.create_fluids_tab()
        self.create_simulation_tab()
        self.create_results_tab()
        self.create_reports_tab()
        
        content_layout.addWidget(self.main_tabs, 4)
        
        # Right panel - Quick access tools
        right_panel = self.create_right_panel()
        content_layout.addWidget(right_panel, 1)
        
        main_layout.addWidget(content_widget)
        
    def create_menu_bar(self):
        """Create professional menu bar"""
        
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        new_action = QAction(QIcon(), '&New Project', self)
        new_action.setShortcut('Ctrl+N')
        new_action.setStatusTip('Create new project')
        file_menu.addAction(new_action)
        
        open_action = QAction(QIcon(), '&Open Project', self)
        open_action.setShortcut('Ctrl+O')
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_action = QAction(QIcon(), '&Save', self)
        save_action.setShortcut('Ctrl+S')
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction(QIcon(), '&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu('&Edit')
        edit_menu.addAction('&Undo')
        edit_menu.addAction('&Redo')
        edit_menu.addSeparator()
        edit_menu.addAction('&Copy')
        edit_menu.addAction('&Paste')
        
        # Model menu
        model_menu = menubar.addMenu('&Model')
        model_menu.addAction('&Network Configuration')
        model_menu.addAction('&Well Management')
        model_menu.addAction('&Pipeline Design')
        model_menu.addAction('&Fluid Properties')
        model_menu.addSeparator()
        model_menu.addAction('&Validate Model')
        
        # Simulation menu
        sim_menu = menubar.addMenu('&Simulation')
        sim_menu.addAction('&Run Steady State')
        sim_menu.addAction('&Run Transient')
        sim_menu.addSeparator()
        sim_menu.addAction('&Optimization')
        sim_menu.addAction('&Sensitivity Analysis')
        
        # Results menu
        results_menu = menubar.addMenu('&Results')
        results_menu.addAction('&View Results')
        results_menu.addAction('&Export Data')
        results_menu.addAction('&Generate Report')
        
        # Tools menu
        tools_menu = menubar.addMenu('&Tools')
        tools_menu.addAction('&Correlations')
        tools_menu.addAction('&Unit Converter')
        tools_menu.addAction('&PVT Calculator')
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        help_menu.addAction('&Documentation')
        help_menu.addAction('&About')
        
    def create_toolbar(self):
        """Create main toolbar"""
        
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Add actions
        new_action = toolbar.addAction('New')
        open_action = toolbar.addAction('Open')
        save_action = toolbar.addAction('Save')
        
        toolbar.addSeparator()
        
        run_action = toolbar.addAction('Run')
        run_action.triggered.connect(self.run_simulation)
        
        stop_action = toolbar.addAction('Stop')
        
        toolbar.addSeparator()
        
        report_action = toolbar.addAction('Report')
        export_action = toolbar.addAction('Export')
        
    def create_left_panel(self):
        """Create left panel with project tree"""
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Project tree
        tree_label = QLabel("Project Explorer")
        tree_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(tree_label)
        
        self.project_tree = QTreeWidget()
        self.project_tree.setHeaderHidden(True)
        
        # Add items
        root = QTreeWidgetItem(self.project_tree, ["Field Development Project"])
        
        network = QTreeWidgetItem(root, ["Network"])
        QTreeWidgetItem(network, ["Configuration"])
        QTreeWidgetItem(network, ["Nodes: 15"])
        QTreeWidgetItem(network, ["Connections: 14"])
        
        wells = QTreeWidgetItem(root, ["Wells (7)"])
        for i in range(1, 8):
            well = QTreeWidgetItem(wells, [f"WELL-{i}"])
            QTreeWidgetItem(well, ["Properties"])
            QTreeWidgetItem(well, ["IPR"])
            QTreeWidgetItem(well, ["Equipment"])
        
        pipes = QTreeWidgetItem(root, ["Pipelines (14)"])
        QTreeWidgetItem(pipes, ["Flowlines (7)"])
        QTreeWidgetItem(pipes, ["Manifolds (2)"])
        QTreeWidgetItem(pipes, ["Trunk Lines (3)"])
        QTreeWidgetItem(pipes, ["Export (2)"])
        
        fluids = QTreeWidgetItem(root, ["Fluids"])
        QTreeWidgetItem(fluids, ["Black Oil"])
        QTreeWidgetItem(fluids, ["PVT Data"])
        
        results = QTreeWidgetItem(root, ["Results"])
        QTreeWidgetItem(results, ["Last Run: Not executed"])
        
        root.setExpanded(True)
        wells.setExpanded(True)
        
        layout.addWidget(self.project_tree)
        
        # Properties panel
        props_label = QLabel("Properties")
        props_label.setStyleSheet("font-weight: bold; padding: 5px; margin-top: 10px;")
        layout.addWidget(props_label)
        
        self.properties_table = QTableWidget(6, 2)
        self.properties_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.properties_table.horizontalHeader().setStretchLastSection(True)
        self.properties_table.verticalHeader().setVisible(False)
        
        # Add sample properties
        self.properties_table.setItem(0, 0, QTableWidgetItem("Type"))
        self.properties_table.setItem(0, 1, QTableWidgetItem("Network"))
        self.properties_table.setItem(1, 0, QTableWidgetItem("Status"))
        self.properties_table.setItem(1, 1, QTableWidgetItem("Not Solved"))
        
        layout.addWidget(self.properties_table)
        
        return panel
        
    def create_right_panel(self):
        """Create right panel with quick tools"""
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Quick actions
        actions_label = QLabel("Quick Actions")
        actions_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(actions_label)
        
        btn_validate = QPushButton("Validate Model")
        btn_run = QPushButton("Run Simulation")
        btn_run.clicked.connect(self.run_simulation)
        btn_results = QPushButton("View Results")
        btn_report = QPushButton("Generate Report")
        
        for btn in [btn_validate, btn_run, btn_results, btn_report]:
            btn.setMinimumHeight(30)
            layout.addWidget(btn)
        
        layout.addSpacing(20)
        
        # Simulation monitor
        monitor_label = QLabel("Simulation Monitor")
        monitor_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(monitor_label)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.monitor_text = QTextEdit()
        self.monitor_text.setReadOnly(True)
        self.monitor_text.setMaximumHeight(200)
        self.monitor_text.append("System ready...")
        layout.addWidget(self.monitor_text)
        
        # Statistics
        stats_label = QLabel("System Statistics")
        stats_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(stats_label)
        
        stats_text = QTextEdit()
        stats_text.setReadOnly(True)
        stats_text.setMaximumHeight(150)
        stats_text.setHtml("""
        <b>Current Model:</b><br>
        • Wells: 7<br>
        • Pipelines: 14<br>
        • Nodes: 15<br>
        • Fluid: Black Oil<br>
        <br>
        <b>Last Simulation:</b><br>
        • Status: Not Run<br>
        • Time: --:--<br>
        • Iterations: ---
        """)
        layout.addWidget(stats_text)
        
        layout.addStretch()
        
        return panel
        
    def create_network_tab(self):
        """Create network configuration tab"""
        
        network_widget = QWidget()
        layout = QVBoxLayout(network_widget)
        
        # Network schematic area
        schematic_group = QGroupBox("Network Schematic")
        schematic_layout = QVBoxLayout(schematic_group)
        
        # Create matplotlib figure for network display
        self.network_figure = Figure(figsize=(12, 8), facecolor='#2e2e2e')
        self.network_canvas = FigureCanvas(self.network_figure)
        schematic_layout.addWidget(self.network_canvas)
        
        # Toolbar for network
        network_toolbar = QToolBar()
        network_toolbar.addAction("Add Well")
        network_toolbar.addAction("Add Node")
        network_toolbar.addAction("Add Pipe")
        network_toolbar.addSeparator()
        network_toolbar.addAction("Zoom In")
        network_toolbar.addAction("Zoom Out")
        network_toolbar.addAction("Pan")
        network_toolbar.addAction("Select")
        
        schematic_layout.addWidget(network_toolbar)
        layout.addWidget(schematic_group)
        
        # Draw initial network
        self.draw_network_schematic()
        
        self.main_tabs.addTab(network_widget, "Network")
        
    def create_wells_tab(self):
        """Create wells configuration tab"""
        
        wells_widget = QWidget()
        layout = QVBoxLayout(wells_widget)
        
        # Wells table
        wells_group = QGroupBox("Well Configuration")
        wells_layout = QVBoxLayout(wells_group)
        
        # Toolbar
        wells_toolbar = QToolBar()
        wells_toolbar.addAction("Add Well")
        wells_toolbar.addAction("Delete Well")
        wells_toolbar.addAction("Import")
        wells_toolbar.addAction("Export")
        wells_layout.addWidget(wells_toolbar)
        
        # Table
        self.wells_table = QTableWidget(7, 8)
        self.wells_table.setHorizontalHeaderLabels([
            "Well Name", "Type", "X (km)", "Y (km)", "Depth (m)",
            "Pressure (bar)", "PI (m³/h/bar)", "Status"
        ])
        
        # Add sample data
        wells_data = [
            ["WELL-A1", "Producer", "-3.0", "2.0", "3500", "320", "2.5", "Active"],
            ["WELL-A2", "Producer", "-3.0", "0.0", "3400", "310", "2.3", "Active"],
            ["WELL-A3", "Producer", "-3.0", "-2.0", "3300", "300", "2.1", "Active"],
            ["WELL-B1", "Producer", "-1.0", "2.5", "3600", "330", "2.8", "Active"],
            ["WELL-B2", "Producer", "-1.0", "0.5", "3550", "325", "2.6", "Active"],
            ["WELL-B3", "Producer", "-1.0", "-1.5", "3450", "315", "2.4", "Active"],
            ["WELL-B4", "Producer", "-1.0", "-3.0", "3400", "305", "2.2", "Active"],
        ]
        
        for row, data in enumerate(wells_data):
            for col, value in enumerate(data):
                item = QTableWidgetItem(value)
                if col == 7:  # Status column
                    if value == "Active":
                        item.setForeground(QColor(0, 255, 0))
                    else:
                        item.setForeground(QColor(255, 0, 0))
                self.wells_table.setItem(row, col, item)
        
        self.wells_table.resizeColumnsToContents()
        wells_layout.addWidget(self.wells_table)
        
        layout.addWidget(wells_group)
        
        # Well details section
        details_group = QGroupBox("Well Details")
        details_layout = QFormLayout(details_group)
        
        self.well_name_edit = QLineEdit("WELL-A1")
        self.well_type_combo = QComboBox()
        self.well_type_combo.addItems(["Producer", "Injector", "Observation"])
        self.well_reservoir_pressure = QDoubleSpinBox()
        self.well_reservoir_pressure.setRange(0, 1000)
        self.well_reservoir_pressure.setValue(320)
        self.well_reservoir_pressure.setSuffix(" bar")
        
        details_layout.addRow("Well Name:", self.well_name_edit)
        details_layout.addRow("Type:", self.well_type_combo)
        details_layout.addRow("Reservoir Pressure:", self.well_reservoir_pressure)
        
        # IPR section
        ipr_group = QGroupBox("Inflow Performance")
        ipr_layout = QVBoxLayout(ipr_group)
        
        ipr_type = QComboBox()
        ipr_type.addItems(["Vogel", "Linear PI", "Forchheimer", "Table"])
        ipr_layout.addWidget(ipr_type)
        
        # IPR plot
        ipr_figure = Figure(figsize=(6, 4), facecolor='#2e2e2e')
        ipr_canvas = FigureCanvas(ipr_figure)
        ipr_ax = ipr_figure.add_subplot(111, facecolor='#2e2e2e')
        
        # Sample IPR curve
        pwf = np.linspace(0, 320, 100)
        q = 2.5 * (320 - pwf)
        ipr_ax.plot(q, pwf, 'g-', linewidth=2)
        ipr_ax.set_xlabel('Flow Rate (m³/h)', color='white')
        ipr_ax.set_ylabel('Bottom Hole Pressure (bar)', color='white')
        ipr_ax.set_title('IPR Curve', color='white')
        ipr_ax.grid(True, alpha=0.3)
        ipr_ax.tick_params(colors='white')
        for spine in ipr_ax.spines.values():
            spine.set_color('white')
        
        ipr_layout.addWidget(ipr_canvas)
        details_layout.addRow(ipr_group)
        
        layout.addWidget(details_group)
        
        self.main_tabs.addTab(wells_widget, "Wells")
        
    def create_pipes_tab(self):
        """Create pipes configuration tab"""
        
        pipes_widget = QWidget()
        layout = QVBoxLayout(pipes_widget)
        
        # Pipes table
        pipes_group = QGroupBox("Pipeline Configuration")
        pipes_layout = QVBoxLayout(pipes_group)
        
        # Toolbar
        pipes_toolbar = QToolBar()
        pipes_toolbar.addAction("Add Pipeline")
        pipes_toolbar.addAction("Delete Pipeline")
        pipes_toolbar.addAction("Calculate")
        pipes_layout.addWidget(pipes_toolbar)
        
        # Table
        self.pipes_table = QTableWidget(10, 7)
        self.pipes_table.setHorizontalHeaderLabels([
            "Pipeline ID", "From", "To", "Length (m)", 
            "Diameter (m)", "Roughness (mm)", "Type"
        ])
        
        # Add sample data
        pipes_data = [
            ["P-001", "WELL-A1", "MANIFOLD-A", "2500", "0.25", "0.05", "Flowline"],
            ["P-002", "WELL-A2", "MANIFOLD-A", "2000", "0.25", "0.05", "Flowline"],
            ["P-003", "WELL-A3", "MANIFOLD-A", "2500", "0.25", "0.05", "Flowline"],
            ["P-004", "WELL-B1", "MANIFOLD-B", "3200", "0.25", "0.05", "Flowline"],
            ["P-005", "WELL-B2", "MANIFOLD-B", "1400", "0.25", "0.05", "Flowline"],
            ["P-006", "WELL-B3", "MANIFOLD-B", "2200", "0.25", "0.05", "Flowline"],
            ["P-007", "WELL-B4", "MANIFOLD-B", "4200", "0.25", "0.05", "Flowline"],
            ["P-008", "MANIFOLD-A", "PLATFORM", "5000", "0.40", "0.05", "Trunk"],
            ["P-009", "MANIFOLD-B", "PLATFORM", "3200", "0.40", "0.05", "Trunk"],
            ["P-010", "PLATFORM", "EXPORT", "2000", "0.50", "0.05", "Export"],
        ]
        
        for row, data in enumerate(pipes_data):
            for col, value in enumerate(data):
                self.pipes_table.setItem(row, col, QTableWidgetItem(value))
        
        self.pipes_table.resizeColumnsToContents()
        pipes_layout.addWidget(self.pipes_table)
        
        layout.addWidget(pipes_group)
        
        # Pipeline profile
        profile_group = QGroupBox("Pipeline Profile")
        profile_layout = QVBoxLayout(profile_group)
        
        profile_figure = Figure(figsize=(10, 4), facecolor='#2e2e2e')
        profile_canvas = FigureCanvas(profile_figure)
        profile_ax = profile_figure.add_subplot(111, facecolor='#2e2e2e')
        
        # Sample elevation profile
        distance = np.linspace(0, 5000, 100)
        elevation = -2500 - 500 * np.sin(distance/1000)
        profile_ax.plot(distance, elevation, 'b-', linewidth=2)
        profile_ax.fill_between(distance, elevation, -3000, alpha=0.3, color='blue')
        profile_ax.set_xlabel('Distance (m)', color='white')
        profile_ax.set_ylabel('Elevation (m)', color='white')
        profile_ax.set_title('Pipeline Elevation Profile', color='white')
        profile_ax.grid(True, alpha=0.3)
        profile_ax.tick_params(colors='white')
        for spine in profile_ax.spines.values():
            spine.set_color('white')
        
        profile_layout.addWidget(profile_canvas)
        layout.addWidget(profile_group)
        
        self.main_tabs.addTab(pipes_widget, "Pipelines")
        
    def create_fluids_tab(self):
        """Create fluids configuration tab"""
        
        fluids_widget = QWidget()
        layout = QVBoxLayout(fluids_widget)
        
        # Fluid model selection
        model_group = QGroupBox("Fluid Model")
        model_layout = QFormLayout(model_group)
        
        fluid_model = QComboBox()
        fluid_model.addItems(["Black Oil", "Compositional", "Gas Condensate", "Dry Gas"])
        fluid_model.setCurrentText("Black Oil")
        model_layout.addRow("Model Type:", fluid_model)
        
        layout.addWidget(model_group)
        
        # Black oil properties
        props_group = QGroupBox("Black Oil Properties")
        props_layout = QGridLayout(props_group)
        
        # Create input fields
        props = [
            ("Oil Density (kg/m³):", 780.0, 0, 0),
            ("Water Density (kg/m³):", 1025.0, 0, 2),
            ("Gas Density (kg/m³):", 15.0, 1, 0),
            ("Oil Viscosity (cP):", 0.8, 1, 2),
            ("Water Viscosity (cP):", 1.1, 2, 0),
            ("Gas Viscosity (cP):", 0.018, 2, 2),
            ("Oil Fraction:", 0.75, 3, 0),
            ("Water Fraction:", 0.20, 3, 2),
            ("Gas Fraction:", 0.05, 4, 0),
            ("GOR (sm³/sm³):", 150.0, 4, 2),
            ("Bubble Point (bar):", 280.0, 5, 0),
            ("API Gravity:", 32.5, 5, 2),
        ]
        
        self.fluid_inputs = {}
        for label, value, row, col in props:
            lbl = QLabel(label)
            spin = QDoubleSpinBox()
            spin.setRange(0, 10000)
            spin.setDecimals(3)
            spin.setValue(value)
            props_layout.addWidget(lbl, row, col)
            props_layout.addWidget(spin, row, col + 1)
            self.fluid_inputs[label] = spin
        
        layout.addWidget(props_group)
        
        # PVT curves
        pvt_group = QGroupBox("PVT Curves")
        pvt_layout = QVBoxLayout(pvt_group)
        
        pvt_figure = Figure(figsize=(12, 6), facecolor='#2e2e2e')
        pvt_canvas = FigureCanvas(pvt_figure)
        
        # Create subplots
        ax1 = pvt_figure.add_subplot(131, facecolor='#2e2e2e')
        ax2 = pvt_figure.add_subplot(132, facecolor='#2e2e2e')
        ax3 = pvt_figure.add_subplot(133, facecolor='#2e2e2e')
        
        # Sample PVT data
        pressure = np.linspace(50, 400, 100)
        
        # Bo curve
        bo = 1.0 + 0.0002 * (280 - pressure) * (pressure < 280)
        ax1.plot(pressure, bo, 'g-', linewidth=2)
        ax1.set_xlabel('Pressure (bar)', color='white')
        ax1.set_ylabel('Bo (rm³/sm³)', color='white')
        ax1.set_title('Oil FVF', color='white')
        ax1.grid(True, alpha=0.3)
        
        # Rs curve
        rs = np.minimum(150, 0.5 * pressure)
        ax2.plot(pressure, rs, 'b-', linewidth=2)
        ax2.set_xlabel('Pressure (bar)', color='white')
        ax2.set_ylabel('Rs (sm³/sm³)', color='white')
        ax2.set_title('Solution GOR', color='white')
        ax2.grid(True, alpha=0.3)
        
        # Viscosity curve
        visc = 0.8 + 0.001 * (pressure - 280) * (pressure > 280)
        ax3.plot(pressure, visc, 'r-', linewidth=2)
        ax3.set_xlabel('Pressure (bar)', color='white')
        ax3.set_ylabel('Viscosity (cP)', color='white')
        ax3.set_title('Oil Viscosity', color='white')
        ax3.grid(True, alpha=0.3)
        
        # Style all axes
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        pvt_figure.tight_layout()
        pvt_layout.addWidget(pvt_canvas)
        
        layout.addWidget(pvt_group)
        
        self.main_tabs.addTab(fluids_widget, "Fluids")
        
    def create_simulation_tab(self):
        """Create simulation settings tab"""
        
        sim_widget = QWidget()
        layout = QVBoxLayout(sim_widget)
        
        # Simulation type
        type_group = QGroupBox("Simulation Type")
        type_layout = QVBoxLayout(type_group)
        
        self.sim_steady = QRadioButton("Steady State")
        self.sim_steady.setChecked(True)
        self.sim_transient = QRadioButton("Transient")
        
        type_layout.addWidget(self.sim_steady)
        type_layout.addWidget(self.sim_transient)
        
        layout.addWidget(type_group)
        
        # Solver settings
        solver_group = QGroupBox("Solver Settings")
        solver_layout = QFormLayout(solver_group)
        
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(1, 1000)
        self.max_iterations.setValue(100)
        
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setDecimals(8)
        self.tolerance.setRange(1e-10, 1)
        self.tolerance.setValue(1e-6)
        
        self.relaxation = QDoubleSpinBox()
        self.relaxation.setRange(0.1, 2.0)
        self.relaxation.setValue(1.0)
        self.relaxation.setSingleStep(0.1)
        
        solver_layout.addRow("Max Iterations:", self.max_iterations)
        solver_layout.addRow("Tolerance:", self.tolerance)
        solver_layout.addRow("Relaxation Factor:", self.relaxation)
        
        layout.addWidget(solver_group)
        
        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout(advanced_group)
        
        self.check_heat_transfer = QCheckBox("Include Heat Transfer")
        self.check_surge = QCheckBox("Surge Analysis")
        self.check_optimization = QCheckBox("Enable Optimization")
        
        advanced_layout.addWidget(self.check_heat_transfer)
        advanced_layout.addWidget(self.check_surge)
        advanced_layout.addWidget(self.check_optimization)
        
        layout.addWidget(advanced_group)
        
        # Run controls
        controls_group = QGroupBox("Simulation Control")
        controls_layout = QHBoxLayout(controls_group)
        
        self.btn_validate = QPushButton("Validate Model")
        self.btn_validate.setMinimumHeight(40)
        
        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.btn_run.clicked.connect(self.run_simulation)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        
        controls_layout.addWidget(self.btn_validate)
        controls_layout.addWidget(self.btn_run)
        controls_layout.addWidget(self.btn_stop)
        
        layout.addWidget(controls_group)
        
        # Convergence monitor
        convergence_group = QGroupBox("Convergence Monitor")
        convergence_layout = QVBoxLayout(convergence_group)
        
        self.convergence_figure = Figure(figsize=(8, 4), facecolor='#2e2e2e')
        self.convergence_canvas = FigureCanvas(self.convergence_figure)
        self.convergence_ax = self.convergence_figure.add_subplot(111, facecolor='#2e2e2e')
        
        self.convergence_ax.set_xlabel('Iteration', color='white')
        self.convergence_ax.set_ylabel('Residual', color='white')
        self.convergence_ax.set_title('Convergence History', color='white')
        self.convergence_ax.set_yscale('log')
        self.convergence_ax.grid(True, alpha=0.3)
        self.convergence_ax.tick_params(colors='white')
        for spine in self.convergence_ax.spines.values():
            spine.set_color('white')
        
        convergence_layout.addWidget(self.convergence_canvas)
        
        layout.addWidget(convergence_group)
        layout.addStretch()
        
        self.main_tabs.addTab(sim_widget, "Simulation")
        
    def create_results_tab(self):
        """Create results visualization tab"""
        
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        
        # Results toolbar
        results_toolbar = QToolBar()
        results_toolbar.addAction("Refresh")
        results_toolbar.addAction("Export")
        results_toolbar.addAction("Print")
        results_toolbar.addSeparator()
        
        # View selector
        view_label = QLabel("View:")
        view_combo = QComboBox()
        view_combo.addItems(["Network Overview", "Pressure Profile", 
                           "Flow Distribution", "Well Performance",
                           "3D Visualization", "Time Series"])
        
        results_toolbar.addWidget(view_label)
        results_toolbar.addWidget(view_combo)
        
        layout.addWidget(results_toolbar)
        
        # Results display area
        self.results_figure = Figure(figsize=(14, 10), facecolor='#2e2e2e')
        self.results_canvas = FigureCanvas(self.results_figure)
        layout.addWidget(self.results_canvas)
        
        # Summary statistics
        summary_group = QGroupBox("Summary Statistics")
        summary_layout = QHBoxLayout(summary_group)
        
        # Create summary cards
        cards = [
            ("Total Production", "2,660 m³/h", "#4CAF50"),
            ("System Pressure", "120-330 bar", "#2196F3"),
            ("Water Cut", "20.0%", "#FF9800"),
            ("GOR", "150 sm³/sm³", "#9C27B0"),
            ("Active Wells", "7 / 7", "#00BCD4"),
            ("Efficiency", "96.5%", "#8BC34A"),
        ]
        
        for title, value, color in cards:
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background-color: {color};
                    border-radius: 8px;
                    padding: 10px;
                }}
            """)
            card_layout = QVBoxLayout(card)
            
            title_label = QLabel(title)
            title_label.setStyleSheet("color: white; font-size: 12px;")
            value_label = QLabel(value)
            value_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
            
            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            
            summary_layout.addWidget(card)
        
        layout.addWidget(summary_group)
        
        self.main_tabs.addTab(results_widget, "Results")
        
    def create_reports_tab(self):
        """Create reports tab"""
        
        reports_widget = QWidget()
        layout = QVBoxLayout(reports_widget)
        
        # Report toolbar
        report_toolbar = QToolBar()
        report_toolbar.addAction("Generate Report")
        report_toolbar.addAction("Export PDF")
        report_toolbar.addAction("Export Excel")
        report_toolbar.addAction("Print")
        layout.addWidget(report_toolbar)
        
        # Report viewer
        self.report_viewer = QTextEdit()
        self.report_viewer.setReadOnly(True)
        
        # Add sample report
        self.report_viewer.setHtml("""
        <h1 style='color: #2196F3;'>Production Analysis Report</h1>
        <h2>Executive Summary</h2>
        <p>The offshore field production network analysis has been completed successfully.
        The system demonstrates stable operation with optimal flow distribution across
        all production wells.</p>
        
        <h2>Key Findings</h2>
        <ul>
            <li>Total production rate: 2,660 m³/h (63,840 m³/day)</li>
            <li>System efficiency: 96.5%</li>
            <li>All wells operating within design parameters</li>
            <li>No flow assurance issues identified</li>
        </ul>
        
        <h2>Well Performance</h2>
        <table border='1' style='border-collapse: collapse; width: 100%;'>
            <tr style='background-color: #2196F3; color: white;'>
                <th>Well</th><th>Production (m³/h)</th><th>Pressure (bar)</th><th>Status</th>
            </tr>
            <tr><td>WELL-A1</td><td>445</td><td>320</td><td style='color: green;'>Active</td></tr>
            <tr><td>WELL-A2</td><td>425</td><td>310</td><td style='color: green;'>Active</td></tr>
            <tr><td>WELL-A3</td><td>405</td><td>300</td><td style='color: green;'>Active</td></tr>
            <tr><td>WELL-B1</td><td>465</td><td>330</td><td style='color: green;'>Active</td></tr>
            <tr><td>WELL-B2</td><td>455</td><td>325</td><td style='color: green;'>Active</td></tr>
            <tr><td>WELL-B3</td><td>435</td><td>315</td><td style='color: green;'>Active</td></tr>
            <tr><td>WELL-B4</td><td>430</td><td>305</td><td style='color: green;'>Active</td></tr>
        </table>
        
        <h2>Recommendations</h2>
        <ol>
            <li>Continue current production strategy</li>
            <li>Monitor WELL-A3 for potential decline</li>
            <li>Consider workover for production enhancement in Q3</li>
            <li>Evaluate artificial lift options for mature wells</li>
        </ol>
        
        <p><i>Report generated: {}</i></p>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        layout.addWidget(self.report_viewer)
        
        self.main_tabs.addTab(reports_widget, "Reports")
        
    def draw_network_schematic(self):
        """Draw the network schematic"""
        
        self.network_figure.clear()
        ax = self.network_figure.add_subplot(111, facecolor='#2e2e2e')
        
        # Set limits
        ax.set_xlim(-5, 6)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Define network
        wells = {
            'WELL-A1': (-3, 2), 'WELL-A2': (-3, 0), 'WELL-A3': (-3, -2),
            'WELL-B1': (-1, 2.5), 'WELL-B2': (-1, 0.5), 'WELL-B3': (-1, -1.5),
            'WELL-B4': (-1, -3)
        }
        
        facilities = {
            'MANIFOLD-A': (-2, 0), 'MANIFOLD-B': (0, 0.5),
            'PLATFORM': (2, 0), 'SEPARATOR': (3.5, 0), 'EXPORT': (5, 0)
        }
        
        # Draw connections
        connections = [
            ('WELL-A1', 'MANIFOLD-A'), ('WELL-A2', 'MANIFOLD-A'), 
            ('WELL-A3', 'MANIFOLD-A'), ('WELL-B1', 'MANIFOLD-B'),
            ('WELL-B2', 'MANIFOLD-B'), ('WELL-B3', 'MANIFOLD-B'),
            ('WELL-B4', 'MANIFOLD-B'), ('MANIFOLD-A', 'PLATFORM'),
            ('MANIFOLD-B', 'PLATFORM'), ('PLATFORM', 'SEPARATOR'),
            ('SEPARATOR', 'EXPORT')
        ]
        
        all_nodes = {**wells, **facilities}
        
        for start, end in connections:
            x1, y1 = all_nodes[start]
            x2, y2 = all_nodes[end]
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.6)
        
        # Draw wells
        for name, (x, y) in wells.items():
            circle = Circle((x, y), 0.15, facecolor='#00ff00', 
                          edgecolor='white', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y-0.3, name, ha='center', fontsize=8, color='white')
        
        # Draw facilities
        for name, (x, y) in facilities.items():
            if 'MANIFOLD' in name:
                rect = FancyBboxPatch((x-0.2, y-0.15), 0.4, 0.3,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#ffcc00', edgecolor='white')
                ax.add_patch(rect)
            elif name == 'PLATFORM':
                rect = Rectangle((x-0.3, y-0.2), 0.6, 0.4,
                               facecolor='#888888', edgecolor='white')
                ax.add_patch(rect)
            elif name == 'SEPARATOR':
                circle = Circle((x, y), 0.2, facecolor='#4488ff', 
                              edgecolor='white')
                ax.add_patch(circle)
            else:  # EXPORT
                rect = Rectangle((x-0.2, y-0.2), 0.4, 0.4,
                               facecolor='#ff4444', edgecolor='white')
                ax.add_patch(rect)
            
            ax.text(x, y+0.4, name, ha='center', fontsize=9, 
                   color='white', weight='bold')
        
        # Title
        ax.text(0.5, 3.5, 'Field Network Schematic', ha='center', 
               fontsize=16, color='white', weight='bold')
        
        self.network_canvas.draw()
        
    def run_simulation(self):
        """Run the simulation"""
        
        self.monitor_text.append("\n" + "="*50)
        self.monitor_text.append("Starting simulation...")
        self.status_bar.showMessage("Running simulation...")
        
        # Update progress
        self.progress_bar.setValue(10)
        QApplication.processEvents()
        
        try:
            # Create network
            self.monitor_text.append("Building network model...")
            network = pipeline_sim.Network()
            
            # Add wells from table
            well_nodes = {}
            for row in range(self.wells_table.rowCount()):
                name = self.wells_table.item(row, 0).text()
                pressure = float(self.wells_table.item(row, 5).text())
                
                node = network.add_node(name, pipeline_sim.NodeType.SOURCE)
                network.set_pressure(node, pressure * 1e5)
                well_nodes[name] = node
                
            self.progress_bar.setValue(20)
            QApplication.processEvents()
            
            # Add facilities
            manifold_a = network.add_node("MANIFOLD-A", pipeline_sim.NodeType.JUNCTION)
            manifold_b = network.add_node("MANIFOLD-B", pipeline_sim.NodeType.JUNCTION)
            platform = network.add_node("PLATFORM", pipeline_sim.NodeType.JUNCTION)
            separator = network.add_node("SEPARATOR", pipeline_sim.NodeType.JUNCTION)
            export = network.add_node("EXPORT", pipeline_sim.NodeType.SINK)
            network.set_pressure(export, 120e5)
            
            facility_nodes = {
                "MANIFOLD-A": manifold_a,
                "MANIFOLD-B": manifold_b,
                "PLATFORM": platform,
                "SEPARATOR": separator,
                "EXPORT": export
            }
            
            self.progress_bar.setValue(30)
            QApplication.processEvents()
            
            # Add pipes from table
            self.monitor_text.append("Adding pipelines...")
            all_nodes = {**well_nodes, **facility_nodes}
            
            for row in range(self.pipes_table.rowCount()):
                from_name = self.pipes_table.item(row, 1).text()
                to_name = self.pipes_table.item(row, 2).text()
                length = float(self.pipes_table.item(row, 3).text())
                diameter = float(self.pipes_table.item(row, 4).text())
                
                if from_name in all_nodes and to_name in all_nodes:
                    network.add_pipe(f"P{row+1}", all_nodes[from_name], 
                                   all_nodes[to_name], length, diameter)
            
            self.progress_bar.setValue(40)
            QApplication.processEvents()
            
            # Create fluid
            self.monitor_text.append("Setting fluid properties...")
            fluid = pipeline_sim.FluidProperties()
            fluid.oil_fraction = self.fluid_inputs["Oil Fraction:"].value()
            fluid.water_fraction = self.fluid_inputs["Water Fraction:"].value()
            fluid.gas_fraction = self.fluid_inputs["Gas Fraction:"].value()
            fluid.oil_density = self.fluid_inputs["Oil Density (kg/m³):"].value()
            fluid.water_density = self.fluid_inputs["Water Density (kg/m³):"].value()
            fluid.gas_density = self.fluid_inputs["Gas Density (kg/m³):"].value()
            fluid.oil_viscosity = self.fluid_inputs["Oil Viscosity (cP):"].value() / 1000
            fluid.water_viscosity = self.fluid_inputs["Water Viscosity (cP):"].value() / 1000
            fluid.gas_viscosity = self.fluid_inputs["Gas Viscosity (cP):"].value() / 1000
            
            self.progress_bar.setValue(50)
            QApplication.processEvents()
            
            # Create solver
            self.monitor_text.append("Initializing solver...")
            solver = pipeline_sim.SteadyStateSolver(network, fluid)
            
            if hasattr(solver, 'config'):
                config = solver.config
                config.max_iterations = self.max_iterations.value()
                config.tolerance = self.tolerance.value()
                config.relaxation_factor = self.relaxation.value()
                config.verbose = True
            
            self.progress_bar.setValue(60)
            QApplication.processEvents()
            
            # Run simulation
            self.monitor_text.append("Solving network equations...")
            self.monitor_text.append(f"Max iterations: {self.max_iterations.value()}")
            self.monitor_text.append(f"Tolerance: {self.tolerance.value()}")
            
            result = solver.solve()
            
            self.progress_bar.setValue(90)
            QApplication.processEvents()
            
            if result.converged:
                self.monitor_text.append(f"✓ Converged in {result.iterations} iterations")
                self.monitor_text.append(f"Final residual: {result.residual:.2e}")
                
                # Store results
                self.simulation_results = result
                
                # Update results display
                self.update_results_display()
                
                self.status_bar.showMessage("Simulation completed successfully")
            else:
                self.monitor_text.append("✗ Simulation did not converge")
                self.status_bar.showMessage("Simulation failed to converge")
            
        except Exception as e:
            self.monitor_text.append(f"✗ Error: {str(e)}")
            self.status_bar.showMessage("Simulation error")
            QMessageBox.critical(self, "Simulation Error", str(e))
        
        finally:
            self.progress_bar.setValue(100)
            QApplication.processEvents()
            
            # Re-enable controls
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            
    def update_results_display(self):
        """Update the results visualization"""
        
        if not self.simulation_results:
            return
            
        # Clear figure
        self.results_figure.clear()
        
        # Create subplots
        gs = self.results_figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        ax1 = self.results_figure.add_subplot(gs[0, 0], facecolor='#2e2e2e')
        ax2 = self.results_figure.add_subplot(gs[0, 1], facecolor='#2e2e2e')
        ax3 = self.results_figure.add_subplot(gs[1, 0], facecolor='#2e2e2e')
        ax4 = self.results_figure.add_subplot(gs[1, 1], facecolor='#2e2e2e')
        
        # 1. Well production rates
        wells = []
        flows = []
        for pipe_name, flow in self.simulation_results.pipe_flow_rates.items():
            if pipe_name.startswith("P") and "WELL" in pipe_name:
                well_name = pipe_name.split("_")[0]
                wells.append(well_name)
                flows.append(flow * 3600)
        
        ax1.bar(wells[:7], flows[:7], color='#4CAF50', alpha=0.8)
        ax1.set_xlabel('Wells', color='white')
        ax1.set_ylabel('Production Rate (m³/h)', color='white')
        ax1.set_title('Well Production Rates', color='white')
        ax1.tick_params(colors='white')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Pressure distribution
        nodes = []
        pressures = []
        for node_name, pressure in self.simulation_results.node_pressures.items():
            nodes.append(node_name[:8])  # Truncate names
            pressures.append(pressure / 1e5)
        
        ax2.plot(nodes[:10], pressures[:10], 'o-', color='#2196F3', linewidth=2)
        ax2.set_xlabel('Nodes', color='white')
        ax2.set_ylabel('Pressure (bar)', color='white')
        ax2.set_title('Node Pressures', color='white')
        ax2.tick_params(colors='white')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Flow distribution pie
        group_a_flow = sum(flows[:3]) if len(flows) >= 3 else 0
        group_b_flow = sum(flows[3:7]) if len(flows) >= 7 else 0
        
        if group_a_flow > 0 or group_b_flow > 0:
            ax3.pie([group_a_flow, group_b_flow], labels=['Group A', 'Group B'],
                   colors=['#00ff00', '#00ffff'], autopct='%1.1f%%',
                   textprops={'color': 'white'})
            ax3.set_title('Production by Group', color='white')
        
        # 4. System overview text
        ax4.axis('off')
        total_production = sum(flows) if flows else 0
        summary_text = f"""System Summary:
        
Total Production: {total_production:.0f} m³/h
Active Wells: 7
Converged: Yes
Iterations: {self.simulation_results.iterations}

Export Pressure: 120 bar
Max Well Pressure: 330 bar
System Efficiency: 96.5%"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=12, color='white', verticalalignment='top',
                fontfamily='monospace')
        
        # Style all axes
        for ax in [ax1, ax2, ax3]:
            for spine in ax.spines.values():
                spine.set_color('white')
        
        self.results_canvas.draw()
        
        # Switch to results tab
        self.main_tabs.setCurrentIndex(5)
        
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        
        dark_stylesheet = """
        QMainWindow {
            background-color: #1e1e1e;
        }
        
        QWidget {
            background-color: #2e2e2e;
            color: #ffffff;
        }
        
        QTabWidget::pane {
            border: 1px solid #3e3e3e;
            background-color: #2e2e2e;
        }
        
        QTabBar::tab {
            background-color: #3e3e3e;
            color: #ffffff;
            padding: 8px 16px;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background-color: #4e4e4e;
            border-bottom: 2px solid #2196F3;
        }
        
        QGroupBox {
            border: 1px solid #4e4e4e;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        
        QPushButton {
            background-color: #4e4e4e;
            border: 1px solid #5e5e5e;
            border-radius: 4px;
            padding: 6px 12px;
            color: #ffffff;
        }
        
        QPushButton:hover {
            background-color: #5e5e5e;
        }
        
        QPushButton:pressed {
            background-color: #3e3e3e;
        }
        
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #3e3e3e;
            border: 1px solid #4e4e4e;
            border-radius: 3px;
            padding: 4px;
            color: #ffffff;
        }
        
        QTableWidget {
            background-color: #2e2e2e;
            alternate-background-color: #3e3e3e;
            gridline-color: #4e4e4e;
        }
        
        QHeaderView::section {
            background-color: #3e3e3e;
            color: #ffffff;
            padding: 4px;
            border: 1px solid #4e4e4e;
        }
        
        QTreeWidget {
            background-color: #2e2e2e;
            alternate-background-color: #3e3e3e;
        }
        
        QTextEdit {
            background-color: #2e2e2e;
            border: 1px solid #4e4e4e;
        }
        
        QProgressBar {
            background-color: #3e3e3e;
            border: 1px solid #4e4e4e;
            border-radius: 3px;
            text-align: center;
        }
        
        QProgressBar::chunk {
            background-color: #2196F3;
            border-radius: 3px;
        }
        
        QMenuBar {
            background-color: #2e2e2e;
        }
        
        QMenuBar::item:selected {
            background-color: #4e4e4e;
        }
        
        QMenu {
            background-color: #2e2e2e;
            border: 1px solid #4e4e4e;
        }
        
        QMenu::item:selected {
            background-color: #4e4e4e;
        }
        
        QStatusBar {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        QToolBar {
            background-color: #2e2e2e;
            border: none;
            spacing: 3px;
        }
        
        QToolButton {
            background-color: #3e3e3e;
            border: 1px solid #4e4e4e;
            border-radius: 3px;
            padding: 4px;
            margin: 2px;
        }
        
        QToolButton:hover {
            background-color: #4e4e4e;
        }
        
        QScrollBar:vertical {
            background-color: #2e2e2e;
            width: 12px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #4e4e4e;
            min-height: 20px;
            border-radius: 6px;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            background: none;
        }
        """
        
        self.setStyleSheet(dark_stylesheet)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application properties
    app.setApplicationName("Pipeline-Sim Professional")
    app.setOrganizationName("Pipeline Analytics Corp")
    
    # Create and show main window
    window = ProductionAnalysisSystem()
    window.showMaximized()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()