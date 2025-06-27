# AI_GENERATED: Real-time monitoring and SCADA integration
# Generated on: 2025-06-27

# ===== monitoring/realtime_dashboard.py =====
"""
Real-Time Pipeline Monitoring Dashboard

Web-based dashboard for live pipeline monitoring with SCADA integration
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils

import pipeline_sim as ps
from pipeline_sim.ml_integration import DigitalTwin, AnomalyDetector


class PipelineMonitor:
    """Real-time pipeline monitoring system"""
    
    def __init__(self, network_file: str):
        # Load network
        self.network, self.fluid = ps.load_network(network_file)
        
        # Initialize digital twin
        self.digital_twin = DigitalTwin()
        self.digital_twin.initialize(self.network, self.fluid)
        
        # Initialize anomaly detector
        self.anomaly_detector = AnomalyDetector()
        self.anomaly_detector.load("models/anomaly_detector.pkl")
        
        # Data storage
        self.historical_data = {
            'timestamps': [],
            'measurements': {},
            'estimates': {},
            'anomalies': [],
            'alarms': []
        }
        
        # Alarm thresholds
        self.alarm_thresholds = {
            'pressure_high': 100e5,     # 100 bar
            'pressure_low': 5e5,        # 5 bar
            'velocity_high': 10.0,      # 10 m/s
            'flow_deviation': 0.1,      # 10% deviation
            'leak_threshold': 0.01      # 10 l/s
        }
        
        # SCADA connections
        self.scada_connections = {}
        
    def connect_scada(self, scada_config: Dict):
        """Connect to SCADA system"""
        # OPC UA connection
        if scada_config['protocol'] == 'opcua':
            from opcua import Client
            
            client = Client(scada_config['url'])
            client.connect()
            
            # Subscribe to tags
            self.scada_connections['opcua'] = {
                'client': client,
                'tags': scada_config['tags']
            }
            
        # Modbus connection
        elif scada_config['protocol'] == 'modbus':
            from pymodbus.client.sync import ModbusTcpClient
            
            client = ModbusTcpClient(
                scada_config['host'],
                port=scada_config.get('port', 502)
            )
            client.connect()
            
            self.scada_connections['modbus'] = {
                'client': client,
                'registers': scada_config['registers']
            }
            
    def read_scada_data(self) -> Dict:
        """Read current data from SCADA"""
        measurements = {
            'pressures': {},
            'flows': {},
            'temperatures': {},
            'valve_positions': {},
            'pump_speeds': {}
        }
        
        # OPC UA
        if 'opcua' in self.scada_connections:
            client = self.scada_connections['opcua']['client']
            tags = self.scada_connections['opcua']['tags']
            
            for tag_name, node_id in tags.items():
                node = client.get_node(node_id)
                value = node.get_value()
                
                # Parse tag name to determine type
                if 'pressure' in tag_name.lower():
                    measurements['pressures'][tag_name] = value * 1e5  # Convert to Pa
                elif 'flow' in tag_name.lower():
                    measurements['flows'][tag_name] = value / 3600  # m³/h to m³/s
                elif 'temp' in tag_name.lower():
                    measurements['temperatures'][tag_name] = value + 273.15  # °C to K
                    
        # Modbus
        if 'modbus' in self.scada_connections:
            client = self.scada_connections['modbus']['client']
            registers = self.scada_connections['modbus']['registers']
            
            for reg_name, address in registers.items():
                result = client.read_holding_registers(address, 2)
                if result:
                    # Convert to float (assuming 32-bit float over 2 registers)
                    value = struct.unpack('>f', struct.pack('>HH', *result.registers))[0]
                    
                    if 'pressure' in reg_name:
                        measurements['pressures'][reg_name] = value * 1e5
                    elif 'flow' in reg_name:
                        measurements['flows'][reg_name] = value / 3600
                        
        return measurements
        
    def update_state(self, timestamp: float):
        """Update system state with new measurements"""
        # Read SCADA data
        measurements = self.read_scada_data()
        
        # Update digital twin
        self.digital_twin.update_with_measurements(
            measurements['pressures'],
            measurements['flows'],
            timestamp
        )
        
        # Get estimated state
        estimated_state = self.digital_twin.estimate_state()
        
        # Detect anomalies
        anomalies = self.digital_twin.detect_discrepancies()
        
        # Check alarms
        alarms = self.check_alarms(estimated_state, measurements)
        
        # Store data
        self.historical_data['timestamps'].append(timestamp)
        self.historical_data['measurements'][timestamp] = measurements
        self.historical_data['estimates'][timestamp] = estimated_state
        self.historical_data['anomalies'].extend(anomalies)
        self.historical_data['alarms'].extend(alarms)
        
        # Limit historical data size
        max_points = 3600  # 1 hour at 1Hz
        if len(self.historical_data['timestamps']) > max_points:
            self.historical_data['timestamps'].pop(0)
            oldest_time = min(self.historical_data['measurements'].keys())
            del self.historical_data['measurements'][oldest_time]
            del self.historical_data['estimates'][oldest_time]
            
        return estimated_state, anomalies, alarms
        
    def check_alarms(self, state, measurements) -> List[Dict]:
        """Check for alarm conditions"""
        alarms = []
        current_time = datetime.now()
        
        # Pressure alarms
        for node_id, pressure in state.node_pressures.items():
            if pressure > self.alarm_thresholds['pressure_high']:
                alarms.append({
                    'timestamp': current_time,
                    'type': 'HIGH_PRESSURE',
                    'location': node_id,
                    'value': pressure / 1e5,
                    'threshold': self.alarm_thresholds['pressure_high'] / 1e5,
                    'severity': 'HIGH',
                    'message': f'High pressure at {node_id}: {pressure/1e5:.1f} bar'
                })
            elif pressure < self.alarm_thresholds['pressure_low']:
                alarms.append({
                    'timestamp': current_time,
                    'type': 'LOW_PRESSURE',
                    'location': node_id,
                    'value': pressure / 1e5,
                    'threshold': self.alarm_thresholds['pressure_low'] / 1e5,
                    'severity': 'MEDIUM',
                    'message': f'Low pressure at {node_id}: {pressure/1e5:.1f} bar'
                })
                
        # Velocity alarms
        for pipe_id, flow in state.pipe_flows.items():
            pipe = self.network.pipes[pipe_id]
            velocity = abs(flow) / pipe.area()
            
            if velocity > self.alarm_thresholds['velocity_high']:
                alarms.append({
                    'timestamp': current_time,
                    'type': 'HIGH_VELOCITY',
                    'location': pipe_id,
                    'value': velocity,
                    'threshold': self.alarm_thresholds['velocity_high'],
                    'severity': 'MEDIUM',
                    'message': f'High velocity in {pipe_id}: {velocity:.1f} m/s'
                })
                
        # Mass balance alarms (leak detection)
        for node_id, node in self.network.nodes.items():
            if node.type == ps.NodeType.JUNCTION:
                inflow = sum(abs(state.pipe_flows.get(p.id, 0)) 
                           for p in self.network.get_upstream_pipes(node))
                outflow = sum(abs(state.pipe_flows.get(p.id, 0)) 
                            for p in self.network.get_downstream_pipes(node))
                
                imbalance = abs(inflow - outflow)
                if imbalance > self.alarm_thresholds['leak_threshold']:
                    alarms.append({
                        'timestamp': current_time,
                        'type': 'LEAK_DETECTED',
                        'location': node_id,
                        'value': imbalance * 1000,  # l/s
                        'threshold': self.alarm_thresholds['leak_threshold'] * 1000,
                        'severity': 'CRITICAL',
                        'message': f'Possible leak at {node_id}: {imbalance*1000:.1f} l/s imbalance'
                    })
                    
        return alarms
        
    def get_network_status(self) -> Dict:
        """Get current network status summary"""
        if not self.historical_data['estimates']:
            return {}
            
        latest_time = self.historical_data['timestamps'][-1]
        latest_state = self.historical_data['estimates'][latest_time]
        
        # Calculate KPIs
        total_flow = sum(abs(f) for f in latest_state.pipe_flows.values())
        avg_pressure = np.mean(list(latest_state.node_pressures.values()))
        
        # Active alarms
        active_alarms = [a for a in self.historical_data['alarms'] 
                        if (datetime.now() - a['timestamp']).seconds < 300]  # Last 5 min
        
        status = {
            'timestamp': latest_time,
            'kpis': {
                'total_flow': total_flow * 3600,  # m³/h
                'average_pressure': avg_pressure / 1e5,  # bar
                'active_alarms': len(active_alarms),
                'network_efficiency': self.calculate_efficiency()
            },
            'alarms': active_alarms[-10:],  # Last 10 alarms
            'anomalies': self.historical_data['anomalies'][-5:]  # Last 5 anomalies
        }
        
        return status
        
    def calculate_efficiency(self) -> float:
        """Calculate network efficiency metric"""
        # Simplified efficiency based on pressure losses
        if not self.historical_data['estimates']:
            return 0.0
            
        latest_state = list(self.historical_data['estimates'].values())[-1]
        
        total_dp = 0
        total_flow = 0
        
        for pipe_id, pipe in self.network.pipes.items():
            if pipe_id in latest_state.pipe_flows:
                flow = abs(latest_state.pipe_flows[pipe_id])
                
                # Get pressure drop
                p_in = latest_state.node_pressures.get(pipe.upstream.id, 0)
                p_out = latest_state.node_pressures.get(pipe.downstream.id, 0)
                dp = p_in - p_out
                
                total_dp += dp * flow
                total_flow += flow
                
        # Efficiency metric (lower pressure loss = higher efficiency)
        if total_flow > 0:
            specific_loss = total_dp / total_flow
            efficiency = 100 * (1 - specific_loss / 10e5)  # Normalized to 10 bar
            return max(0, min(100, efficiency))
        
        return 0.0


# ===== monitoring/web_dashboard.py =====
"""
Web Dashboard Application

Flask-based real-time dashboard with WebSocket support
"""

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pipeline-sim-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global monitor instance
monitor = None


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/network')
def get_network():
    """Get network topology"""
    if not monitor:
        return jsonify({'error': 'Monitor not initialized'}), 500
        
    nodes = []
    edges = []
    
    # Convert network to graph format
    for node_id, node in monitor.network.nodes.items():
        nodes.append({
            'id': node_id,
            'label': node_id,
            'type': str(node.type).split('.')[-1],
            'x': np.random.rand() * 1000,  # Layout would be calculated
            'y': np.random.rand() * 600
        })
        
    for pipe_id, pipe in monitor.network.pipes.items():
        edges.append({
            'id': pipe_id,
            'source': pipe.upstream.id,
            'target': pipe.downstream.id,
            'label': pipe_id
        })
        
    return jsonify({
        'nodes': nodes,
        'edges': edges
    })


@app.route('/api/status')
def get_status():
    """Get current system status"""
    if not monitor:
        return jsonify({'error': 'Monitor not initialized'}), 500
        
    status = monitor.get_network_status()
    return jsonify(status)


@app.route('/api/trends/<node_id>')
def get_trends(node_id):
    """Get historical trends for a node"""
    if not monitor:
        return jsonify({'error': 'Monitor not initialized'}), 500
        
    # Extract time series data
    times = monitor.historical_data['timestamps'][-100:]  # Last 100 points
    pressures = []
    flows = []
    
    for t in times:
        if t in monitor.historical_data['estimates']:
            state = monitor.historical_data['estimates'][t]
            pressures.append(state.node_pressures.get(node_id, 0) / 1e5)
            
            # Sum flows through node
            node = monitor.network.nodes[node_id]
            node_flow = 0
            for pipe in monitor.network.get_downstream_pipes(node):
                node_flow += abs(state.pipe_flows.get(pipe.id, 0))
            flows.append(node_flow * 3600)  # m³/h
            
    return jsonify({
        'times': times,
        'pressures': pressures,
        'flows': flows
    })


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'data': 'Connected to pipeline monitor'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')


@socketio.on('start_monitoring')
def handle_start_monitoring():
    """Start real-time monitoring"""
    socketio.start_background_task(monitoring_loop)


def monitoring_loop():
    """Background task for continuous monitoring"""
    while True:
        try:
            # Update state
            timestamp = time.time()
            state, anomalies, alarms = monitor.update_state(timestamp)
            
            # Prepare real-time data
            realtime_data = {
                'timestamp': timestamp,
                'pressures': {k: v/1e5 for k, v in state.node_pressures.items()},
                'flows': {k: v*3600 for k, v in state.pipe_flows.items()},
                'alarms': alarms,
                'anomalies': [{'location': a.location, 'type': a.type} 
                            for a in anomalies]
            }
            
            # Emit to all connected clients
            socketio.emit('realtime_update', realtime_data)
            
            # Sleep for update interval
            socketio.sleep(1.0)  # 1 Hz update rate
            
        except Exception as e:
            print(f"Monitoring error: {e}")
            socketio.sleep(5.0)  # Wait before retry


# ===== monitoring/templates/dashboard.html =====
"""
<!DOCTYPE html>
<html>
<head>
    <title>Pipeline-Sim Real-Time Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    
    <!-- CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    
    <style>
        body {
            background-color: #1a1a1a;
            color: #e0e0e0;
            font-family: 'Arial', sans-serif;
        }
        
        .dashboard-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .kpi-card {
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.2s;
        }
        
        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }
        
        .kpi-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #4CAF50;
        }
        
        .alarm-badge {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .network-canvas {
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 10px;
            height: 500px;
        }
        
        .trend-chart {
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 10px;
            padding: 15px;
            height: 300px;
        }
        
        .alarm-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .alarm-item {
            background: #3a3a3a;
            border-left: 4px solid #f44336;
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        
        .alarm-critical {
            border-left-color: #f44336;
        }
        
        .alarm-high {
            border-left-color: #ff9800;
        }
        
        .alarm-medium {
            border-left-color: #ffc107;
        }
        
        .status-indicator {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 10px;
        }
        
        .status-ok {
            background: #4CAF50;
            box-shadow: 0 0 10px #4CAF50;
        }
        
        .status-warning {
            background: #ffc107;
            box-shadow: 0 0 10px #ffc107;
        }
        
        .status-error {
            background: #f44336;
            box-shadow: 0 0 10px #f44336;
        }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <div class="container-fluid">
            <div class="row align-items-center">
                <div class="col-md-6">
                    <h1><i class="fas fa-chart-line"></i> Pipeline-Sim Monitor</h1>
                    <p class="mb-0">Real-Time Pipeline Network Monitoring</p>
                </div>
                <div class="col-md-6 text-end">
                    <span class="status-indicator status-ok"></span>
                    <span id="connection-status">Connected</span>
                    <span class="ms-4" id="current-time"></span>
                </div>
            </div>
        </div>
    </div>
    
    <div class="container-fluid">
        <!-- KPI Row -->
        <div class="row">
            <div class="col-md-3">
                <div class="kpi-card">
                    <h5>Total Flow</h5>
                    <div class="kpi-value" id="kpi-flow">0</div>
                    <small>m³/h</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card">
                    <h5>Avg Pressure</h5>
                    <div class="kpi-value" id="kpi-pressure">0</div>
                    <small>bar</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card">
                    <h5>Network Efficiency</h5>
                    <div class="kpi-value" id="kpi-efficiency">0</div>
                    <small>%</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="kpi-card">
                    <h5>Active Alarms</h5>
                    <div class="kpi-value" id="kpi-alarms">
                        <span class="alarm-badge">0</span>
                    </div>
                    <small>alarms</small>
                </div>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="row mt-4">
            <!-- Network Visualization -->
            <div class="col-md-8">
                <h4>Network Topology</h4>
                <div id="network-container" class="network-canvas"></div>
                
                <!-- Trend Charts -->
                <div class="row mt-4">
                    <div class="col-md-6">
                        <h5>Pressure Trends</h5>
                        <div id="pressure-trend" class="trend-chart"></div>
                    </div>
                    <div class="col-md-6">
                        <h5>Flow Trends</h5>
                        <div id="flow-trend" class="trend-chart"></div>
                    </div>
                </div>
            </div>
            
            <!-- Alarms and Events -->
            <div class="col-md-4">
                <h4>Alarms & Events</h4>
                <div class="alarm-list" id="alarm-list">
                    <!-- Alarms will be added dynamically -->
                </div>
                
                <h4 class="mt-4">System Health</h4>
                <div class="list-group">
                    <div class="list-group-item bg-dark">
                        <i class="fas fa-server"></i> SCADA Connection
                        <span class="float-end text-success">OK</span>
                    </div>
                    <div class="list-group-item bg-dark">
                        <i class="fas fa-brain"></i> Digital Twin
                        <span class="float-end text-success">Active</span>
                    </div>
                    <div class="list-group-item bg-dark">
                        <i class="fas fa-shield-alt"></i> Anomaly Detection
                        <span class="float-end text-success">Running</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.socket.io/4.4.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.0/vis-network.min.js"></script>
    
    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        // Data storage
        let networkData = null;
        let trendData = {
            times: [],
            pressures: {},
            flows: {}
        };
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            updateTime();
            loadNetwork();
            initializeTrends();
            
            // Start monitoring
            socket.emit('start_monitoring');
        });
        
        // Socket event handlers
        socket.on('connect', function() {
            document.getElementById('connection-status').textContent = 'Connected';
            document.querySelector('.status-indicator').className = 'status-indicator status-ok';
        });
        
        socket.on('disconnect', function() {
            document.getElementById('connection-status').textContent = 'Disconnected';
            document.querySelector('.status-indicator').className = 'status-indicator status-error';
        });
        
        socket.on('realtime_update', function(data) {
            updateKPIs(data);
            updateNetwork(data);
            updateTrends(data);
            updateAlarms(data.alarms);
        });
        
        // Load network topology
        async function loadNetwork() {
            try {
                const response = await fetch('/api/network');
                const data = await response.json();
                drawNetwork(data);
            } catch (error) {
                console.error('Failed to load network:', error);
            }
        }
        
        // Draw network using vis.js
        function drawNetwork(data) {
            const container = document.getElementById('network-container');
            
            const options = {
                nodes: {
                    shape: 'dot',
                    size: 30,
                    font: {
                        size: 12,
                        color: '#ffffff'
                    },
                    borderWidth: 2
                },
                edges: {
                    width: 2,
                    color: { color: '#848484' },
                    arrows: 'to',
                    smooth: {
                        type: 'curvedCW',
                        roundness: 0.2
                    }
                },
                physics: {
                    stabilization: true,
                    barnesHut: {
                        gravitationalConstant: -30000,
                        springConstant: 0.04,
                        springLength: 100
                    }
                }
            };
            
            const network = new vis.Network(container, data, options);
            networkData = network;
        }
        
        // Initialize trend charts
        function initializeTrends() {
            // Pressure trend
            const pressureLayout = {
                paper_bgcolor: '#2a2a2a',
                plot_bgcolor: '#2a2a2a',
                font: { color: '#e0e0e0' },
                xaxis: { title: 'Time' },
                yaxis: { title: 'Pressure (bar)' },
                showlegend: true
            };
            
            Plotly.newPlot('pressure-trend', [], pressureLayout);
            
            // Flow trend
            const flowLayout = {
                paper_bgcolor: '#2a2a2a',
                plot_bgcolor: '#2a2a2a',
                font: { color: '#e0e0e0' },
                xaxis: { title: 'Time' },
                yaxis: { title: 'Flow (m³/h)' },
                showlegend: true
            };
            
            Plotly.newPlot('flow-trend', [], flowLayout);
        }
        
        // Update KPIs
        function updateKPIs(data) {
            // Animate value changes
            animateValue('kpi-flow', 
                         parseFloat(document.getElementById('kpi-flow').textContent), 
                         data.flows['total'] || 0, 
                         500);
            
            // Calculate average pressure
            const pressures = Object.values(data.pressures);
            const avgPressure = pressures.reduce((a, b) => a + b, 0) / pressures.length;
            
            animateValue('kpi-pressure', 
                         parseFloat(document.getElementById('kpi-pressure').textContent), 
                         avgPressure, 
                         500);
        }
        
        // Update network visualization
        function updateNetwork(data) {
            if (!networkData) return;
            
            // Update node colors based on pressure
            const nodes = [];
            for (const [nodeId, pressure] of Object.entries(data.pressures)) {
                const color = getPressureColor(pressure);
                nodes.push({
                    id: nodeId,
                    color: {
                        background: color,
                        border: color
                    }
                });
            }
            
            networkData.body.data.nodes.update(nodes);
        }
        
        // Update trend charts
        function updateTrends(data) {
            const time = new Date(data.timestamp * 1000).toLocaleTimeString();
            
            // Keep last 100 points
            if (trendData.times.length > 100) {
                trendData.times.shift();
                for (const key in trendData.pressures) {
                    trendData.pressures[key].shift();
                }
                for (const key in trendData.flows) {
                    trendData.flows[key].shift();
                }
            }
            
            trendData.times.push(time);
            
            // Update pressure data
            for (const [nodeId, pressure] of Object.entries(data.pressures)) {
                if (!trendData.pressures[nodeId]) {
                    trendData.pressures[nodeId] = [];
                }
                trendData.pressures[nodeId].push(pressure);
            }
            
            // Update Plotly charts
            const pressureTraces = Object.entries(trendData.pressures)
                .slice(0, 5)  // Show top 5 nodes
                .map(([nodeId, values]) => ({
                    x: trendData.times,
                    y: values,
                    name: nodeId,
                    type: 'scatter',
                    mode: 'lines'
                }));
            
            Plotly.react('pressure-trend', pressureTraces);
        }
        
        // Update alarms
        function updateAlarms(alarms) {
            const alarmList = document.getElementById('alarm-list');
            
            // Add new alarms
            alarms.forEach(alarm => {
                const alarmDiv = document.createElement('div');
                alarmDiv.className = `alarm-item alarm-${alarm.severity.toLowerCase()}`;
                alarmDiv.innerHTML = `
                    <div class="d-flex justify-content-between">
                        <strong>${alarm.type}</strong>
                        <small>${new Date(alarm.timestamp).toLocaleTimeString()}</small>
                    </div>
                    <p class="mb-0">${alarm.message}</p>
                `;
                
                alarmList.insertBefore(alarmDiv, alarmList.firstChild);
            });
            
            // Limit to 20 alarms
            while (alarmList.children.length > 20) {
                alarmList.removeChild(alarmList.lastChild);
            }
            
            // Update alarm count
            document.getElementById('kpi-alarms').textContent = alarms.length;
        }
        
        // Helper functions
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleString();
            setTimeout(updateTime, 1000);
        }
        
        function getPressureColor(pressure) {
            // Color scale from blue (low) to red (high)
            const min = 0;
            const max = 100;
            const normalized = (pressure - min) / (max - min);
            
            if (normalized < 0.5) {
                // Blue to green
                const r = 0;
                const g = Math.floor(255 * normalized * 2);
                const b = Math.floor(255 * (1 - normalized * 2));
                return `rgb(${r}, ${g}, ${b})`;
            } else {
                // Green to red
                const r = Math.floor(255 * (normalized - 0.5) * 2);
                const g = Math.floor(255 * (1 - (normalized - 0.5) * 2));
                const b = 0;
                return `rgb(${r}, ${g}, ${b})`;
            }
        }
        
        function animateValue(id, start, end, duration) {
            const element = document.getElementById(id);
            const range = end - start;
            const minTimer = 50;
            let stepTime = Math.abs(Math.floor(duration / range));
            stepTime = Math.max(stepTime, minTimer);
            const steps = Math.floor(duration / stepTime);
            const increment = range / steps;
            let current = start;
            
            const timer = setInterval(() => {
                current += increment;
                element.textContent = current.toFixed(1);
                if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
                    element.textContent = end.toFixed(1);
                    clearInterval(timer);
                }
            }, stepTime);
        }
    </script>
</body>
</html>
"""


# ===== monitoring/opc_connector.py =====
# AI_GENERATED: OPC UA connector for SCADA integration
"""
OPC UA Connector for SCADA Integration

Provides seamless integration with industrial control systems
"""

from opcua import Client, ua
from opcua.common.subscription_handler import SubHandler
import asyncio
import logging
from typing import Dict, List, Callable
import json


class OPCUAConnector:
    """OPC UA client for pipeline monitoring"""
    
    def __init__(self, server_url: str, namespace: str = "Pipeline-Sim"):
        self.server_url = server_url
        self.namespace = namespace
        self.client = None
        self.subscription = None
        self.handlers = {}
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Connect to OPC UA server"""
        try:
            self.client = Client(self.server_url)
            self.client.connect()
            
            # Get namespace index
            self.ns_idx = self.client.get_namespace_index(self.namespace)
            
            self.logger.info(f"Connected to OPC UA server: {self.server_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from server"""
        if self.subscription:
            self.subscription.delete()
            
        if self.client:
            self.client.disconnect()
            
    def browse_nodes(self, start_node=None) -> List[Dict]:
        """Browse available nodes"""
        if not self.client:
            return []
            
        if start_node is None:
            start_node = self.client.get_root_node()
            
        nodes = []
        for child in start_node.get_children():
            node_info = {
                'node_id': child.nodeid.to_string(),
                'name': child.get_display_name().Text,
                'class': child.get_node_class().name,
                'value': None
            }
            
            # Try to read value
            if child.get_node_class() == ua.NodeClass.Variable:
                try:
                    node_info['value'] = child.get_value()
                except:
                    pass
                    
            nodes.append(node_info)
            
        return nodes
        
    def read_tag(self, tag_path: str):
        """Read single tag value"""
        try:
            node = self.client.get_node(tag_path)
            return node.get_value()
        except Exception as e:
            self.logger.error(f"Failed to read tag {tag_path}: {e}")
            return None
            
    def write_tag(self, tag_path: str, value):
        """Write value to tag"""
        try:
            node = self.client.get_node(tag_path)
            node.set_value(value)
            return True
        except Exception as e:
            self.logger.error(f"Failed to write tag {tag_path}: {e}")
            return False
            
    def subscribe_tags(self, tags: List[str], callback: Callable):
        """Subscribe to tag changes"""
        if not self.subscription:
            self.subscription = self.client.create_subscription(500, self)
            
        for tag in tags:
            try:
                node = self.client.get_node(tag)
                handle = self.subscription.subscribe_data_change(node)
                self.handlers[handle] = callback
                self.logger.info(f"Subscribed to {tag}")
            except Exception as e:
                self.logger.error(f"Failed to subscribe to {tag}: {e}")
                
    def datachange_notification(self, node, val, data):
        """Handle data change notifications"""
        if data.monitored_item.handle in self.handlers:
            callback = self.handlers[data.monitored_item.handle]
            callback(node, val)


class SCADATagMapper:
    """Map SCADA tags to pipeline network elements"""
    
    def __init__(self, mapping_file: str):
        with open(mapping_file) as f:
            self.mapping = json.load(f)
            
        self.reverse_mapping = {}
        for category, tags in self.mapping.items():
            for tag_name, tag_info in tags.items():
                self.reverse_mapping[tag_info['opc_path']] = {
                    'category': category,
                    'name': tag_name,
                    'scaling': tag_info.get('scaling', 1.0),
                    'offset': tag_info.get('offset', 0.0),
                    'unit': tag_info.get('unit', '')
                }
                
    def map_to_pipeline(self, opc_path: str, raw_value):
        """Convert SCADA value to pipeline units"""
        if opc_path not in self.reverse_mapping:
            return None
            
        info = self.reverse_mapping[opc_path]
        scaled_value = raw_value * info['scaling'] + info['offset']
        
        return {
            'category': info['category'],
            'name': info['name'],
            'value': scaled_value,
            'unit': info['unit']
        }
        
    def get_opc_paths(self) -> List[str]:
        """Get all OPC paths to subscribe"""
        return list(self.reverse_mapping.keys())


# Example SCADA mapping configuration
SCADA_MAPPING = {
    "pressures": {
        "wellhead_pressure": {
            "opc_path": "ns=2;s=Field.Well01.Pressure",
            "scaling": 100000,  # bar to Pa
            "unit": "Pa"
        },
        "manifold_pressure": {
            "opc_path": "ns=2;s=Field.Manifold.Pressure",
            "scaling": 100000,
            "unit": "Pa"
        }
    },
    "flows": {
        "well_flow": {
            "opc_path": "ns=2;s=Field.Well01.Flow",
            "scaling": 0.000277778,  # m³/h to m³/s
            "unit": "m³/s"
        }
    },
    "temperatures": {
        "process_temp": {
            "opc_path": "ns=2;s=Field.Process.Temperature",
            "offset": 273.15,  # °C to K
            "unit": "K"
        }
    }
}


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline-Sim Monitoring Dashboard')
    parser.add_argument('--network', required=True, help='Network JSON file')
    parser.add_argument('--scada-url', help='OPC UA server URL')
    parser.add_argument('--port', type=int, default=5000, help='Dashboard port')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = PipelineMonitor(args.network)
    
    # Connect to SCADA if URL provided
    if args.scada_url:
        scada_config = {
            'protocol': 'opcua',
            'url': args.scada_url,
            'tags': SCADA_MAPPING
        }
        monitor.connect_scada(scada_config)
    
    # Run Flask app
    socketio.run(app, host='0.0.0.0', port=args.port)
