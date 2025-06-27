# ===== examples/digital_twin_demo.py =====
# AI_GENERATED: Digital twin demonstration
"""
Digital Twin Demonstration

Shows real-time monitoring, state estimation, and anomaly detection
using the pipeline digital twin functionality.
"""

import pipeline_sim as ps
from pipeline_sim.ml_integration import DigitalTwin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


class DigitalTwinDemo:
    def __init__(self):
        # Create network
        self.network = self.create_demo_network()
        self.fluid = self.create_fluid()
        
        # Initialize digital twin
        self.twin = DigitalTwin()
        self.twin.initialize(self.network, self.fluid)
        
        # Simulation time
        self.current_time = 0.0
        self.time_step = 1.0  # seconds
        
        # Data storage
        self.time_history = []
        self.measurement_history = []
        self.estimated_history = []
        self.anomaly_history = []
        
    def create_demo_network(self):
        """Create demonstration network"""
        network = ps.Network()
        
        # Simple network: Source -> Pump -> Junction -> Valve -> Sink
        source = network.add_node("Source", ps.NodeType.SOURCE)
        pump_in = network.add_node("Pump_In", ps.NodeType.JUNCTION)
        pump_out = network.add_node("Pump_Out", ps.NodeType.PUMP)
        junction = network.add_node("Junction", ps.NodeType.JUNCTION)
        valve_in = network.add_node("Valve_In", ps.NodeType.JUNCTION)
        valve_out = network.add_node("Valve_Out", ps.NodeType.VALVE)
        sink = network.add_node("Sink", ps.NodeType.SINK)
        
        # Connect with pipes
        network.add_pipe("P1", source, pump_in, 500, 0.3)
        network.add_pipe("P2", pump_in, pump_out, 10, 0.3)
        network.add_pipe("P3", pump_out, junction, 2000, 0.3)
        network.add_pipe("P4", junction, valve_in, 1500, 0.25)
        network.add_pipe("P5", valve_in, valve_out, 10, 0.25)
        network.add_pipe("P6", valve_out, sink, 1000, 0.25)
        
        # Set boundary conditions
        network.set_pressure(source, 20e5)  # 20 bar
        network.set_flow_rate(sink, 0.1)    # 100 l/s
        
        return network
        
    def create_fluid(self):
        """Create fluid properties"""
        fluid = ps.FluidProperties()
        fluid.oil_density = 850
        fluid.oil_viscosity = 0.01
        fluid.oil_fraction = 1.0
        fluid.gas_fraction = 0.0
        fluid.water_fraction = 0.0
        return fluid
        
    def simulate_measurements(self, true_state, noise_level=0.02):
        """Simulate noisy sensor measurements"""
        measurements = {}
        
        # Pressure sensors at key locations
        pressure_sensors = ["Source", "Pump_Out", "Junction", "Sink"]
        flow_sensors = ["P1", "P3", "P6"]
        
        # Add noise to true values
        pressure_measurements = {}
        for node in pressure_sensors:
            if node in true_state.node_pressures:
                true_value = true_state.node_pressures[node]
                noise = np.random.normal(0, noise_level * true_value)
                pressure_measurements[node] = true_value + noise
                
        flow_measurements = {}
        for pipe in flow_sensors:
            if pipe in true_state.pipe_flow_rates:
                true_value = true_state.pipe_flow_rates[pipe]
                noise = np.random.normal(0, noise_level * abs(true_value))
                flow_measurements[pipe] = true_value + noise
                
        return pressure_measurements, flow_measurements
        
    def introduce_anomaly(self, time):
        """Introduce anomalies at specific times"""
        if 50 <= time < 100:
            # Simulate partial valve closure
            return "valve_restriction"
        elif 150 <= time < 200:
            # Simulate small leak
            return "leak"
        elif 250 <= time < 300:
            # Simulate pump degradation
            return "pump_degradation"
        return None
        
    def run_demo(self, duration=300):
        """Run digital twin demonstration"""
        print("Digital Twin Demonstration")
        print("="*50)
        print(f"Simulating {duration} seconds of operation...")
        print("Anomalies will be introduced at:")
        print("  t=50s: Valve restriction")
        print("  t=150s: Pipeline leak")
        print("  t=250s: Pump degradation")
        print()
        
        # Initial steady state
        solver = ps.SteadyStateSolver(self.network, self.fluid)
        true_state = solver.solve()
        
        # Simulation loop
        for step in range(int(duration / self.time_step)):
            self.current_time = step * self.time_step
            
            # Apply anomalies
            anomaly_type = self.introduce_anomaly(self.current_time)
            if anomaly_type:
                true_state = self.apply_anomaly(true_state, anomaly_type)
            
            # Simulate measurements
            pressure_meas, flow_meas = self.simulate_measurements(true_state)
            
            # Update digital twin
            self.twin.update_with_measurements(
                pressure_meas, flow_meas, self.current_time
            )
            
            # Get estimated state
            estimated_state = self.twin.estimate_state()
            
            # Detect anomalies
            discrepancies = self.twin.detect_discrepancies()
            
            # Store history
            self.time_history.append(self.current_time)
            self.measurement_history.append((pressure_meas, flow_meas))
            self.estimated_history.append(estimated_state)
            self.anomaly_history.append(discrepancies)
            
            # Print status every 10 seconds
            if step % 10 == 0:
                self.print_status(estimated_state, discrepancies)
                
        print("\nSimulation complete!")
        self.analyze_results()
        self.plot_results()
        
    def apply_anomaly(self, state, anomaly_type):
        """Apply anomaly to true state"""
        # In practice, would modify network and re-solve
        # Here we simulate the effect
        
        if anomaly_type == "valve_restriction":
            # Increase pressure upstream, decrease flow
            for pipe in ["P4", "P5", "P6"]:
                if pipe in state.pipe_flow_rates:
                    state.pipe_flow_rates[pipe] *= 0.8
            state.node_pressures["Junction"] *= 1.1
            
        elif anomaly_type == "leak":
            # Pressure drop at leak location
            state.node_pressures["Junction"] *= 0.9
            state.pipe_flow_rates["P3"] *= 1.05  # Increased flow to leak
            state.pipe_flow_rates["P4"] *= 0.95  # Decreased downstream flow
            
        elif anomaly_type == "pump_degradation":
            # Reduced pump performance
            state.node_pressures["Pump_Out"] *= 0.85
            for pipe in state.pipe_flow_rates:
                state.pipe_flow_rates[pipe] *= 0.9
                
        return state
        
    def print_status(self, estimated_state, discrepancies):
        """Print current status"""
        print(f"\nTime: {self.current_time:.0f}s")
        
        # Key measurements
        junction_p = estimated_state.node_pressures.get("Junction", 0) / 1e5
        main_flow = estimated_state.pipe_flows.get("P3", 0) * 1000
        
        print(f"  Junction pressure: {junction_p:.1f} bar "
              f"(±{estimated_state.uncertainties.get('Junction', 0)/1e5:.2f})")
        print(f"  Main flow: {main_flow:.1f} l/s "
              f"(±{estimated_state.uncertainties.get('P3', 0)*1000:.2f})")
        
        if discrepancies:
            print("  Anomalies detected:")
            for disc in discrepancies:
                print(f"    - {disc.type} at {disc.location} "
                      f"(severity: {disc.severity:.2f}, confidence: {disc.confidence:.1%})")
                      
    def analyze_results(self):
        """Analyze digital twin performance"""
        print("\n" + "="*50)
        print("DIGITAL TWIN PERFORMANCE ANALYSIS")
        print("="*50)
        
        # Detection performance
        true_anomalies = {
            (50, 100): "valve_restriction",
            (150, 200): "leak",
            (250, 300): "pump_degradation"
        }
        
        detections = {
            "valve_restriction": [],
            "leak": [],
            "pump_degradation": [],
            "false_positive": []
        }
        
        for i, (time, anomalies) in enumerate(zip(self.time_history, self.anomaly_history)):
            if anomalies:
                # Check if true anomaly period
                true_anomaly = None
                for (start, end), atype in true_anomalies.items():
                    if start <= time < end:
                        true_anomaly = atype
                        break
                        
                if true_anomaly:
                    detections[true_anomaly].append(time)
                else:
                    detections["false_positive"].append(time)
                    
        # Print detection statistics
        print("\nAnomaly Detection Results:")
        for (start, end), atype in true_anomalies.items():
            detected = [t for t in detections[atype] if start <= t < end]
            if detected:
                detection_delay = detected[0] - start
                print(f"  {atype}:")
                print(f"    Detection delay: {detection_delay:.0f}s")
                print(f"    Detection rate: {len(detected)/(end-start)*100:.1f}%")
            else:
                print(f"  {atype}: NOT DETECTED")
                
        if detections["false_positive"]:
            print(f"\n  False positives: {len(detections['false_positive'])}")
            
    def plot_results(self):
        """Plot digital twin results"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        # Extract time series data
        times = np.array(self.time_history)
        
        # 1. Junction pressure comparison
        ax = axes[0, 0]
        
        measured_p = []
        estimated_p = []
        uncertainty_p = []
        
        for meas, est in zip(self.measurement_history, self.estimated_history):
            if "Junction" in meas[0]:
                measured_p.append(meas[0]["Junction"] / 1e5)
            else:
                measured_p.append(np.nan)
                
            estimated_p.append(est.node_pressures.get("Junction", 0) / 1e5)
            uncertainty_p.append(est.uncertainties.get("Junction", 0) / 1e5)
            
        estimated_p = np.array(estimated_p)
        uncertainty_p = np.array(uncertainty_p)
        
        ax.plot(times, measured_p, 'o', markersize=3, alpha=0.5, label='Measured')
        ax.plot(times, estimated_p, '-', linewidth=2, label='Estimated')
        ax.fill_between(times, 
                       estimated_p - 2*uncertainty_p,
                       estimated_p + 2*uncertainty_p,
                       alpha=0.2, label='95% CI')
        
        # Mark anomaly periods
        for (start, end), atype in [(50, 100, "Valve"), 
                                   (150, 200, "Leak"), 
                                   (250, 300, "Pump")]:
            ax.axvspan(start, end, alpha=0.1, color='red')
            ax.text((start+end)/2, ax.get_ylim()[1]*0.95, atype,
                   ha='center', fontsize=8)
            
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pressure (bar)')
        ax.set_title('Junction Pressure - State Estimation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Main flow comparison
        ax = axes[0, 1]
        
        measured_q = []
        estimated_q = []
        
        for meas, est in zip(self.measurement_history, self.estimated_history):
            if "P3" in meas[1]:
                measured_q.append(meas[1]["P3"] * 1000)
            else:
                measured_q.append(np.nan)
                
            estimated_q.append(est.pipe_flows.get("P3", 0) * 1000)
            
        ax.plot(times, measured_q, 'o', markersize=3, alpha=0.5, label='Measured')
        ax.plot(times, estimated_q, '-', linewidth=2, label='Estimated')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Flow Rate (l/s)')
        ax.set_title('Main Pipeline Flow - State Estimation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Anomaly detection
        ax = axes[1, 0]
        
        anomaly_scores = []
        for anomalies in self.anomaly_history:
            if anomalies:
                anomaly_scores.append(max(a.severity for a in anomalies))
            else:
                anomaly_scores.append(0)
                
        ax.plot(times, anomaly_scores, '-', linewidth=2, color='red')
        ax.axhline(y=0.01, color='orange', linestyle='--', label='Threshold')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Anomaly Detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. State uncertainty
        ax = axes[1, 1]
        
        avg_uncertainty = []
        for est in self.estimated_history:
            uncertainties = list(est.uncertainties.values())
            if uncertainties:
                avg_uncertainty.append(np.mean(uncertainties) / 1e5)
            else:
                avg_uncertainty.append(0)
                
        ax.plot(times, avg_uncertainty, '-', linewidth=2, color='purple')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Average Uncertainty (bar)')
        ax.set_title('State Estimation Uncertainty')
        ax.grid(True, alpha=0.3)
        
        # 5. Measurement residuals
        ax = axes[2, 0]
        
        pressure_residuals = []
        for meas, est in zip(self.measurement_history, self.estimated_history):
            residuals = []
            for node, p_meas in meas[0].items():
                p_est = est.node_pressures.get(node, p_meas)
                residuals.append(abs(p_meas - p_est) / p_meas)
            if residuals:
                pressure_residuals.append(np.mean(residuals) * 100)
            else:
                pressure_residuals.append(0)
                
        ax.plot(times, pressure_residuals, '-', linewidth=2, color='green')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Residual (%)')
        ax.set_title('Average Measurement Residuals')
        ax.grid(True, alpha=0.3)
        
        # 6. Future prediction
        ax = axes[2, 1]
        
        # Predict 30 seconds into future at t=150
        prediction_time = 150
        if prediction_time in self.time_history:
            idx = self.time_history.index(prediction_time)
            
            # Get predictions
            future_times = np.arange(prediction_time, prediction_time + 30, 1)
            predicted_pressures = []
            
            for dt in range(30):
                future_state = self.twin.predict_future(dt)
                predicted_pressures.append(
                    future_state.node_pressures.get("Junction", 0) / 1e5
                )
                
            # Get actual values (if available)
            actual_pressures = []
            for t in future_times:
                if t in times:
                    idx_t = list(times).index(t)
                    actual_pressures.append(estimated_p[idx_t])
                else:
                    actual_pressures.append(np.nan)
                    
            ax.plot(future_times, predicted_pressures, '--', linewidth=2, 
                   label='Predicted', color='orange')
            ax.plot(future_times, actual_pressures, '-', linewidth=2, 
                   label='Actual', color='blue')
            
            ax.axvline(x=prediction_time, color='black', linestyle=':', 
                      label='Prediction start')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Junction Pressure (bar)')
            ax.set_title('30-Second Future Prediction (from t=150s)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('digital_twin_results.png', dpi=150)
        plt.show()


if __name__ == "__main__":
    # Run digital twin demonstration
    demo = DigitalTwinDemo()
    demo.run_demo(duration=300)
    
    # Additional analysis
    print("\nDigital Twin Capabilities Demonstrated:")
    print("1. Real-time state estimation with Kalman filtering")
    print("2. Anomaly detection using machine learning")
    print("3. Uncertainty quantification")
    print("4. Future state prediction")
    print("5. Sensor fusion and noise filtering")
    print("\nSee digital_twin_results.png for detailed visualizations")