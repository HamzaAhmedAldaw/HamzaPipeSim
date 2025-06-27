import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Pipeline-Sim Demo")
print("=" * 50)

# Simulate pipeline data
length = 5000  # meters
segments = 50
x = np.linspace(0, length, segments)

# Pressure profile (linear drop with some variation)
inlet_pressure = 70  # bar
outlet_pressure = 20  # bar
pressure = np.linspace(inlet_pressure, outlet_pressure, segments)
pressure += np.sin(x/500) * 2  # Add some variation

# Flow data
flow_rate = 0.5  # m³/s
velocity = flow_rate / (np.pi * 0.3**2 / 4)  # Assuming 0.3m diameter

# Create DataFrame
df = pd.DataFrame({
    'Distance (m)': x,
    'Pressure (bar)': pressure,
    'Velocity (m/s)': [velocity] * segments
})

# Display summary
print(f"Pipeline Summary:")
print(f"- Length: {length} m")
print(f"- Inlet Pressure: {inlet_pressure} bar")
print(f"- Outlet Pressure: {outlet_pressure:.1f} bar")
print(f"- Pressure Drop: {inlet_pressure - outlet_pressure:.1f} bar")
print(f"- Flow Rate: {flow_rate} m³/s")
print(f"- Average Velocity: {velocity:.2f} m/s")

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Pressure profile
ax1.plot(x, pressure, 'b-', linewidth=2)
ax1.set_xlabel('Distance (m)')
ax1.set_ylabel('Pressure (bar)')
ax1.set_title('Pipeline Pressure Profile')
ax1.grid(True, alpha=0.3)
ax1.fill_between(x, pressure, alpha=0.3)

# Pressure gradient
gradient = np.gradient(pressure, x) * 1000  # bar/km
ax2.plot(x[1:], gradient[1:], 'r-', linewidth=2)
ax2.set_xlabel('Distance (m)')
ax2.set_ylabel('Pressure Gradient (bar/km)')
ax2.set_title('Pressure Gradient Along Pipeline')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('pipeline_demo.png', dpi=150)
print(f"\nVisualization saved as 'pipeline_demo.png'")
plt.show()

# Save data
df.to_csv('pipeline_data.csv', index=False)
print(f"Data saved as 'pipeline_data.csv'")