import numpy as np
import matplotlib.pyplot as plt

class SimplePipeline:
    def __init__(self, length, diameter, roughness=0.045e-3):
        self.length = length
        self.diameter = diameter
        self.roughness = roughness
        
    def calculate_pressure_drop(self, flow_rate, density=850, viscosity=0.01):
        """Calculate pressure drop using Darcy-Weisbach equation"""
        area = np.pi * (self.diameter/2)**2
        velocity = flow_rate / area
        reynolds = density * velocity * self.diameter / viscosity
        
        # Friction factor (Colebrook-White approximation)
        if reynolds < 2300:
            f = 64 / reynolds
        else:
            f = 0.0791 / reynolds**0.25  # Blasius approximation for smooth pipes
            
        # Pressure drop in Pa
        dp = f * (self.length / self.diameter) * (density * velocity**2 / 2)
        return dp

# Example usage
print("Simple Pipeline Calculation")
print("-" * 40)

pipeline = SimplePipeline(length=1000, diameter=0.3)
flow_rate = 0.1  # m³/s

dp = pipeline.calculate_pressure_drop(flow_rate)
print(f"Flow rate: {flow_rate} m³/s")
print(f"Pressure drop: {dp/1e5:.2f} bar")
print(f"Pressure drop per km: {dp/1e5 * 1000/pipeline.length:.2f} bar/km")