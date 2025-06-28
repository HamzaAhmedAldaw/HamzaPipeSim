# ===== python/pipeline_sim/correlations.py =====
# AI_GENERATED: Python implementation of flow correlations
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BeggsBrillResults:
    pressure_gradient: float
    liquid_holdup: float
    flow_pattern: int  # 0=segregated, 1=intermittent, 2=distributed, 3=annular
    flow_pattern_name: str


class BeggsBrill:
    """Beggs and Brill multiphase flow correlation"""
    
    FLOW_PATTERNS = {
        0: "Segregated",
        1: "Intermittent", 
        2: "Distributed",
        3: "Annular"
    }
    
    @staticmethod
    def calculate(fluid, pipe, flow_rate: float) -> BeggsBrillResults:
        """Calculate pressure drop using Beggs-Brill correlation"""
        
        # Calculate superficial velocities
        area = pipe.area()
        vsl = flow_rate * (fluid.oil_fraction + fluid.water_fraction) / area
        vsg = flow_rate * fluid.gas_fraction / area
        vm = vsl + vsg
        
        # No-slip liquid holdup
        lambda_l = vsl / vm if vm > 0 else 0
        
        # Froude number
        g = 9.81  # m/s²
        froude = vm**2 / (g * pipe.diameter) if pipe.diameter > 0 else 0
        
        # Determine flow pattern
        flow_pattern = BeggsBrill._determine_flow_pattern(lambda_l, froude)
        
        # Calculate liquid holdup
        holdup = BeggsBrill._calculate_holdup(
            lambda_l, froude, pipe.inclination, flow_pattern
        )
        
        # Calculate mixture density
        rho_l = (fluid.oil_density * fluid.oil_fraction + 
                fluid.water_density * fluid.water_fraction)
        rho_g = fluid.gas_density * 1.225  # Convert to absolute
        rho_m = rho_l * holdup + rho_g * (1 - holdup)
        
        # Calculate mixture viscosity
        mu_l = (fluid.oil_viscosity * fluid.oil_fraction + 
               fluid.water_viscosity * fluid.water_fraction)
        mu_g = fluid.gas_viscosity
        mu_m = mu_l * lambda_l + mu_g * (1 - lambda_l)
        
        # Reynolds number and friction factor
        Re = rho_m * vm * pipe.diameter / mu_m if mu_m > 0 else 0
        f = BeggsBrill._friction_factor(Re, pipe.roughness / pipe.diameter)
        
        # Pressure gradients
        dp_friction = f * rho_m * vm**2 / (2 * pipe.diameter)
        dp_gravity = rho_m * g * np.sin(pipe.inclination)
        
        pressure_gradient = dp_friction + dp_gravity
        
        return BeggsBrillResults(
            pressure_gradient=pressure_gradient,
            liquid_holdup=holdup,
            flow_pattern=flow_pattern,
            flow_pattern_name=BeggsBrill.FLOW_PATTERNS[flow_pattern]
        )
    
    @staticmethod
    def _determine_flow_pattern(lambda_l: float, froude: float) -> int:
        """Determine flow pattern based on flow map"""
        
        # Flow pattern boundaries
        L1 = 316 * lambda_l**0.302
        L2 = 0.0009252 * lambda_l**(-2.4684)
        L3 = 0.10 * lambda_l**(-1.4516)
        L4 = 0.5 * lambda_l**(-6.738)
        
        if lambda_l < 0.01 and froude < L1:
            return 0  # Segregated
        elif lambda_l >= 0.01 and lambda_l < 0.4:
            if froude > L1 and froude <= L2:
                return 1  # Intermittent
            elif froude <= L1:
                return 2  # Distributed
        elif lambda_l >= 0.4:
            if froude > L1 and froude <= L4:
                return 1  # Intermittent
            elif froude <= L1:
                return 2  # Distributed
        
        return 3  # Annular (default)
    
    @staticmethod
    def _calculate_holdup(lambda_l: float, froude: float, 
                         angle: float, flow_pattern: int) -> float:
        """Calculate liquid holdup with inclination correction"""
        
        # Horizontal liquid holdup correlations
        if flow_pattern == 0:  # Segregated
            Hl_0 = 0.98 * lambda_l**0.4846 / froude**0.0868
        elif flow_pattern == 1:  # Intermittent
            Hl_0 = 0.845 * lambda_l**0.5351 / froude**0.0173
        elif flow_pattern == 2:  # Distributed
            Hl_0 = 1.065 * lambda_l**0.5824 / froude**0.0609
        else:  # Annular
            Hl_0 = lambda_l  # No-slip
        
        # Limit holdup to physical bounds
        Hl_0 = np.clip(Hl_0, lambda_l, 1.0)
        
        # Inclination correction
        if abs(angle) < 0.001:  # Horizontal
            return Hl_0
        
        # Calculate C factor
        C = max(0, (1 - lambda_l) * np.log(
            0.011 * lambda_l**(-3.768) * froude**3.539
        ))
        
        # Inclination correction factor
        psi = 1 + C * (np.sin(1.8 * angle) - 
                      0.333 * np.sin(1.8 * angle)**3)
        
        return Hl_0 * psi
    
    @staticmethod
    def _friction_factor(Re: float, rel_roughness: float) -> float:
        """Calculate Fanning friction factor"""
        
        if Re < 2300:  # Laminar
            return 16 / Re if Re > 0 else 0.01
        else:  # Turbulent - Colebrook-White
            # Swamee-Jain approximation
            a = -2 * np.log10(rel_roughness/3.7 + 5.74/Re**0.9)
            return 0.25 / a**2 / 4  # Convert to Fanning


class HagedornBrown:
    """Hagedorn and Brown vertical flow correlation"""
    
    @staticmethod
    def calculate(fluid, pipe, flow_rate: float) -> dict:
        """Calculate pressure drop for vertical flow"""
        
        # This is a placeholder - implement full correlation
        area = pipe.area()
        velocity = flow_rate / area
        
        # Simplified calculation
        rho_m = fluid.mixture_density()
        mu_m = fluid.mixture_viscosity()
        
        Re = rho_m * velocity * pipe.diameter / mu_m
        f = 0.001  # Simplified
        
        dp_friction = f * rho_m * velocity**2 * pipe.length / (2 * pipe.diameter)
        dp_gravity = rho_m * 9.81 * pipe.length * np.sin(pipe.inclination)
        
        return {
            'pressure_drop': dp_friction + dp_gravity,
            'liquid_holdup': 0.8,  # Placeholder
            'flow_regime': 'Bubble'
        }
