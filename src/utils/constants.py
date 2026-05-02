# Physical constants and default parameters
import numpy as np

# Constants
G = 9.81  # Acceleration due to gravity [m/s^2]

# Default parameters for a typical fluid (e.g., water-like or oil-like)
DEFAULT_PARAMS = {
    "rho": 1000.0,          # Density [kg/m^3]
    "mu": 0.001,            # Dynamic viscosity [Pa*s] (water: 0.001, oil: 0.01-0.1)
    "sigma": 0.072,         # Surface tension [N/m] (water: 0.072)
    "h0": 0.001,            # Average film thickness [m] (1 mm)
    "L": 0.5,               # Wall height [m]
    "T": 10.0,              # Total observation time [s]
    "delta": 3 * 1e-5,          # Initial perturbation amplitude [m]
    "Nx": 100,              # Number of spatial points
    "dt": 1e-6,             # Time step [s] (Needs to be very small for stability)
}
