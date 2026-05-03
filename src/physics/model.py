import numpy as np
from src.utils.constants import G

class KapitzaModel:
    def __init__(self, rho, mu, sigma, h0, L, Nx, delta):
        self.rho = rho
        self.mu = mu
        self.sigma = sigma
        self.h0 = h0
        self.L = L
        self.Nx = Nx
        self.delta = delta
        self.alpha = 0.96  # Balance between Central (1.0) and Upwind (0.0) for grav flux
        
        self.dx = L / (Nx - 1)
        self.x = np.linspace(0, L, Nx)
        
        # Initialize h with random perturbation
        self.h = h0 + np.random.uniform(-delta, delta, Nx)
        # Initialize h with a sine wave for controlled instability growth (2 periods across L)
        # self.h = h0 + delta * np.sin(2 * np.pi * self.x / (self.L / 5)) + np.random.uniform(-delta / 4, delta / 4, Nx)
        self.h[0] = h0  # Inlet boundary condition

    def compute_q(self):
        """
        Compute volumetric flux components:
        q_grav = (h^3 / 3mu) * rho*g
        q_cap = (h^3 / 3mu) * (-sigma * d3h/dx3)
        """
        h = self.h
        dx = self.dx

        # Compute 3rd derivative using central differences
        d3h = np.zeros_like(h)
        # Vectorized 3rd derivative: (h[i+2] - 2h[i+1] + 2h[i-1] - h[i-2]) / (2 * dx^3)
        d3h[2:-2] = (h[4:] - 2*h[3:-1] + 2*h[1:-3] - h[:-4]) / (2 * dx**3)

        # Boundary handling for 3rd derivative
        d3h[0:2] = d3h[2]
        d3h[-2:] = d3h[-3]

        common_factor = h**3 / (3 * self.mu)
        q_grav = common_factor * (self.rho * G)
        q_cap = common_factor * (-self.sigma * d3h)
        
        return q_grav, q_cap

    def step(self, dt):
        """
        Update h using the continuity equation: dh/dt + dq_grav/dx + dq_cap/dx = 0
        - Gravitational part: Weighted combination of Upwind and Central
        - Capillary part: Central scheme (accuracy)
        """
        q_grav, q_cap = self.compute_q()
        dx = self.dx

        # 1. Gravitational part: Hybrid Central/Upwind
        dq_grav_upwind = np.zeros_like(q_grav)
        dq_grav_upwind[1:] = (q_grav[1:] - q_grav[:-1]) / dx
        dq_grav_upwind[0] = (q_grav[1] - q_grav[0]) / dx

        dq_grav_central = np.zeros_like(q_grav)
        dq_grav_central[1:-1] = (q_grav[2:] - q_grav[:-2]) / (2 * dx)
        dq_grav_central[0] = (q_grav[1] - q_grav[0]) / dx
        dq_grav_central[-1] = (q_grav[-1] - q_grav[-2]) / dx

        # Weighted combination
        dq_grav_dx = self.alpha * dq_grav_central + (1 - self.alpha) * dq_grav_upwind

        # 2. Capillary part: Central differences
        dq_cap_dx = np.zeros_like(q_cap)
        dq_cap_dx[1:-1] = (q_cap[2:] - q_cap[:-2]) / (2 * dx)
        dq_cap_dx[0] = (q_cap[1] - q_cap[0]) / dx
        dq_cap_dx[-1] = (q_cap[-1] - q_cap[-2]) / dx

        # Update h
        self.h -= dt * (dq_grav_dx + dq_cap_dx)

        # Apply boundary conditions
        self.h[0] = self.h0  # Fixed inlet
        self.h[-1] = self.h[-2]  # Zero gradient outlet

        # Physical constraint: h cannot be negative
        self.h = np.maximum(self.h, 1e-7)

    def get_h(self):
        return self.h
