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
        
        self.dx = L / (Nx - 1)
        self.x = np.linspace(0, L, Nx)
        
        # Initialize h with random perturbation
        # self.h = h0 + np.random.uniform(-delta, delta, Nx)
        # Initialize h with a sine wave for controlled instability growth (2 periods across L)
        self.h = h0 + delta * np.sin(2 * np.pi * self.x / (self.L / 10)) + np.random.uniform(-delta / 4, delta / 4, Nx)
        self.h[0] = h0  # Inlet boundary condition

    def compute_q(self):
        """
        Compute volumetric flux q = (h^3 / 3mu) * (rho*g - sigma * d3h/dx3)
        """
        h = self.h
        dx = self.dx

        # Compute 3rd derivative using central differences
        # d3h/dx3[i] = (h[i+2] - 2h[i+1] + 2h[i-1] - h[i-2]) / (2 * dx^3)
        d3h = np.zeros_like(h)
        for i in range(2, self.Nx - 2):
            d3h[i] = (h[i+2] - 2*h[i+1] + 2*h[i-1] - h[i-2]) / (2 * dx**3)

        # Boundary handling for 3rd derivative (simple padding/extrapolation)
        d3h[0:2] = d3h[2]
        d3h[-2:] = d3h[-3]

        q = (h**3 / (3 * self.mu)) * (self.rho * G - self.sigma * d3h)
        return q

    def step(self, dt):
        """
        Update h using the continuity equation: dh/dt + dq/dx = 0
        """
        q = self.compute_q()
        dx = self.dx

        # Compute dq/dx using central differences
        dq_dx = np.zeros_like(q)
        for i in range(1, self.Nx - 1):
            dq_dx[i] = (q[i+1] - q[i-1]) / (2 * dx)

        # Boundary handling for dq/dx
        dq_dx[0] = (q[1] - q[0]) / dx
        dq_dx[-1] = (q[-1] - q[-2]) / dx

        # Update h
        self.h -= dt * dq_dx

        # Apply boundary conditions
        self.h[0] = self.h0  # Fixed inlet
        self.h[-1] = self.h[-2]  # Zero gradient outlet

        # Physical constraint: h cannot be negative
        self.h = np.maximum(self.h, 1e-7)

    def get_h(self):
        return self.h
