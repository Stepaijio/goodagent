import numpy as np
from numba import njit
from src.utils.constants import G

@njit
def _compute_step_jit(h, rho_g, mu, sigma, dx, alpha, dt, h0):
    """
    JIT-optimized step computation.
    Handles the physics of the Kapitza wave evolution.
    """
    nx = h.shape[0]
    
    # 1. Compute 3rd derivative (central differences)
    d3h = np.zeros(nx)
    # Vectorized slice operation for 3rd derivative
    d3h[2:-2] = (h[4:] - 2*h[3:-1] + 2*h[1:-3] - h[:-4]) / (2 * dx**3)
    
    # Boundary handling for 3rd derivative
    d3h[0:2] = d3h[2]
    d3h[-2:] = d3h[-3]
    
    # 2. Compute Fluxes
    # common_factor = h^3 / (3 * mu)
    h3_3mu = (h**3) / (3 * mu)
    q_grav = h3_3mu * rho_g
    q_cap = h3_3mu * (-sigma * d3h)
    
    # 3. Compute dq_grav/dx (Hybrid Upwind/Central)
    dq_grav_dx = np.zeros(nx)
    # Upwind part
    dq_up = np.zeros(nx)
    dq_up[1:] = (q_grav[1:] - q_grav[:-1]) / dx
    dq_up[0] = (q_grav[1] - q_grav[0]) / dx
    
    # Central part
    dq_cen = np.zeros(nx)
    dq_cen[1:-1] = (q_grav[2:] - q_grav[:-2]) / (2 * dx)
    dq_cen[0] = (q_grav[1] - q_grav[0]) / dx
    dq_cen[-1] = (q_grav[-1] - q_grav[-2]) / dx
    
    dq_grav_dx = alpha * dq_cen + (1 - alpha) * dq_up
    
    # 4. Compute dq_cap/dx (Central)
    dq_cap_dx = np.zeros(nx)
    dq_cap_dx[1:-1] = (q_cap[2:] - q_cap[:-2]) / (2 * dx)
    dq_cap_dx[0] = (q_cap[1] - q_cap[0]) / dx
    dq_cap_dx[-1] = (q_cap[-1] - q_cap[-2]) / dx
    
    # 5. Update h
    h -= dt * (dq_grav_dx + dq_cap_dx)
    
    # Boundary conditions
    h[0] = h0
    h[-1] = h[-2]
    
    # Physical constraint: h cannot be negative
    # Using a loop for maximum efficiency in Numba for the clipping
    for i in range(nx):
        if h[i] < 1e-7:
            h[i] = 1e-7
            
    return h

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
        
        # Pre-calculate constant for JIT
        self.rho_g = rho * G
        
        # Initialize h with random perturbation
        self.h = h0 + np.random.uniform(-delta, delta, Nx).astype(np.float64)
        self.h[0] = h0  # Inlet boundary condition

    def step(self, dt):
        """
        Update h using the optimized JIT-compiled function.
        """
        self.h = _compute_step_jit(
            self.h, 
            self.rho_g, 
            self.mu, 
            self.sigma, 
            self.dx, 
            self.alpha, 
            dt, 
            self.h0
        )

    def get_h(self):
        return self.h
