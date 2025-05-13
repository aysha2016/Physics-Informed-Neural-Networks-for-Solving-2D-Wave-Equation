import torch
import torch.nn as nn

class WaveEquationLoss:
    def __init__(self, c=1.0, lambda_pde=1.0, lambda_ic=1.0, lambda_bc=1.0):
        """
        Loss function for the 2D Wave Equation
        
        Args:
            c (float): Wave speed
            lambda_pde (float): Weight for PDE loss
            lambda_ic (float): Weight for initial condition loss
            lambda_bc (float): Weight for boundary condition loss
        """
        self.c = c
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
        
    def pde_loss(self, model, x, t):
        """Compute the PDE residual loss"""
        _, _, u_tt, _, u_xx = model.compute_derivatives(x, t)
        residual = u_tt - self.c**2 * u_xx
        return torch.mean(residual**2)
    
    def initial_condition_loss(self, model, x, t0):
        """
        Compute the initial condition loss (u(x,0) = f(x) and u_t(x,0) = g(x))
        
        Args:
            model: PINN model
            x: Spatial coordinates
            t0: Initial time (usually 0)
        """
        # Initial displacement condition
        u, u_t, _, _, _ = model.compute_derivatives(x, t0)
        
        # Example initial conditions (can be modified)
        f = torch.sin(torch.pi * x)  # Initial displacement
        g = torch.zeros_like(x)      # Initial velocity
        
        ic_loss = torch.mean((u - f)**2) + torch.mean((u_t - g)**2)
        return ic_loss
    
    def boundary_condition_loss(self, model, t, x_left, x_right):
        """
        Compute the boundary condition loss (e.g., u(0,t) = u(L,t) = 0)
        
        Args:
            model: PINN model
            t: Temporal coordinates
            x_left: Left boundary spatial coordinate
            x_right: Right boundary spatial coordinate
        """
        # Left boundary
        u_left, _, _, _, _ = model.compute_derivatives(x_left, t)
        
        # Right boundary
        u_right, _, _, _, _ = model.compute_derivatives(x_right, t)
        
        # Example: Dirichlet boundary conditions (can be modified)
        bc_loss = torch.mean(u_left**2) + torch.mean(u_right**2)
        return bc_loss
    
    def __call__(self, model, x, t, x_left, x_right, t0):
        """
        Compute the total loss
        
        Args:
            model: PINN model
            x: Interior spatial coordinates
            t: Interior temporal coordinates
            x_left: Left boundary spatial coordinate
            x_right: Right boundary spatial coordinate
            t0: Initial time
        """
        # PDE loss
        pde = self.pde_loss(model, x, t)
        
        # Initial condition loss
        ic = self.initial_condition_loss(model, x, t0)
        
        # Boundary condition loss
        bc = self.boundary_condition_loss(model, t, x_left, x_right)
        
        # Total loss
        total_loss = (self.lambda_pde * pde + 
                     self.lambda_ic * ic + 
                     self.lambda_bc * bc)
        
        return total_loss, {'pde': pde.item(), 'ic': ic.item(), 'bc': bc.item()}
