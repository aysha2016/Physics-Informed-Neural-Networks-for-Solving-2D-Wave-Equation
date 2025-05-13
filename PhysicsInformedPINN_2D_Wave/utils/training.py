import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class WaveEquationTrainer:
    def __init__(self, model, loss_fn, optimizer, device='cpu'):
        """
        Trainer for the Wave Equation PINN
        
        Args:
            model: PINN model
            loss_fn: Wave equation loss function
            optimizer: PyTorch optimizer
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
    def generate_training_data(self, n_interior=1000, n_boundary=100, n_initial=100,
                             x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        """Generate training points for interior, boundary, and initial conditions"""
        # Interior points
        x = torch.rand(n_interior, 1, device=self.device) * (x_max - x_min) + x_min
        t = torch.rand(n_interior, 1, device=self.device) * (t_max - t_min) + t_min
        
        # Boundary points
        t_boundary = torch.rand(n_boundary, 1, device=self.device) * (t_max - t_min) + t_min
        x_left = torch.ones(n_boundary, 1, device=self.device) * x_min
        x_right = torch.ones(n_boundary, 1, device=self.device) * x_max
        
        # Initial condition points
        x_initial = torch.rand(n_initial, 1, device=self.device) * (x_max - x_min) + x_min
        t_initial = torch.ones(n_initial, 1, device=self.device) * t_min
        
        return x, t, x_left, x_right, t_initial
    
    def train(self, epochs, n_interior=1000, n_boundary=100, n_initial=100,
             x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0, log_interval=100):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            n_interior: Number of interior training points
            n_boundary: Number of boundary training points
            n_initial: Number of initial condition training points
            x_min, x_max: Spatial domain bounds
            t_min, t_max: Temporal domain bounds
            log_interval: Interval for logging training progress
        """
        history = {'total_loss': [], 'pde_loss': [], 'ic_loss': [], 'bc_loss': []}
        
        for epoch in tqdm(range(epochs)):
            # Generate training data
            x, t, x_left, x_right, t0 = self.generate_training_data(
                n_interior, n_boundary, n_initial, x_min, x_max, t_min, t_max
            )
            
            # Compute loss
            loss, loss_components = self.loss_fn(
                self.model, x, t, x_left, x_right, t0
            )
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Log progress
            if epoch % log_interval == 0:
                history['total_loss'].append(loss.item())
                history['pde_loss'].append(loss_components['pde'])
                history['ic_loss'].append(loss_components['ic'])
                history['bc_loss'].append(loss_components['bc'])
                
                print(f"Epoch {epoch}")
                print(f"Total Loss: {loss.item():.6f}")
                print(f"PDE Loss: {loss_components['pde']:.6f}")
                print(f"IC Loss: {loss_components['ic']:.6f}")
                print(f"BC Loss: {loss_components['bc']:.6f}\n")
        
        return history
    
    def plot_solution(self, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0, n_points=100):
        """
        Plot the solution u(x,t) as a surface plot
        
        Args:
            x_min, x_max: Spatial domain bounds
            t_min, t_max: Temporal domain bounds
            n_points: Number of points for plotting
        """
        # Create meshgrid
        x = torch.linspace(x_min, x_max, n_points, device=self.device)
        t = torch.linspace(t_min, t_max, n_points, device=self.device)
        X, T = torch.meshgrid(x, t, indexing='ij')
        
        # Compute solution
        with torch.no_grad():
            X_flat = X.reshape(-1, 1)
            T_flat = T.reshape(-1, 1)
            U = self.model(X_flat, T_flat).reshape(n_points, n_points)
            U = U.cpu().numpy()
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(X.cpu().numpy(), T.cpu().numpy(), U, shading='auto')
        plt.colorbar(label='u(x,t)')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Wave Equation Solution')
        plt.show()
    
    def plot_training_history(self, history):
        """Plot training loss history"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(history['total_loss'])
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 2)
        plt.plot(history['pde_loss'])
        plt.title('PDE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 3)
        plt.plot(history['ic_loss'])
        plt.title('Initial Condition Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 4)
        plt.plot(history['bc_loss'])
        plt.title('Boundary Condition Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.show() 