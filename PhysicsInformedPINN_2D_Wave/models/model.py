import torch
import torch.nn as nn

class PINN2DWave(nn.Module):
    def __init__(self, hidden_layers=[128, 128, 128], activation=nn.Tanh()):
        """
        Physics-Informed Neural Network for 2D Wave Equation
        
        Args:
            hidden_layers (list): List of integers specifying the number of neurons in each hidden layer
            activation (nn.Module): Activation function to use in hidden layers
        """
        super(PINN2DWave, self).__init__()
        
        # Build the neural network
        layers = []
        input_dim = 2  # (x, t)
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(activation)
        
        # Hidden layers
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(activation)
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, t):
        """
        Forward pass of the network
        
        Args:
            x (torch.Tensor): Spatial coordinates
            t (torch.Tensor): Temporal coordinates
            
        Returns:
            torch.Tensor: Predicted wave displacement u(x,t)
        """
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)
    
    def compute_derivatives(self, x, t):
        """
        Compute all required derivatives for the wave equation
        
        Args:
            x (torch.Tensor): Spatial coordinates
            t (torch.Tensor): Temporal coordinates
            
        Returns:
            tuple: (u, u_t, u_tt, u_x, u_xx) containing the solution and its derivatives
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.forward(x, t)
        
        # Compute first derivatives
        u_t = torch.autograd.grad(u, t, 
                                grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
        u_x = torch.autograd.grad(u, x,
                                grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
        
        # Compute second derivatives
        u_tt = torch.autograd.grad(u_t, t,
                                 grad_outputs=torch.ones_like(u_t),
                                 create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x,
                                 grad_outputs=torch.ones_like(u_x),
                                 create_graph=True)[0]
        
        return u, u_t, u_tt, u_x, u_xx
