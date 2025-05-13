import torch
import torch.optim as optim
from models.model import PINN2DWave
from utils.losses import WaveEquationLoss
from utils.training import WaveEquationTrainer

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = PINN2DWave(
        hidden_layers=[128, 128, 128, 128],  # 4 hidden layers with 128 neurons each
        activation=torch.nn.Tanh()
    )
    
    # Create loss function
    loss_fn = WaveEquationLoss(
        c=1.0,                # Wave speed
        lambda_pde=1.0,       # PDE loss weight
        lambda_ic=10.0,       # Initial condition loss weight
        lambda_bc=10.0        # Boundary condition loss weight
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Create trainer
    trainer = WaveEquationTrainer(model, loss_fn, optimizer, device)
    
    # Training parameters
    epochs = 5000
    n_interior = 1000  # Number of interior points
    n_boundary = 100   # Number of boundary points
    n_initial = 100    # Number of initial condition points
    
    # Domain parameters
    x_min, x_max = 0.0, 1.0  # Spatial domain
    t_min, t_max = 0.0, 1.0  # Temporal domain
    
    # Train the model
    print("Starting training...")
    history = trainer.train(
        epochs=epochs,
        n_interior=n_interior,
        n_boundary=n_boundary,
        n_initial=n_initial,
        x_min=x_min,
        x_max=x_max,
        t_min=t_min,
        t_max=t_max,
        log_interval=100
    )
    
    # Plot training history
    print("\nPlotting training history...")
    trainer.plot_training_history(history)
    
    # Plot final solution
    print("\nPlotting final solution...")
    trainer.plot_solution(
        x_min=x_min,
        x_max=x_max,
        t_min=t_min,
        t_max=t_max,
        n_points=100
    )
    
    # Save the model
    torch.save(model.state_dict(), 'wave_equation_model.pth')
    print("\nModel saved as 'wave_equation_model.pth'")

if __name__ == "__main__":
    main()
