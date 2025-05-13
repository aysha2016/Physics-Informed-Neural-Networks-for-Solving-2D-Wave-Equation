# Physics-Informed Neural Network for 2D Wave Equation

This repository implements a Physics-Informed Neural Network (PINN) to solve the 2D wave equation using deep learning. The implementation incorporates physical constraints directly into the loss function, allowing the model to learn solutions that satisfy both data-driven and physics-based principles.

## Problem Description

The 2D wave equation is given by:

∂²u/∂t² = c²∂²u/∂x²

where:
- u(x,t) is the wave displacement
- c is the wave speed
- x is the spatial coordinate
- t is the temporal coordinate

The implementation includes:
- Initial conditions: u(x,0) = f(x) and ∂u/∂t(x,0) = g(x)
- Boundary conditions: u(0,t) = u(L,t) = 0 (Dirichlet boundary conditions)

## Project Structure

```
.
├── models/
│   └── model.py          # PINN model implementation
├── utils/
│   ├── losses.py         # Loss functions including PDE, IC, and BC
│   └── training.py       # Training utilities and visualization
├── train.py             # Main training script
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PhysicsInformedPINN_2D_Wave.git
cd PhysicsInformedPINN_2D_Wave
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To train the model, simply run:
```bash
python train.py
```

The script will:
1. Train the PINN model
2. Display training progress
3. Plot the training history
4. Visualize the final solution
5. Save the trained model

## Model Architecture

The PINN consists of:
- Input layer: 2 neurons (x, t coordinates)
- Hidden layers: 4 layers with 128 neurons each
- Output layer: 1 neuron (wave displacement u)
- Activation: Tanh

## Loss Function

The total loss function combines:
1. PDE loss: Enforces the wave equation
2. Initial condition loss: Enforces u(x,0) = f(x) and ∂u/∂t(x,0) = g(x)
3. Boundary condition loss: Enforces u(0,t) = u(L,t) = 0

## Training

The model is trained using:
- Optimizer: Adam
- Learning rate: 1e-3
- Training points:
  - 1000 interior points
  - 100 boundary points
  - 100 initial condition points
- Domain: x ∈ [0,1], t ∈ [0,1]

## Visualization

The training script generates:
1. Training history plots showing the evolution of different loss components
2. Solution visualization showing u(x,t) as a function of space and time

## Customization

You can modify various parameters in `train.py`:
- Model architecture (number of layers, neurons)
- Loss weights (lambda_pde, lambda_ic, lambda_bc)
- Training parameters (epochs, learning rate)
- Domain size and number of training points

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```
@software{pinn_wave_equation,
  author = {Your Name},
  title = {Physics-Informed Neural Network for 2D Wave Equation},
  year = {2024},
  url = {https://github.com/yourusername/PhysicsInformedPINN_2D_Wave}
}
``` 