
import torch
import torch.nn as nn
import torch.autograd as autograd
from models.model import PINN2DWave
from utils.losses import wave_residual

# Synthetic training data
def create_training_data(N=1000):
    x = torch.rand((N, 1), requires_grad=True)
    t = torch.rand((N, 1), requires_grad=True)
    return x, t

def train(model, epochs=5000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        x, t = create_training_data()
        residual = wave_residual(model, x, t)
        loss = torch.mean(residual**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    model = PINN2DWave()
    train(model)
