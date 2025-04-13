
import torch
import torch.autograd as autograd

def wave_residual(model, x, t, c=1.0):
    u = model(x, t)
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_tt = autograd.grad(u_t, t, torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u), create_graph=True)[0]

    return u_tt - c**2 * u_xx
