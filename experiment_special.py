from kan import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy import special
import random

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dictionary of special function implementations using scipy.special.
# Each lambda takes a tensor input (with two columns) and returns a tensor computed
# by iterating over the rows and calling the corresponding scipy.special function.
function_implementations = {
    # Jacobi elliptic functions
    "ellipj": lambda x: torch.tensor(
        [special.ellipj(u.item(), m.item())[0] for u, m in zip(x[:, 0], x[:, 1])],
        dtype=torch.float32,
        device=device,
    ),
    "ellipkinc": lambda x: torch.tensor(
        [special.ellipkinc(phi.item(), m.item()) for phi, m in zip(x[:, 0], x[:, 1])],
        dtype=torch.float32,
        device=device,
    ),
    "ellipeinc": lambda x: torch.tensor(
        [special.ellipeinc(phi.item(), m.item()) for phi, m in zip(x[:, 0], x[:, 1])],
        dtype=torch.float32,
        device=device,
    ),
    # Bessel functions
    "jv": lambda x: torch.tensor(
        [special.jv(v.item(), z.item()) for v, z in zip(x[:, 0], x[:, 1])],
        dtype=torch.float32,
        device=device,
    ),
    "yv": lambda x: torch.tensor(
        [special.yv(v.item(), z.item()) for v, z in zip(x[:, 0], x[:, 1])],
        dtype=torch.float32,
        device=device,
    ),
    "kv": lambda x: torch.tensor(
        [special.kv(v.item(), z.item()) for v, z in zip(x[:, 0], x[:, 1])],
        dtype=torch.float32,
        device=device,
    ),
    "iv": lambda x: torch.tensor(
        [special.iv(v.item(), z.item()) for v, z in zip(x[:, 0], x[:, 1])],
        dtype=torch.float32,
        device=device,
    ),
    # Associated Legendre functions
    "lpmv_m_0": lambda x: torch.tensor(
        [
            special.lpmv(0, int(n.item()), x_val.item())
            for x_val, n in zip(x[:, 0], x[:, 1])
        ],
        dtype=torch.float32,
        device=device,
    ),
    "lpmv_m_1": lambda x: torch.tensor(
        [
            special.lpmv(1, int(n.item()), x_val.item())
            for x_val, n in zip(x[:, 0], x[:, 1])
        ],
        dtype=torch.float32,
        device=device,
    ),
    "lpmv_m_2": lambda x: torch.tensor(
        [
            special.lpmv(2, int(n.item()), x_val.item())
            for x_val, n in zip(x[:, 0], x[:, 1])
        ],
        dtype=torch.float32,
        device=device,
    ),
    # Spherical harmonics
    "sph_harm_m_0_n_1": lambda x: torch.tensor(
        [
            special.sph_harm(0, 1, phi.item(), theta.item()).real
            for theta, phi in zip(x[:, 0], x[:, 1])
        ],
        dtype=torch.float32,
        device=device,
    ),
    "sph_harm_m_1_n_1": lambda x: torch.tensor(
        [
            special.sph_harm(1, 1, phi.item(), theta.item()).real
            for theta, phi in zip(x[:, 0], x[:, 1])
        ],
        dtype=torch.float32,
        device=device,
    ),
    "sph_harm_m_0_n_2": lambda x: torch.tensor(
        [
            special.sph_harm(0, 2, phi.item(), theta.item()).real
            for theta, phi in zip(x[:, 0], x[:, 1])
        ],
        dtype=torch.float32,
        device=device,
    ),
    "sph_harm_m_1_n_2": lambda x: torch.tensor(
        [
            special.sph_harm(1, 2, phi.item(), theta.item()).real
            for theta, phi in zip(x[:, 0], x[:, 1])
        ],
        dtype=torch.float32,
        device=device,
    ),
    "sph_harm_m_2_n_2": lambda x: torch.tensor(
        [
            special.sph_harm(2, 2, phi.item(), theta.item()).real
            for theta, phi in zip(x[:, 0], x[:, 1])
        ],
        dtype=torch.float32,
        device=device,
    ),
}

# List of width specifications for the KAN models; one list per function.
widths = [
    [2, 3, 2, 1, 1, 1],
    [2, 2, 1, 1, 1],
    [2, 2, 1, 1],
    [2, 3, 1, 1, 1],
    [2, 2, 2, 1],
    [2, 2, 1],
    [2, 4, 3, 2, 1, 1],
    [2, 2, 1],
    [2, 4, 1],
    [2, 3, 2, 1],
    [2, 1, 1],
    [2, 3, 2, 1],
    [2, 1, 1],
    [2, 2, 1, 1],
    [2, 2, 3, 2, 1],
]

# Iterate over each special function and its corresponding width configuration.
# Iterate over each special function and width configuration
for (name, f), width in zip(function_implementations.items(), widths):
    # Create dataset with input validation
    def safe_f(x):
        try:
            return f(x)
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            return torch.zeros(x.shape[0], 1, device=device)
    
    dataset = create_dataset(safe_f, n_var=2, device=device)
    input_dim = 2

    # Define grid sizes
    grids = np.array([3, 5, 10, 20, 50, 100, 200, 500, 1000])
    train_losses = []
    test_losses = []
    steps = 200
    k = 3

    # Parameter counting function
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # MLP definition
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    # Improved MLP architecture search
    def find_mlp_architecture(input_dim, output_dim, target_params):
        candidates = []
        for depth in range(1, 6):
            for width in [10, 20, 50, 100, 200, 300]:
                hidden = [width] * depth
                params = input_dim*width + width + sum(width*width + width for _ in range(depth-1)) + width*output_dim + output_dim
                candidates.append((abs(params - target_params), hidden))
        candidates.sort()
        return candidates[0][1] if candidates else [100]

    # Store results
    kan_n_params = []
    kan_test_rmse = []
    mlp_n_params = []
    mlp_test_rmse = []

    # Main training loop
    for i in range(len(grids)):
        # Train KAN
        if i == 0:
            model = KAN(width=width, grid=grids[i], k=k, seed=seed, device=device)
        else:
            model = model.refine(grids[i])
        
        results = model.fit(dataset, opt="LBFGS", steps=steps)
        train_losses += results["train_loss"]
        test_losses += results["test_loss"]

        # Store KAN metrics
        kan_n_params.append(count_parameters(model))
        kan_test_rmse.append(results["test_loss"][-1])

        # Train comparable MLP
        hidden_dims = find_mlp_architecture(2, 1, kan_n_params[-1])
        mlp = MLP(2, hidden_dims, 1).to(device)
        
        # MLP training
        optimizer = optim.LBFGS(mlp.parameters(), max_iter=steps, line_search_fn='strong_wolfe')
        criterion = nn.MSELoss()
        
        x_train, y_train = dataset["train_input"], dataset["train_label"]
        x_test, y_test = dataset["test_input"], dataset["test_label"]

        def closure():
            optimizer.zero_grad()
            loss = criterion(mlp(x_train), y_train)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        # Store MLP metrics
        with torch.no_grad():
            test_loss = torch.sqrt(criterion(mlp(x_test), y_test)).item()
            mlp_n_params.append(count_parameters(mlp))
            mlp_test_rmse.append(test_loss)

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='KAN Train')
    plt.plot(test_losses, label='KAN Test')
    plt.yscale('log')
    plt.xlabel('Steps')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title(f'{name} Learning Curve')

    # Pareto frontier
    plt.subplot(1, 2, 2)
    plt.plot(kan_n_params, kan_test_rmse, 'o-', label='KAN')
    plt.plot(mlp_n_params, mlp_test_rmse, 's--', label='MLP')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Test RMSE')
    plt.legend()
    plt.title(f'{name} Performance Comparison')

    plt.tight_layout()
    plt.savefig(f'images/special_{name}.png')
    plt.close()