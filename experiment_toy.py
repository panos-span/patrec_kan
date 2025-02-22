from kan import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j0  # For Bessel function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import random

# Set random seed for reproducibility
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# Ensure deterministic behavior for cuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Initialize KAN with G=3
model = KAN(
    width=[1, 1], grid=3, k=3, seed=1, device=device
)  # Changed input dimension to 1


# Create dataset f(x) = J0(20x)
def f1(x):
    # Convert torch tensor to numpy, apply Bessel function, convert back to torch
    x_np = x.cpu().numpy() if x.is_cuda else x.numpy()
    result = j0(20 * x_np[:, [0]])  # Only use first column since input is 1D
    return torch.from_numpy(result).to(x.device).float()


def f2(x):
    # f(x) = exp(sin(pi * x1) + x2^2)
    return torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)


def f3(x):
    # f(x) = x1 * x2
    return x[:, [0]] * x[:, [1]]


def f4(x):
    inner_sum = torch.sin(np.pi * x / 2).pow(2).sum(dim=1, keepdim=True) / 100
    return torch.exp(inner_sum)


def f5(x):
    # Split input into pairs
    x1_x2 = x[:, [0, 1]]  # First two dimensions
    x3_x4 = x[:, [2, 3]]  # Last two dimensions

    # Calculate squared sums for each pair
    sum_squares_1 = (x1_x2**2).sum(dim=1, keepdim=True)
    sum_squares_2 = (x3_x4**2).sum(dim=1, keepdim=True)

    # Apply the full function
    inner = 0.5 * (torch.sin(np.pi * sum_squares_1) + torch.sin(np.pi * sum_squares_2))
    return torch.exp(inner)


widths = [
    [1, 1],
    [2, 1, 1],
    [2, 2, 1],
    [100, 1, 1],
    [4, 4, 2, 1],
]

vars = [1, 2, 2, 100, 4]
index = 1

# Iterate over functions
for f, width, var in zip([f1, f2, f3, f4, f5], widths, vars):
    dataset = create_dataset(f, n_var=var, device=device)
    input_dim = var

    grids = np.array([3, 5, 10, 20, 50, 100, 200, 500, 1000])
    train_losses = []
    test_losses = []
    steps = 200
    k = 3

    # Function to count parameters in MLP
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Define the MLP model
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim):
            super(MLP, self).__init__()
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

    # Function to find MLP architecture with approximately N parameters
    def find_mlp_architecture(
        input_dim, output_dim, target_params, max_depth=5, max_width=500
    ):
        for depth in range(1, max_depth + 1):
            for width in [10, 20, 50, 100, 200, 300, 400, 500]:
                hidden_dims = [width] * depth
                # Calculate number of parameters
                params = 0
                prev = input_dim
                for h in hidden_dims:
                    params += (prev * h) + h  # weights and biases
                    prev = h
                params += (prev * output_dim) + output_dim
                if params >= target_params:
                    return hidden_dims
        # If not found, return max configuration
        return [max_width] * max_depth

    # Lists to store MLP results
    mlp_n_params = []
    mlp_test_rmse = []
    mlt_train_rmnse = []

    for i in range(grids.shape[0]):
        if i == 0:
            model = KAN(
                width=width, grid=grids[i], k=k, seed=0, device=device
            )  # Changed input dimension
        if i != 0:
            model = model.refine(grids[i])
        results = model.fit(dataset, opt="LBFGS", steps=steps)
        train_losses += results["train_loss"]
        test_losses += results["test_loss"]

        # KAN number of parameters: assuming 3 * grid as per original code
        kan_n_params = 3 * grids[i]

        # Now, define an MLP with approximately kan_n_params parameters
        hidden_dims = find_mlp_architecture(
            input_dim=1, output_dim=1, target_params=kan_n_params
        )
        mlp = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1).to(device)

        # Count actual number of parameters
        actual_params = count_parameters(mlp)
        mlp_n_params.append(actual_params)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.LBFGS(
            mlp.parameters(),
            max_iter=steps,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        # Prepare data
        x_train, y_train = dataset["train_input"], dataset["train_label"]
        x_test, y_test = dataset["test_input"], dataset["test_label"]

        # Training function for LBFGS
        def closure():
            optimizer.zero_grad()
            outputs = mlp(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            return loss

        # Train MLP
        optimizer.step(closure)

        # Evaluate on test set
        with torch.no_grad():
            y_pred = mlp(x_test)
            test_loss = torch.sqrt(criterion(y_pred, y_test)).item()
            mlp_test_rmse.append(test_loss)

    # First plot: Loss vs. Steps for KANs
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.ylabel("RMSE")
    plt.xlabel("Step")
    plt.yscale("log")
    plt.title("KAN Training and Testing Loss vs Steps")
    plt.grid(True)

    # Second plot: Error vs. Number of Parameters
    plt.subplot(1, 2, 2)
    n_params = 3 * grids
    train_vs_G = train_losses[(steps - 1) :: steps]
    test_vs_G = test_losses[(steps - 1) :: steps]

    plt.plot(n_params, train_vs_G, marker="o", label="KAN Train")
    plt.plot(n_params, test_vs_G, marker="o", label="KAN Test")
    plt.plot(
        n_params, 100 * n_params ** (-4.0), ls="--", color="red", label=r"$N^{-4}$"
    )

    # Plot MLP results
    plt.plot(
        mlp_n_params,
        mlp_test_rmse,
        marker="s",
        linestyle="-",
        color="blue",
        label="MLP Test",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Number of Parameters")
    plt.ylabel("Test RMSE")
    plt.title("Error vs. Number of Parameters")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"images/results_{index}.png")
    index +=1
    #plt.show()
