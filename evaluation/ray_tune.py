import sys
import os
import ray
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

ray.init(runtime_env={"working_dir": project_root, "pip": ["torch", "ray[tune]", "optuna"]})


import torch
from utils.helper import ImageDataset, view_random_image, calculate_mean_std, NormalizedSubset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from training.training import train, load_model, predict

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# import the images
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# import the images
image_dir='/Users/amir.taqieddin/Desktop/Hackathon/X-ray-PBD-main/x_ray_pbd_datasets/train/img'
image_dir2='/Users/amir.taqieddin/Desktop/Hackathon/X-ray-PBD-main/train_crop_data/img_crop'
image_dir3='/Users/amir.taqieddin/Desktop/Hackathon/X-ray-PBD-main/x_ray_pbd_datasets/test/img'


dataset = ImageDataset(image_dir, image_dir2, mean=[0.5], std=[0.5])
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Calculate normalization parameters for training data only
train_mean, train_std = calculate_mean_std(train_dataset)
print(f"Training Data Mean: {train_mean}, Std: {train_std}")

# Wrap subsets with normalization
train_normalized = NormalizedSubset(train_dataset, mean=[train_mean], std=[train_std])
val_normalized = NormalizedSubset(val_dataset, mean=[train_mean], std=[train_std])

# Step 6: Create DataLoaders
train_loader = DataLoader(train_normalized, batch_size=16, shuffle=True)
val_loader = DataLoader(val_normalized, batch_size=16, shuffle=False)

search_space = {
    "lr": tune.loguniform(1e-7, 1e-2),
    "activation_fn": tune.choice([
        "ReLU",         # Rectified Linear Unit
        "LeakyReLU",    # Leaky ReLU
        "ELU",          # Exponential Linear Unit
        "PReLU",        # Parametric ReLU
        "SELU",         # Scaled ELU
        "GELU",         # Gaussian Error Linear Unit
        "SiLU",         # Swish activation
        "Tanh",         # Hyperbolic Tangent
        "Sigmoid",      # Sigmoid activation
        "Softplus"      # Softplus activation
    ]),
    "in_channels": 1,
    "out_channels": 1,
    "base": tune.choice([8, 16, 32, 64, 128, 256]),
    "kernel_size":  tune.choice([1, 3, 7, 9, 11]),
    "ratio": tune.uniform(0.05, 0.5),
    "stride": tune.choice([1, 2, 4,3, 5]),
    "num_kernels": 4,
    "temperature": tune.uniform(1, 40)  # Fixed range for tune.uniform
}

# Define the scheduler and search algorithm
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=20,
    grace_period=1,
    reduction_factor=2
)

search_alg = OptunaSearch(metric="loss", mode="min")

# Run the hyperparameter search
analysis = tune.run(
    tune.with_parameters(train, train_loader=train_loader, val_loader=val_loader, train_mean=train_mean, train_std=train_std),
    resources_per_trial={"cpu": 1,"memory": 4 * 1024 * 1024 * 1024},
    config=search_space,
    num_samples=100,
    scheduler=scheduler,
    search_alg=search_alg
)