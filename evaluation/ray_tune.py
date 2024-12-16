import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
from utils.helper import read_and_transform_images, ImageDataset, view_random_image
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from training.training import train, load_model, predict

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.optuna import OptunaSearch

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# import the images
image_dir='/Users/amir.taqieddin/Desktop/Hackathon/X-ray-PBD-main/x_ray_pbd_datasets/train/img'
images_org = read_and_transform_images(image_dir)
image_dir='/Users/amir.taqieddin/Desktop/Hackathon/X-ray-PBD-main/train_crop_data/img_crop'
images_crop = read_and_transform_images(image_dir)

image_dir='/Users/amir.taqieddin/Desktop/Hackathon/X-ray-PBD-main/x_ray_pbd_datasets/test/img'
images_test = read_and_transform_images(image_dir)

dataset = ImageDataset(images_org, images_crop)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

#Check size
for input_image, target_image in train_loader:
    print(f"Input batch shape: {input_image.shape}")   # [batch_size, 1, 224, 224]
    print(f"Target batch shape: {target_image.shape}") # [batch_size, 1, 224, 224]
    break  # Just show one batch for demonstration


search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "activation_fn": tune.choice(["ReLU", "LeakyReLU", "ELU"]),
        "in_channels": 1,
        "out_channels": 1,
        "base": tune.choice([32, 64, 128]),
        "kernel_size": tune.choice([3, 5, 7])
    }

# Define the scheduler and search algorithm
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=10,
    grace_period=1,
    reduction_factor=2
)

search_alg = OptunaSearch(metric="loss", mode="min")

# Run the hyperparameter search
analysis = tune.run(
    tune.with_parameters(train, train_loader=train_loader, val_loader=val_loader),
    resources_per_trial={"cpu": 1,"memory": 4 * 1024 * 1024 * 1024},
    config=search_space,
    num_samples=200,
    scheduler=scheduler,
    search_alg=search_alg
)

print("Best hyperparameters found were: ", analysis.best_config)