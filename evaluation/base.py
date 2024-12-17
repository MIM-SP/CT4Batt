import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
from utils.helper import ImageDataset, view_random_image, calculate_mean_std, NormalizedSubset, read_and_resize
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from training.training import train, load_model, predict
from torch.utils.data import DataLoader, Subset
import random
import matplotlib.pyplot as plt

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
test_normalized = NormalizedSubset(read_and_resize(image_dir3), mean=[train_mean], std=[train_std])

# Create DataLoaders
train_loader = DataLoader(train_normalized, batch_size=16, shuffle=False)
val_loader = DataLoader(val_normalized, batch_size=16, shuffle=False)
test_normalized = DataLoader(test_normalized, batch_size=16, shuffle=False)

#Check size
#for input_image, target_image in train_loader:
#    print(f"Input batch shape: {input_image.shape}")   # [batch_size, 1, 256, 256]
#    print(f"Target batch shape: {target_image.shape}") # [batch_size, 1, 256, 256]
#    break  # Just show one batch for demonstration


print("Data preparation completed. Training and validation datasets are ready.")
for images_org, images_crop in train_loader:
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(images_org[0].permute(1, 2, 0).cpu().numpy())

    plt.subplot(1, 2, 2)
    plt.title("Cropped Image (Target)")
    plt.imshow(images_crop[0].permute(1, 2, 0).cpu().numpy())

    plt.show()
    break

#Define model parameters 
config = {
    'in_channels': 1,
    'out_channels': 1,
    'base': 16,
    'kernel_size': 11,
    'activation_fn': "Softplus",
    'lr':0.00241224 ,
    'ratio':0.320002 ,
    'stride':1,
    'num_kernels':3,
    'temperature':1.11906 
}

#Training
train(config,train_loader, val_loader, train_mean, train_std)

images_test = ImageDataset(
    [images_test[i][0] for i in range(len(train_dataset))],
    [images_test[i][1] for i in range(len(train_dataset))],
    mean=train_mean,
    std=train_std
)

# Load the saved model
model = load_model(config, 'final_model.pth')

# Make prediction
output = predict(model, images_test)
print(output.shape)
view_random_image(dataset, model)
