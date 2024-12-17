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


#Define model parameters 
config = {
    'in_channels': 1,
    'out_channels': 1,
    'base': 64,
    'kernel_size': 3,
    'activation_fn': nn.ReLU
}

#Training
train(config,train_loader, val_loader)

#Load model and make prediction
images_test=torch.stack(images_test)
# Load the saved model
model = load_model(config, 'final_model.pth')

# Make prediction
output = predict(model, images_test)
print(output.shape)
view_random_image(dataset, model)